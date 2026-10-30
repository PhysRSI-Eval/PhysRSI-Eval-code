"""finepose step 2 (user 10-29): DINOv3 keypoint model. Backbone = DINOv3 ViT-B/16 (HF transformers DINOv3ViTModel, weights dinov3-vitb16 from
inhand-pose-20261026 v1pkg, sha256-checked), fine-tuned. Head = patch tokens (stride 16) -> 3-stage conv/upsample decoder to stride 4 -> per-keypoint
heatmap -> soft-argmax (same as kpnet3) + visibility logit from the CLS token. Data, split, augmentation (ColorJitter) and loss are kpnet3's
(imported unchanged), so the only difference to KPNet is the backbone/decoder.
  python dinokp.py train --data <dir> --dev-layouts ... --split-field seed --nk 8 --backbone <dinov3 dir> --out <ckpt> --gpu 0 --epochs 40 --seed 0"""
import argparse, json, math, random, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kpnet3 as K3  # noqa: E402


class DinoKP(nn.Module):
    def __init__(self, backbone, nk=8):
        super().__init__(); from transformers import AutoModel
        self.bb = AutoModel.from_pretrained(backbone, torch_dtype=torch.float32); C = self.bb.config.hidden_size; self.nreg = 1 + self.bb.config.num_register_tokens; self.nk = nk
        def up(i, o): return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.ReLU(inplace=True))
        self.dec = nn.Sequential(nn.Conv2d(C, 256, 1), nn.GroupNorm(8, 256), nn.ReLU(inplace=True), up(256, 128), up(128, 128)); self.head = nn.Conv2d(128, nk, 1); self.vis = nn.Linear(C, nk)

    def forward(self, x):
        B, _, H, W = x.shape; t = self.bb(pixel_values=x).last_hidden_state; cls, pt = t[:, 0], t[:, self.nreg:]
        g = pt.transpose(1, 2).reshape(B, -1, H // 16, W // 16); hm = self.head(self.dec(g)); _, Kk, h, w = hm.shape
        prob = F.softmax(hm.view(B, Kk, -1).float(), -1).view(B, Kk, h, w)
        xs = (torch.arange(w, device=x.device, dtype=torch.float32) + 0.5) * (W / w); ys = (torch.arange(h, device=x.device, dtype=torch.float32) + 0.5) * (H / h)
        u = (prob.sum(2) * xs).sum(-1); v = (prob.sum(3) * ys).sum(-1); return torch.stack([u, v], -1), self.vis(cls).float(), prob.view(B, Kk, -1).max(-1).values


@torch.no_grad()
def evaluate(net, recs, dev):
    net.eval(); ea, ek, eh = [], [], []
    for i in range(0, len(recs), 16):
        b = recs[i:i + 16]; x = torch.stack([K3.to_tensor(Image.open(r['path'])) for r in b]).to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16): p, _, _ = net(x)
        for r, pp in zip(b, p.float().cpu().numpy()):
            for k in range(len(r['in_image'])):
                if r['in_image'][k]: e = float(np.linalg.norm(pp[k] - np.array(r['outlets_uv'][k]))); ea.append(e); (ek if k < 5 else eh).append(e)
    net.train(); q = lambda e: dict(median=round(float(np.median(e)), 3), p90=round(float(np.percentile(e, 90)), 3), n=len(e)) if e else None
    return dict(median=q(ea)['median'], p90=q(ea)['p90'], n=len(ea), key=q(ek), keyhole=q(eh))


def train(a):
    dev_l = {int(x) for x in a.dev_layouts.split(',')}; dirs = a.data.split(','); tr = K3.load(dirs, exclude=dev_l, field=a.split_field); dv = K3.load(dirs, layouts=dev_l, field=a.split_field)
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed); dev = torch.device(f'cuda:{a.gpu}'); net = DinoKP(a.backbone, a.nk).to(dev)
    bb = list(net.bb.parameters()); hd = [p for n, p in net.named_parameters() if not n.startswith('bb.')]
    opt = torch.optim.AdamW([dict(params=bb, lr=a.lr_bb, weight_decay=0.05), dict(params=hd, lr=a.lr, weight_decay=1e-4)])
    dl = torch.utils.data.DataLoader(K3.DS(tr, True), batch_size=a.bs, shuffle=True, num_workers=8, drop_last=True); tot_steps = a.epochs * len(dl); warm = len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, tot_steps - warm)))); log = []; best = None
    def save(path, ep):
        torch.save(dict(state=net.state_dict(), nk=a.nk, kind='dinokp', backbone=a.backbone, dev_layouts=sorted(dev_l), n_train=len(tr), n_dev=len(dv), log=log, epoch=ep,
                        recipe='dinokp: DINOv3 ViT-B/16 finetune lr_bb %g / head %g, warmup+cosine, bf16, seed %d' % (a.lr_bb, a.lr, a.seed)), path)
    for ep in range(a.epochs):
        net.train(); tot = 0
        for x, uv, vis in dl:
            x, uv, vis = x.to(dev), uv.to(dev), vis.to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16): p, vl, _ = net(x)
            l = ((p - uv).abs().sum(-1) * vis).sum() / vis.sum().clamp(min=1) + F.binary_cross_entropy_with_logits(vl, vis)
            opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step(); tot += float(l)
        rec = dict(epoch=ep, train_loss=round(tot / len(dl), 3), dev_px=evaluate(net, dv, dev)); log.append(rec); print(json.dumps(rec), flush=True)
        if best is None or rec['dev_px']['median'] < best: best = rec['dev_px']['median']; save(a.out, ep)
    save(a.out + '.last.pt', a.epochs - 1); print(json.dumps(dict(best_dev_median_px=best)), flush=True)


class Detector:
    """RGB (H x W x 3 uint8, 640x480) -> [(u, v, visible_prob, peak)] x nk."""
    def __init__(self, ckpt, gpu=0, backbone=None):
        d = torch.load(ckpt, map_location='cpu', weights_only=False); self.dev = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
        self.net = DinoKP(backbone or d['backbone'], d['nk']).to(self.dev); self.net.load_state_dict(d['state']); self.net.eval(); self.nk = d['nk']

    @torch.no_grad()
    def __call__(self, img):
        x = K3.to_tensor(Image.fromarray(np.asarray(img, np.uint8)))[None].to(self.dev)
        with torch.autocast('cuda', dtype=torch.bfloat16): p, vl, pk = self.net(x)
        return [(float(p[0, k, 0]), float(p[0, k, 1]), float(torch.sigmoid(vl[0, k])), float(pk[0, k])) for k in range(self.nk)]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--data'); ap.add_argument('--dev-layouts', default=''); ap.add_argument('--out'); ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--bs', type=int, default=16); ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--lr-bb', type=float, default=2e-5)
    ap.add_argument('--split-field', default='seed'); ap.add_argument('--nk', type=int, default=8); ap.add_argument('--backbone', required=True); ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
    {'train': train}[a.cmd](a)
