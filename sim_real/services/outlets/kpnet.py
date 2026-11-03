"""loc1 outlet keypoint network (lane C): input one 640x480 wrist RGB image; output, for each of the 3 outlets (fixed order
along the strip), a sub-pixel image position (soft-argmax over a stride-4 heatmap, in full-resolution pixels) and a
visibility logit. ResNet-18 encoder (ImageNet weights if available, else random init), 3-level upsampling decoder.
Training labels come from simulator truth projections (training only); deployment input is the RGB image alone.
  train: python kpnet.py train --data <dir>[,<dir>] --dev-layouts 16,17 --out <ckpt.pt> --gpu N --epochs 40
"""
import argparse, json, math, random, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
W, H, NK, STRIDE = 640, 480, 3, 4


class KPNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        import torchvision
        try:
            m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None); self.pretrained = pretrained
        except Exception:  # noqa: BLE001  (no network for the weights)
            m = torchvision.models.resnet18(weights=None); self.pretrained = False
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool); self.l1, self.l2, self.l3, self.l4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.lat4, self.lat3, self.lat2, self.lat1 = nn.Conv2d(512, 128, 1), nn.Conv2d(256, 128, 1), nn.Conv2d(128, 128, 1), nn.Conv2d(64, 128, 1)
        self.head = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, NK, 1))
        self.vis = nn.Linear(512, NK)

    def forward(self, x):
        c1 = self.l1(self.stem(x)); c2 = self.l2(c1); c3 = self.l3(c2); c4 = self.l4(c3)
        p = self.lat4(c4); p = F.interpolate(p, size=c3.shape[-2:], mode='bilinear', align_corners=False) + self.lat3(c3)
        p = F.interpolate(p, size=c2.shape[-2:], mode='bilinear', align_corners=False) + self.lat2(c2)
        p = F.interpolate(p, size=c1.shape[-2:], mode='bilinear', align_corners=False) + self.lat1(c1)
        hm = self.head(p); B, K, h, w = hm.shape; prob = F.softmax(hm.view(B, K, -1), -1).view(B, K, h, w)
        xs = (torch.arange(w, device=x.device, dtype=torch.float32) + 0.5) * (W / w); ys = (torch.arange(h, device=x.device, dtype=torch.float32) + 0.5) * (H / h)
        u = (prob.sum(2) * xs).sum(-1); v = (prob.sum(3) * ys).sum(-1); peak = prob.view(B, K, -1).max(-1).values
        return torch.stack([u, v], -1), self.vis(F.adaptive_avg_pool2d(c4, 1).flatten(1)), peak


def to_tensor(img):
    return (torch.from_numpy(np.asarray(img.convert('RGB'), np.float32) / 255.0).permute(2, 0, 1) - MEAN) / STD


def load(dirs, layouts=None, exclude=None):
    out = []
    for d in dirs:
        for f in sorted(Path(d).glob('L*-s*.jsonl')):
            for l in open(f):
                r = json.loads(l)
                if layouts is not None and r['layout'] not in layouts: continue
                if exclude is not None and r['layout'] in exclude: continue
                r['path'] = str(Path(d) / 'images' / r['image']); out.append(r)
    return out


class DS(torch.utils.data.Dataset):
    def __init__(self, recs, aug):
        self.r, self.aug = recs, aug

    def __len__(self):
        return len(self.r)

    def __getitem__(self, i):
        r = self.r[i]; img = Image.open(r['path'])
        if self.aug:
            from torchvision.transforms import ColorJitter
            img = ColorJitter(0.3, 0.3, 0.3, 0.03)(img)
        uv = torch.tensor([u if u is not None else [0.0, 0.0] for u in r['outlets_uv']], dtype=torch.float32)
        vis = torch.tensor([float(x) for x in r['in_image']])
        return to_tensor(img), uv, vis


def train(a):
    dev_l = {int(x) for x in a.dev_layouts.split(',')} if a.dev_layouts else set()
    dirs = a.data.split(','); tr = load(dirs, exclude=dev_l); dv = load(dirs, layouts=dev_l) if dev_l else []
    torch.manual_seed(0); random.seed(0); dev = torch.device(f'cuda:{a.gpu}')
    net = KPNet().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    dl = torch.utils.data.DataLoader(DS(tr, True), batch_size=a.bs, shuffle=True, num_workers=6, drop_last=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl))
    log = []
    for ep in range(a.epochs):
        net.train(); tot = 0
        for x, uv, vis in dl:
            x, uv, vis = x.to(dev), uv.to(dev), vis.to(dev); p, vl, _ = net(x)
            l = ((p - uv).abs().sum(-1) * vis).sum() / vis.sum().clamp(min=1) + F.binary_cross_entropy_with_logits(vl, vis)
            opt.zero_grad(); l.backward(); opt.step(); sched.step(); tot += float(l)
        rec = dict(epoch=ep, train_loss=round(tot / len(dl), 3))
        if dv and (ep % 5 == 4 or ep == a.epochs - 1):
            rec['dev_px'] = evaluate(net, dv, dev)
        log.append(rec); print(json.dumps(rec), flush=True)
    torch.save(dict(state=net.state_dict(), pretrained=net.pretrained, train_layouts=sorted({r['layout'] for r in tr}), dev_layouts=sorted(dev_l),
                    n_train=len(tr), n_dev=len(dv), log=log), a.out)


@torch.no_grad()
def evaluate(net, recs, dev):
    net.eval(); errs = []
    for i in range(0, len(recs), 16):
        b = recs[i:i + 16]; x = torch.stack([to_tensor(Image.open(r['path'])) for r in b]).to(dev); p, _, _ = net(x)
        for r, pp in zip(b, p.cpu().numpy()):
            for k in range(NK):
                if r['in_image'][k]: errs.append(float(np.linalg.norm(pp[k] - np.array(r['outlets_uv'][k]))))
    net.train(); return dict(median=round(float(np.median(errs)), 3), p90=round(float(np.percentile(errs, 90)), 3), n=len(errs))


class Detector:
    """Deployment wrapper: RGB image (H x W x 3 uint8) -> [(u, v, visible_prob, peak)] for the 3 outlets."""
    def __init__(self, ckpt, gpu=0):
        d = torch.load(ckpt, map_location='cpu'); self.dev = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
        self.net = KPNet(pretrained=False).to(self.dev); self.net.load_state_dict(d['state']); self.net.eval()

    @torch.no_grad()
    def __call__(self, img):
        x = to_tensor(Image.fromarray(np.asarray(img, np.uint8)))[None].to(self.dev); p, vl, pk = self.net(x)
        return [(float(p[0, k, 0]), float(p[0, k, 1]), float(torch.sigmoid(vl[0, k])), float(pk[0, k])) for k in range(NK)]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--data'); ap.add_argument('--dev-layouts', default='')
    ap.add_argument('--out'); ap.add_argument('--gpu', type=int, default=0); ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3); a = ap.parse_args()
    {'train': train}[a.cmd](a)
