"""in-hand pose v4 training (PREREG-IP-v4 §6 + Amendment 1). One CAD-conditioned network for all objects (no class-ID embedding).
  python train_v4.py --cache DIR --seed S --out DIR [--epochs 30] [--gpu G] [--predict-only CKPT]
Training uses split TRAIN only; DEV is evaluated every epoch with the full solver; the DEV-best epoch (median rotation error) is saved.
After training, predictions (keypoints, scales, regression) are written for every row (all splits) to <out>/pred_s<seed>.npz."""
import argparse, json, math, sys, time, os
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fn
import torchvision
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_v4 as LB  # noqa: E402
import solver_v4 as SV  # noqa: E402

D = 192
DINO = Path(os.environ.get('PHYSRSI_DINO', '/lizekai/RSIws/experiments/inhand-pose-20261026/model_v1/pkg/v1pkg_1026/dinov3-vitb16'))


def rot6_to_mat(x):
    a, b = x[..., :3], x[..., 3:]; a = Fn.normalize(a, dim=-1); b = Fn.normalize(b - (a * b).sum(-1, keepdim=True) * a, dim=-1); c = torch.cross(a, b, dim=-1)
    return torch.stack([a, b, c], -1)


class Net(nn.Module):
    def __init__(self, arch='resnet18', unfreeze=4):
        super().__init__()
        self.arch = arch
        if arch == 'dinov3':  # DINOv3-B/16 (same local snapshot as v3), last 4 blocks trainable; stride 16
            from transformers import AutoModel
            self.vit = AutoModel.from_pretrained(str(DINO), torch_dtype=torch.float32); self.n_reg = self.vit.config.num_register_tokens
            for q in self.vit.parameters(): q.requires_grad_(False)
            for blk in list(self.vit.layer)[-unfreeze:]:
                for q in blk.parameters(): q.requires_grad_(True)
            for q in self.vit.norm.parameters(): q.requires_grad_(True)
            self.fuse = nn.Conv2d(768, D, 1)
        else:
            m = getattr(torchvision.models, arch)(weights=None); ck = dict(resnet18='resnet18-f37072fd.pth', resnet34='resnet34-b627a593.pth')[arch]
            try: m.load_state_dict(torch.load(Path.home() / '.cache/torch/hub/checkpoints' / ck, map_location='cpu'))
            except Exception as e: print('no imagenet weights', e)  # noqa: BLE001
            self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2); self.l3 = m.layer3
            self.fuse = nn.Conv2d(128 + 256, D, 1)
        self.stride = 16 if arch == 'dinov3' else 8; self.pos = nn.Sequential(nn.Linear(6, D), nn.GELU(), nn.Linear(D, D)); self.view = nn.Embedding(2, D)
        self.pn = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 128), nn.GELU(), nn.Linear(128, D))
        self.qkp = nn.Sequential(nn.Linear(3, D), nn.GELU(), nn.Linear(D, D));
        self.prop = nn.Sequential(nn.Linear(13, D), nn.GELU(), nn.Linear(D, D))
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(D, 6, 4 * D, 0.0, batch_first=True, norm_first=True), 3)
        self.wq = nn.Linear(D, D); self.wk = nn.Linear(D, D); self.sig = nn.Linear(D, 2); self.pose = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, 9))
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)); self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def feats(self, x):
        x = (x - self.mean) / self.std
        if self.arch == 'dinov3':
            h, w = x.shape[-2] // 16, x.shape[-1] // 16; t = self.vit(pixel_values=x).last_hidden_state[:, 1 + self.n_reg:]
            return self.fuse(t.transpose(1, 2).reshape(x.shape[0], -1, h, w))
        f2 = self.stem(x); f3 = self.l3(f2)
        return self.fuse(torch.cat([f2, Fn.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)], 1))

    def forward(self, wimg, himg, rw, ow, rh, oh, kp, pts, prop):
        B = wimg.shape[0]; fw = self.feats(wimg); fh = self.feats(himg); Tw = fw.flatten(2).transpose(1, 2); Th = fh.flatten(2).transpose(1, 2)
        Tw = Tw + self.pos(torch.cat([rw, ow[:, None].expand(-1, rw.shape[1], -1) / 0.3], -1)) + self.view.weight[0]
        Th = Th + self.pos(torch.cat([rh, oh[:, None].expand(-1, rh.shape[1], -1) / 0.3], -1)) + self.view.weight[1]
        shape = self.pn(pts).max(1).values; q = self.qkp(kp) + shape[:, None] + self.prop(prop)[:, None]
        q = self.dec(q, torch.cat([Tw, Th], 1)); qq = self.wq(q) / math.sqrt(D)
        lw = torch.einsum('bkd,btd->bkt', qq, self.wk(Tw)); lh = torch.einsum('bkd,btd->bkt', qq, self.wk(Th))
        sig = self.sig(q); pose = self.pose(torch.cat([q.mean(1), shape], -1))
        return lw, lh, sig, pose, fw.shape[-2:], fh.shape[-2:]


def local_uv(logits, hw, stride=8, r=2):
    """inference: soft-argmax in a (2r+1)^2 window around the heatmap peak (a multimodal heatmap is not averaged across modes)."""
    p = logits.float().softmax(-1); H, W = hw; B, K, _ = p.shape; pk = p.argmax(-1); py, px = pk // W, pk % W
    ys, xs = torch.meshgrid(torch.arange(H, device=p.device), torch.arange(W, device=p.device), indexing='ij'); ys, xs = ys.flatten(), xs.flatten()
    m = ((ys[None, None] - py[..., None]).abs() <= r) & ((xs[None, None] - px[..., None]).abs() <= r); q = p * m; q = q / q.sum(-1, keepdim=True).clamp_min(1e-12)
    g = torch.stack([(xs + 0.5) * stride, (ys + 0.5) * stride], -1).float(); return q @ g


def soft_uv(logits, hw, stride=8):
    p = logits.float().softmax(-1); H, W = hw; ys, xs = torch.meshgrid(torch.arange(H, device=p.device), torch.arange(W, device=p.device), indexing='ij')
    g = torch.stack([(xs.flatten() + 0.5) * stride, (ys.flatten() + 0.5) * stride], -1).float(); return p @ g, p


class Data:
    def __init__(self, cache, dev, st=8):
        C = Path(cache); self.rows = [json.loads(l) for l in open(C / 'samples_v4.jsonl')]; print('loading images', len(self.rows), flush=True)
        self.W = np.load(C / 'wrist.npy'); self.H = np.load(C / 'head.npy'); M = np.load(C / 'meta.npz'); self.M = {k: M[k] for k in M.files}
        cad = LB.load_cad(); self.cad = cad; N = len(self.rows); self.dev = dev
        self.kp = np.zeros((N, 16, 3), np.float32); self.pts = np.zeros((N, 512, 3), np.float32); self.diag = np.zeros(N, np.float32); self.ctr = np.zeros((N, 3)); self.axes = np.zeros((N, 3, 3))
        self.axis = np.zeros((N, 3)); self.axial = np.zeros(N, bool); self.kpm = np.zeros((N, 16, 3))
        for i, r in enumerate(self.rows):
            c = cad[r['variant']]; s = c['diag']; self.kp[i] = (c['kp'] - c['ctr']) / s; self.pts[i] = (c['pts'] - c['ctr']) / s; self.diag[i] = s; self.ctr[i] = c['ctr']; self.axes[i] = c['axes']; self.kpm[i] = c['kp']
            if c['axial']: self.axial[i] = True; self.axis[i] = c['axis']
        rw, ow = LB.rays_wrist(st); self.rw = torch.tensor(rw.reshape(-1, 3), dtype=torch.float32); self.ow = torch.tensor(ow, dtype=torch.float32)
        self.rh = np.zeros((N, (LB.HS // st) ** 2, 3), np.float32); self.oh = np.zeros((N, 3), np.float32)
        for i in range(N):
            a, b = LB.rays_head(self.M['L'][i].astype(float), self.M['crop'][i].astype(float), st); self.rh[i] = a.reshape(-1, 3); self.oh[i] = b
        L = self.M['L']; self.prop = np.concatenate([L[:, :3, :3].reshape(N, 9), L[:, :3, 3], self.M['grip'][:, None]], 1).astype(np.float32)
        uvw, zw, uvh = self.M['uvw'], self.M['zw'], self.M['uvh']
        self.mw = (zw > 0.02) & (uvw[..., 0] > -4) & (uvw[..., 0] < 324) & (uvw[..., 1] > -4) & (uvw[..., 1] < 244)
        self.mh = (uvh[..., 0] > -4) & (uvh[..., 0] < 164) & (uvh[..., 1] > -4) & (uvh[..., 1] < 164)
        G = self.M['G']; self.Rt = G[:, :3, :3].astype(np.float32); self.tt = ((G[:, :3, 3] - LB.TCP) / 0.1).astype(np.float32)

    def idx(self, split): return np.array([i for i, r in enumerate(self.rows) if r['split'] == split])

    def batch(self, ii, aug=False):
        d = self.dev; t = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=d)  # noqa: E731
        w = t(self.W[ii]).permute(0, 3, 1, 2).float() / 255; h = t(self.H[ii]).permute(0, 3, 1, 2).float() / 255
        if aug:
            B = len(ii); g = lambda: (1 + 0.25 * (torch.rand(B, 1, 1, 1, device=d) - 0.5) * 2)  # noqa: E731
            w = (w * g() + 0.08 * (torch.rand(B, 3, 1, 1, device=d) - 0.5)).clamp(0, 1); h = (h * g() + 0.08 * (torch.rand(B, 3, 1, 1, device=d) - 0.5)).clamp(0, 1)
        return dict(w=w, h=h, rw=self.rw.to(d)[None].expand(len(ii), -1, -1), ow=self.ow.to(d)[None].expand(len(ii), -1), rh=t(self.rh[ii]), oh=t(self.oh[ii]),
                    kp=t(self.kp[ii]), pts=t(self.pts[ii]), prop=t(self.prop[ii]), uvw=t(self.M['uvw'][ii]), uvh=t(self.M['uvh'][ii]), mw=t(self.mw[ii]), mh=t(self.mh[ii]),
                    Rt=t(self.Rt[ii]), tt=t(self.tt[ii]), axial=t(self.axial[ii]), axis=t(self.axis[ii]).float())


LOGITS = {}


def run(net, b):
    lw, lh, sig, pose, hw, hh = net(b['w'], b['h'], b['rw'], b['ow'], b['rh'], b['oh'], b['kp'], b['pts'], b['prop']); LOGITS['w'], LOGITS['h'] = lw, lh
    st = net.stride; uvw, pw = soft_uv(lw, hw, st); uvh, ph = soft_uv(lh, hh, st); return uvw, uvh, pw, ph, sig.float(), pose.float(), hw, hh


def losses(out, b, st=8):
    uvw, uvh, pw, ph, sig, pose, hw, hh = out
    bw = sig[..., 0].clamp(-3, 5).exp() * 2; bh = sig[..., 1].clamp(-3, 5).exp() * 2  # px scale (wrist px, head crop px)
    ew = (uvw - b['uvw']).abs().sum(-1); eh = (uvh - b['uvh']).abs().sum(-1)
    lk = ((ew / bw + torch.log(bw)) * b['mw']).sum() / b['mw'].sum().clamp_min(1) + ((eh / bh + torch.log(bh)) * b['mh']).sum() / b['mh'].sum().clamp_min(1)
    def ce(p, uv, m, hwv):
        H, W = hwv; ys, xs = torch.meshgrid(torch.arange(H, device=p.device), torch.arange(W, device=p.device), indexing='ij')
        g = torch.stack([(xs.flatten() + 0.5) * st, (ys.flatten() + 0.5) * st], -1).float(); d2 = ((g[None, None] - uv[:, :, None]) ** 2).sum(-1) / (2 * float(st) ** 2)
        tgt = torch.softmax(-d2, -1); return (-(tgt * torch.log(p.clamp_min(1e-9))).sum(-1) * m).sum() / m.sum().clamp_min(1)
    lc = ce(pw, b['uvw'], b['mw'], hw) + ce(ph, b['uvh'], b['mh'], hh)
    Rp = rot6_to_mat(pose[:, :6]); tp = pose[:, 6:]
    geo = ((Rp.transpose(1, 2) @ b['Rt']).diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2; lr_full = 1 - geo
    ax_p = (Rp @ b['axis'][..., None])[..., 0]; ax_t = (b['Rt'] @ b['axis'][..., None])[..., 0]; lr_ax = 1 - (ax_p * ax_t).sum(-1)
    lr = torch.where(b['axial'], lr_ax, lr_full).mean(); lt = (tp - b['tt']).abs().sum(-1).mean()
    return lk + 0.5 * lc + lr + lt, dict(kp=float(lk), ce=float(lc), rot=float(lr), tr=float(lt))


@torch.no_grad()
def predict(net, data, ii, bs=64, local=False):
    net.eval(); O = dict(uvw=[], uvh=[], bw=[], bh=[], R0=[], t0=[])
    for s in range(0, len(ii), bs):
        b = data.batch(ii[s:s + bs])
        with torch.autocast('cuda', dtype=torch.bfloat16): out = run(net, b)
        uvw, uvh, pw, ph, sig, pose, hw, hh = out
        if local:
            lw, lh = LOGITS['w'], LOGITS['h']; uvw, uvh = local_uv(lw, hw, net.stride), local_uv(lh, hh, net.stride)
        O['uvw'].append(uvw.cpu()); O['uvh'].append(uvh.cpu()); O['bw'].append((sig[..., 0].clamp(-3, 5).exp() * 2).cpu()); O['bh'].append((sig[..., 1].clamp(-3, 5).exp() * 2).cpu())
        R0 = rot6_to_mat(pose[:, :6]); t0 = pose[:, 6:] * 0.1 + torch.tensor(LB.TCP, device=pose.device, dtype=torch.float32)
        O['R0'].append(R0.cpu()); O['t0'].append(t0.cpu())
    net.train(); return {k: torch.cat(v).float().numpy() for k, v in O.items()}


def solve_np(data, ii, P, dev, bs=256):
    """run the solver on predictions P for rows ii; returns R, t, cost, flip margin."""
    hc = LB.head_cam(); wr = dict(R_LC=LB.R_LC, T_LC=LB.T_LC, fx=LB.FXW, cx=LB.CXW, cy=LB.CYW); out = dict(R=[], t=[], c=[], m=[])
    for s in range(0, len(ii), bs):
        j = ii[s:s + bs]; sl = slice(s, s + bs); f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64, device=dev)  # noqa: E731
        geo = SV.Geo(data.M['L'][j].astype(float), data.M['crop'][j].astype(float), hc, wr, dev)
        R, t, c, m = SV.solve(geo, f(data.kpm[j]), f(data.ctr[j]), f(data.axes[j]), f(data.axis[j]), torch.as_tensor(data.axial[j], device=dev),
                              f(P['uvw'][sl]), f(P['bw'][sl]), f(P['uvh'][sl]), f(P['bh'][sl]), f(P['R0'][sl]), f(P['t0'][sl]))
        out['R'].append(R.cpu().numpy()); out['t'].append(t.cpu().numpy()); out['c'].append(c.cpu().numpy()); out['m'].append(m.cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def errors(data, ii, R, t):
    G = data.M['G'][ii]; pe = np.linalg.norm(t - G[:, :3, 3], axis=1) * 1000; re = np.zeros(len(ii))
    for n, i in enumerate(ii): re[n] = LB.rot_err_deg(R[n], G[n, :3, :3], data.axis[i] if data.axial[i] else None)
    return pe, re


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cache', required=True); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--bs', type=int, default=48); ap.add_argument('--gpu', type=int, default=0); ap.add_argument('--max-steps', type=int, default=0); ap.add_argument('--arch', default='resnet18'); ap.add_argument('--balance', type=float, default=0.0); ap.add_argument('--local', type=int, default=1); ap.add_argument('--unfreeze', type=int, default=4)
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device('cuda:%d' % a.gpu); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = Data(a.cache, dev, 16 if a.arch == 'dinov3' else 8); tr = data.idx('TRAIN'); dv = data.idx('DEV'); print('train', len(tr), 'dev', len(dv), flush=True)
    net = Net(a.arch, a.unfreeze).to(dev); bb = [p for n, p in net.named_parameters() if n.startswith(('stem', 'l3', 'vit')) and p.requires_grad]; hd = [p for n, p in net.named_parameters() if not n.startswith(('stem', 'l3', 'vit'))]
    opt = torch.optim.AdamW([dict(params=bb, lr=1e-4), dict(params=hd, lr=3e-4)], weight_decay=1e-4); spe = len(tr) // a.bs; total = spe * a.epochs; warm = spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    best = 1e9; log = open(out / ('log_s%d.jsonl' % a.seed), 'a'); step = 0
    for ep in range(a.epochs):
        if a.balance:  # category-balanced resampling: weight ~ count^-balance
            cc = np.array([data.rows[i]['cat'] for i in tr]); u, inv, cnt = np.unique(cc, return_inverse=True, return_counts=True); w = cnt[inv] ** -a.balance; perm = np.random.choice(tr, len(tr), p=w / w.sum())
        else: perm = np.random.permutation(tr)
        t0 = time.time(); agg = {}
        for s in range(spe):
            b = data.batch(np.sort(perm[s * a.bs:(s + 1) * a.bs]), aug=True)
            with torch.autocast('cuda', dtype=torch.bfloat16): o = run(net, b)
            loss, parts = losses(o, b, net.stride); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); step += 1
            for k, v in parts.items(): agg[k] = agg.get(k, 0) + v / spe
            if a.max_steps and step >= a.max_steps: break
        P = predict(net, data, dv, local=bool(a.local)); S = solve_np(data, dv, P, dev); pe, re = errors(data, dv, S['R'], S['t'])
        kpw = np.abs(P['uvw'] - data.M['uvw'][dv]).sum(-1)[data.mw[dv]]; kph = np.abs(P['uvh'] - data.M['uvh'][dv]).sum(-1)[data.mh[dv]]
        rec = dict(epoch=ep, train=agg, dev_pos_med=float(np.median(pe)), dev_rot_med=float(np.median(re)), dev_flip=float(np.mean(re > 90)), dev_kpw_px=float(np.median(kpw)), dev_kph_px=float(np.median(kph)), s=round(time.time() - t0))
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
        if rec['dev_rot_med'] < best: best = rec['dev_rot_med']; torch.save(dict(state=net.state_dict(), epoch=ep, dev=rec), out / ('best_s%d.pt' % a.seed))
        if a.max_steps and step >= a.max_steps: break
    ck = torch.load(out / ('best_s%d.pt' % a.seed), map_location=dev); net.load_state_dict(ck['state'])
    P = predict(net, data, np.arange(len(data.rows)), local=bool(a.local)); np.savez(out / ('pred_s%d.npz' % a.seed), **P); print('saved preds, best epoch', ck['epoch'], flush=True)


if __name__ == '__main__':
    main()
