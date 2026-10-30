"""Stage model v2: + optional per-frame proprio (--prop, data with a P array) and a boundary-tolerant score (a frame counts as correct
if the predicted stage equals the true stage of any frame within +-TOL steps; frames are every 4 steps). v1 docstring follows.
K1 stage model (router v4 step 2): causal temporal transformer over per-frame DINOv3 features (head, left, right; [CLS, mean patch])
-> stage in {0 approach, 1 grasp, 2 lift, 3 transport, 4 align, 5 inserted/placed}. Grouped 5-fold CV by context. Self-contained (numpy +
torch only) so the same file runs on the H200: python train_stage.py --data stage_k1_v1.npz --d 512 --layers 8 --experts 4 --device cuda:0
  --experts > 0: token-wise top-1 mixture-of-experts feed-forward (switch style) in every layer.
Outputs RESULT-stage-<tag>.json: frame accuracy, per-stage F1, per task, confusion."""
import argparse, json, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap = argparse.ArgumentParser(); ap.add_argument('--data', default='stage_k1_v1.npz'); ap.add_argument('--d', type=int, default=256); ap.add_argument('--layers', type=int, default=4)
ap.add_argument('--experts', type=int, default=0); ap.add_argument('--epochs', type=int, default=60); ap.add_argument('--device', default='cuda:0'); ap.add_argument('--tag', default='dev'); ap.add_argument('--prop', action='store_true'); ap.add_argument('--tol-steps', type=int, default=5); a = ap.parse_args()
Z = np.load(a.data, allow_pickle=True); X, Y, META = list(Z['X']), list(Z['Y']), json.loads(str(Z['meta'])); PP = list(Z['P']) if a.prop else None
dev = torch.device(a.device); NS = 6
class MoE(nn.Module):
    def __init__(s, d, e):
        super().__init__(); s.g = nn.Linear(d, e); s.ex = nn.ModuleList([nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for _ in range(e)])
    def forward(s, x):
        w = F.softmax(s.g(x), -1); k = w.argmax(-1); out = torch.zeros_like(x)
        for i, ex in enumerate(s.ex):
            m = k == i
            if m.any(): out[m] = ex(x[m]) * w[m][:, i:i + 1]
        return out
class Block(nn.Module):
    def __init__(s, d, e):
        super().__init__(); s.n1 = nn.LayerNorm(d); s.att = nn.MultiheadAttention(d, 8, batch_first=True, dropout=0.1); s.n2 = nn.LayerNorm(d)
        s.ff = MoE(d, e) if e else nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(s, x, mask):
        h = s.n1(x); x = x + s.att(h, h, h, attn_mask=mask, need_weights=False)[0]; return x + s.ff(s.n2(x))
class Net(nn.Module):
    def __init__(s, d, L, e):
        super().__init__(); s.inp = nn.Linear(3 * 1536 + (14 if a.prop else 0), d); s.pos = nn.Embedding(512, d); s.blocks = nn.ModuleList([Block(d, e) for _ in range(L)]); s.out = nn.Linear(d, NS)
    def forward(s, x):
        T = x.shape[1]; mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        h = s.inp(x) + s.pos(torch.arange(T, device=x.device).clamp(max=511))[None]
        for b in s.blocks: h = b(h, mask)
        return s.out(h)
ctx = [m['ctx'] for m in META]; cs = sorted(set(ctx)); random.Random(20261030).shuffle(cs); fold = {c: i % 5 for i, c in enumerate(cs)}
mu = np.concatenate([x.reshape(len(x), -1) for x in X if len(x)]).astype(np.float32).mean(0); sd = np.concatenate([x.reshape(len(x), -1) for x in X if len(x)]).astype(np.float32).std(0) + 1e-6
def T(i):
    v = (X[i].reshape(len(X[i]), -1).astype(np.float32) - mu) / sd
    if PP is not None: v = np.concatenate([v, (PP[i] - pmu) / psd], 1)
    return torch.tensor(v[None], device=dev)
if PP is not None:
    _P = np.concatenate([p for p in PP if len(p)]); pmu, psd = _P.mean(0), _P.std(0) + 1e-6
P = {}
for k in range(5):
    tr = [i for i in range(len(X)) if fold[ctx[i]] != k and len(X[i])]; te = [i for i in range(len(X)) if fold[ctx[i]] == k and len(X[i])]
    cnt = np.bincount(np.concatenate([Y[i] for i in tr]).astype(int), minlength=NS); w = torch.tensor(1.0 / np.maximum(cnt, 1), dtype=torch.float32, device=dev); w = w / w.mean()
    torch.manual_seed(0); m = Net(a.d, a.layers, a.experts).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.05)
    for ep in range(a.epochs):
        m.train(); random.shuffle(tr)
        for i in tr:
            opt.zero_grad(); l = F.cross_entropy(m(T(i))[0], torch.tensor(Y[i].astype(np.int64), device=dev), weight=w); l.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        for i in te: P[i] = m(T(i))[0].argmax(-1).cpu().numpy()
    print('fold', k, 'done', flush=True)
yt = np.concatenate([Y[i] for i in sorted(P)]).astype(int); yp = np.concatenate([P[i] for i in sorted(P)])
def f1(c): tp = ((yp == c) & (yt == c)).sum(); return round(float(2 * tp / max((yp == c).sum() + (yt == c).sum(), 1)), 3)
tol = []
for i in sorted(P):
    st = np.asarray(META[i]['steps']); yy = Y[i].astype(int); pp = P[i]
    for j in range(len(yy)):
        near = yy[np.abs(st - st[j]) <= a.tol_steps]; tol.append(pp[j] in near)
ytol = np.where(np.asarray(tol), yp, yt)   # boundary-tolerant: a prediction within +-tol steps of a frame with that stage counts as correct
def f1t(c): tp = ((yp == c) & (ytol == c)).sum(); return round(float(2 * tp / max((yp == c).sum() + (ytol == c).sum(), 1)), 3)
conf = np.zeros((NS, NS), int); np.add.at(conf, (yt, yp), 1)
per_task = {}
for t in sorted({m['task'] for m in META}):
    ii = [i for i in sorted(P) if META[i]['task'] == t]; a_ = np.concatenate([Y[i] for i in ii]).astype(int); b_ = np.concatenate([P[i] for i in ii]); per_task[t] = dict(episodes=len(ii), acc=round(float((a_ == b_).mean()), 3))
R = dict(tag=a.tag, args=vars(a), frames=int(len(yt)), acc=round(float((yt == yp).mean()), 4), acc_tol=round(float(np.mean(tol)), 4), f1_tol={s_: f1t(c) for c, s_ in enumerate(['approach', 'grasp', 'lift', 'transport', 'align', 'inserted'])}, f1={s: f1(c) for c, s in enumerate(['approach', 'grasp', 'lift', 'transport', 'align', 'inserted'])},
         per_task=per_task, confusion=conf.tolist(), label_counts=np.bincount(yt, minlength=NS).tolist())
json.dump(R, open('RESULT-stage-%s.json' % a.tag, 'w'), indent=1); print(json.dumps({k: R[k] for k in ('acc', 'acc_tol', 'f1', 'f1_tol', 'per_task', 'label_counts')}))
