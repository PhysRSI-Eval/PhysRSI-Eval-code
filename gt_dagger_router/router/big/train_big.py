"""ROUTER-BIG v1 trainer (self-contained: numpy + torch; runs on the cluster and on the H200).
Causal transformer over windows of W frames; input per frame = V (768 PCA DINOv3) + P (14 proprio); dense FFN or top-1 MoE (--experts).
Heads: task (42), stage (6), progress (sigmoid), success (logit), takeover (logit); masked losses. Train on the 'dev = False' groups,
report per-head numbers on the held-out groups.  python train_big.py --data big_v1.npz --d 512 --layers 8 --experts 0 --device cuda:0 --tag X"""
import argparse, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap = argparse.ArgumentParser(); ap.add_argument('--data', default='big_v1.npz'); ap.add_argument('--d', type=int, default=512); ap.add_argument('--layers', type=int, default=8)
ap.add_argument('--experts', type=int, default=0); ap.add_argument('--W', type=int, default=32); ap.add_argument('--steps', type=int, default=20000); ap.add_argument('--bs', type=int, default=64)
ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--device', default='cuda:0'); ap.add_argument('--tag', default='dev'); ap.add_argument('--save', default='')
a = ap.parse_args(); dev = torch.device(a.device); torch.manual_seed(0); rng = np.random.default_rng(0)
Z = np.load(a.data, allow_pickle=True); E = len(Z['V']); NT = 42
V, P = list(Z['V']), list(Z['P']); TASK = Z['task']; STG, PRG, SUC, TAK, SRC, DEV = list(Z['stage']), list(Z['prog']), Z['succ'], list(Z['take']), Z['src'], Z['dev'].astype(bool)
pm = np.concatenate([p for p in P if len(p)]).mean(0); ps = np.concatenate([p for p in P if len(p)]).std(0) + 1e-6
class MoE(nn.Module):
    def __init__(s, d, e):
        super().__init__(); s.g = nn.Linear(d, e); s.ex = nn.ModuleList([nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for _ in range(e)])
    def forward(s, x):
        w = F.softmax(s.g(x), -1); k = w.argmax(-1); out = torch.zeros_like(x)
        for i, ex in enumerate(s.ex):
            m = k == i
            if m.any(): out[m] = (ex(x[m]) * w[m][:, i:i + 1]).to(out.dtype)
        return out
class Block(nn.Module):
    def __init__(s, d, e):
        super().__init__(); s.n1 = nn.LayerNorm(d); s.att = nn.MultiheadAttention(d, 8, batch_first=True, dropout=0.1); s.n2 = nn.LayerNorm(d)
        s.ff = MoE(d, e) if e else nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(s, x, mask):
        h = s.n1(x); x = x + s.att(h, h, h, attn_mask=mask, need_weights=False)[0]; return x + s.ff(s.n2(x))
class Net(nn.Module):
    def __init__(s, d, L, e):
        super().__init__(); s.inp = nn.Linear(768 + 14, d); s.pos = nn.Embedding(a.W, d); s.blocks = nn.ModuleList([Block(d, e) for _ in range(L)]); s.n = nn.LayerNorm(d)
        s.task = nn.Linear(d, NT); s.stage = nn.Linear(d, 6); s.prog = nn.Linear(d, 1); s.succ = nn.Linear(d, 1); s.take = nn.Linear(d, 1)
    def forward(s, x):
        T = x.shape[1]; mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1); h = s.inp(x) + s.pos(torch.arange(T, device=x.device))[None]
        for b in s.blocks: h = b(h, mask)
        h = s.n(h); return s.task(h), s.stage(h), s.prog(h)[..., 0], s.succ(h)[..., 0], s.take(h)[..., 0]
def window(i, end=None):
    L = len(V[i]); end = rng.integers(1, L + 1) if end is None else end; st = max(0, end - a.W)
    x = np.c_[V[i][st:end].astype(np.float32), (P[i][st:end] - pm) / ps]; n = end - st
    lab = dict(task=np.full(n, TASK[i]), stage=STG[i][st:end].astype(int), prog=PRG[i][st:end], succ=np.full(n, SUC[i], np.float32), take=TAK[i][st:end])
    return x, lab, n
def batch(ids):
    xs, ls = [], []
    for i in ids: x, l, n = window(i); xs.append(x); ls.append(l)
    T = max(len(x) for x in xs); X = np.zeros((len(xs), T, 782), np.float32); M = np.zeros((len(xs), T), bool); out = {k: np.full((len(xs), T), -1 if k in ('task', 'stage') else np.nan, np.float32) for k in ('task', 'stage', 'prog', 'succ', 'take')}
    for b, (x, l) in enumerate(zip(xs, ls)):
        X[b, T - len(x):] = x; M[b, T - len(x):] = True
        for k in out: out[k][b, T - len(x):] = l[k]
    return torch.tensor(X, device=dev), torch.tensor(M, device=dev), {k: torch.tensor(v, device=dev) for k, v in out.items()}
tr = np.where(~DEV)[0]; srcs = sorted(set(SRC)); by = {s: tr[SRC[tr] == s] for s in srcs}
m = Net(a.d, a.layers, a.experts).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.05); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
t0 = time.time(); log = []
for step in range(a.steps):
    ids = np.concatenate([rng.choice(by[s], a.bs // len(srcs)) for s in srcs if len(by[s])])   # balanced over sources
    X, M, L = batch(ids)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        tk, sg, pr, sc, ta = m(X)
    loss = 0
    k = (L['task'] >= 0) & M; loss = loss + (F.cross_entropy(tk[k].float(), L['task'][k].long()) if k.any() else 0)
    k = (L['stage'] >= 0) & M; loss = loss + (F.cross_entropy(sg[k].float(), L['stage'][k].long()) if k.any() else 0)
    k = ~torch.isnan(L['prog']) & M; loss = loss + (F.l1_loss(torch.sigmoid(pr[k].float()), L['prog'][k]) if k.any() else 0)
    k = ~torch.isnan(L['succ']) & M; loss = loss + (F.binary_cross_entropy_with_logits(sc[k].float(), L['succ'][k]) if k.any() else 0)
    k = ~torch.isnan(L['take']) & M; loss = loss + (F.binary_cross_entropy_with_logits(ta[k].float(), L['take'][k], pos_weight=torch.tensor(4.0, device=dev)) if k.any() else 0)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
    if step % 1000 == 0: log.append(dict(step=step, loss=round(float(loss), 4), sec=round(time.time() - t0))); print(log[-1], flush=True)
# evaluation on held-out groups: predict every frame causally (window ending at the frame)
m.eval(); te = np.where(DEV)[0]; R = dict(tag=a.tag, args=vars(a), log=log); acc = {}
def auroc(y, p):
    y, p = np.asarray(y), np.asarray(p); pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg): return None
    r = np.argsort(np.argsort(np.r_[pos, neg])) + 1; return round(float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 4)
rec = {s: dict(task_ok=[], task_first=[], st_y=[], st_p=[], pr_e=[], sc_y=[], sc_p=[], sc_stage=[], tk_y=[], tk_p=[]) for s in srcs}
with torch.no_grad():
    for i in te:
        L_ = len(V[i]); xs = [np.c_[V[i][max(0, e - a.W):e].astype(np.float32), (P[i][max(0, e - a.W):e] - pm) / ps] for e in range(1, L_ + 1)]
        for c0 in range(0, L_, 256):
            xb = xs[c0:c0 + 256]; T = max(len(x) for x in xb); X = np.zeros((len(xb), T, 782), np.float32)
            for b, x in enumerate(xb): X[b, T - len(x):] = x
            with torch.autocast('cuda', dtype=torch.bfloat16): tk, sg, pr, sc, ta = m(torch.tensor(X, device=dev))
            tk, sg, pr, sc, ta = [v[:, -1].float().cpu().numpy() for v in (tk, sg, pr, sc, ta)]
            r = rec[SRC[i]]
            for j in range(len(xb)):
                f = c0 + j; r['task_ok'].append(int(tk[j].argmax() == TASK[i]) if TASK[i] >= 0 else None)
                if f == 0: r['task_first'].append(int(tk[j].argmax() == TASK[i]))
                if STG[i][f] >= 0: r['st_y'].append(int(STG[i][f])); r['st_p'].append(int(sg[j].argmax()))
                if not np.isnan(PRG[i][f]): r['pr_e'].append(abs(1 / (1 + np.exp(-pr[j])) - PRG[i][f]))
                if not np.isnan(SUC[i]): r['sc_y'].append(SUC[i]); r['sc_p'].append(float(sc[j])); r['sc_stage'].append(int(STG[i][f]))
                if not np.isnan(TAK[i][f]): r['tk_y'].append(TAK[i][f]); r['tk_p'].append(float(ta[j]))
for s, r in rec.items():
    out = dict(task_acc=round(float(np.mean([x for x in r['task_ok'] if x is not None])), 4) if r['task_ok'] else None, task_acc_first_frame=round(float(np.mean(r['task_first'])), 4) if r['task_first'] else None)
    if r['st_y']:
        y_, p_ = np.asarray(r['st_y']), np.asarray(r['st_p']); out['stage_f1'] = {c: round(float(2 * ((p_ == c) & (y_ == c)).sum() / max((p_ == c).sum() + (y_ == c).sum(), 1)), 3) for c in range(6) if (y_ == c).any()}; out['stage_acc'] = round(float((y_ == p_).mean()), 4)
    if r['pr_e']: out['progress_mae'] = round(float(np.mean(r['pr_e'])), 4)
    if r['sc_y'] and len(set(r['sc_y'])) > 1:
        out['success_auroc'] = auroc(r['sc_y'], r['sc_p']); ss = np.asarray(r['sc_stage']); out['success_auroc_by_stage'] = {int(c): auroc(np.asarray(r['sc_y'])[ss == c], np.asarray(r['sc_p'])[ss == c]) for c in sorted(set(ss)) if c >= 0}
    if r['tk_y']: out['takeover'] = dict(n=len(r['tk_y']), pos=int(sum(r['tk_y'])), auroc=auroc(r['tk_y'], r['tk_p']))
    acc[s] = out
R['dev'] = acc; json.dump(R, open('RESULT-big-%s.json' % a.tag, 'w'), indent=1); print(json.dumps(acc, indent=1))
if a.save: torch.save(dict(state_dict=m.state_dict(), args=vars(a), pm=pm, ps=ps), a.save)
