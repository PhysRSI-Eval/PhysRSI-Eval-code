"""One router stage model for the taxonomy's shared skills on correct trajectories (dispatcher 10-31): stage + progress heads on all 9
official GT tasks, GT.md split 80 / 10 / 10 (big_v3 'split'), per task and per category (TASK-SKILL-TAXONOMY-20261030: K1 plug / key /
tubes / coin, K3 pour, K5 Xylophone, K6 push_T, K7 make_kong / organize_table).
Shared stages, PROPRIO-DERIVED (labels only, hindsight allowed): per arm, a hold = gripper reading < 0.9 from close step c to open step r;
  1 grasp-lift : held and the EEF raised < 3 cm since c (and before it first reaches 3 cm);
  2 transport  : held, after grasp-lift, EEF xy > 5 cm from the release xy (EEF xy at r);
  3 align/act  : held, within 5 cm (xy) of the release xy;
  4 release    : < 25 steps (1 s at 25 Hz) after an open that ended a hold, or from the last open to the episode end;
  0 reach      : otherwise.  Frame label = the held arm's stage (higher of the two if both hold), else release, else reach.
--act_defs 1 (one change): K3 act = held and EEF rotated > 45 deg from the grasp orientation (tilt); K6 act = gripper closed + EEF within
2 cm of its lowest z + planar speed > 1 mm / step (else transport). Other categories unchanged.
Caveats: no object truth (GT.md 8) -> 'held' = gripper closed (push_T closes its gripper to push; pour holds at reading ~0.68 < 0.9);
'act' for strike / push is the last 5 cm before the gripper opens, not the contact itself.
Model = train_stage_tf's no-cond causal transformer (d256 L4, window 32) on PCA-768 DINOv3 + 14 proprio (big_v3 demo rows, same order as
official/feat).  python train_shared_stage.py --seed 0 --device cuda:0 -> RESULT-shared-stage-s0.json"""
import argparse, glob, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--steps', type=int, default=10000); ap.add_argument('--device', default='cuda:0'); ap.add_argument('--act_defs', type=int, default=0); ap.add_argument('--tag', default='')
a = ap.parse_args(); dev = torch.device(a.device); H = Path(__file__).resolve().parent; W, NS = 32, 5
NAMES = ['reach', 'grasp-lift', 'transport', 'align/act', 'release']
CAT = {'plug_in_charger': 'K1', 'insert_key': 'K1', 'insert_tubes': 'K1', 'deposit_coin': 'K1', 'pour_liquid_into_cup': 'K3', 'play_Xylophone': 'K5', 'push_T': 'K6', 'make_kong': 'K7', 'organize_table': 'K7'}
def qang(q, q0): return np.degrees(2 * np.arccos(np.clip(np.abs((q * q0).sum(-1)) / (np.linalg.norm(q, axis=-1) * np.linalg.norm(q0) + 1e-9), 0, 1)))   # order-agnostic (same convention both)
def arm_stage(g, z, xy, t, q=None, mode=''):
    L = len(g); st = np.zeros(L, int); closed = g < 0.9; k = 0
    v = np.r_[0, np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(np.diff(t), 1)] * 1000; zmin = z.min()   # planar speed mm / step
    while k < L:
        if not closed[k]: k += 1; continue
        c = k
        while k < L and closed[k]: k += 1
        r = k; rel = xy[min(r, L) - 1]; lifted = False
        for j in range(c, r):
            lifted = lifted or (z[j] - z[c]) >= 0.03
            if mode == 'tilt': st[j] = 3 if qang(q[j], q[c]) > 45 else (1 if not lifted else 2)   # K3: act = tilted > 45 deg from the grasp orientation
            elif mode == 'push': st[j] = 3 if (z[j] - zmin < 0.02 and v[j] > 1.0) else 2   # K6: act = closed + EEF within 2 cm of its lowest + planar motion > 1 mm / step
            else: st[j] = 1 if not lifted else (3 if np.linalg.norm(xy[j] - rel) <= 0.05 else 2)
        if r < L:
            nxt = next((j for j in range(r, L) if closed[j]), L)
            for j in range(r, nxt): st[j] = 4 if (t[j] - t[r] < 25 or nxt == L) else 0
    return st
def labels(f):
    z = np.load(f, allow_pickle=True); t = z['t']
    mode = {'K3': 'tilt', 'K6': 'push'}.get(CAT.get(str(z['task'])), '') if a.act_defs else ''
    S = [arm_stage(z['%s_ee_joint_states' % s][:, 0], z['%s_ee_poses' % s][:, 2], z['%s_ee_poses' % s][:, :2], t, z['%s_ee_poses' % s][:, 3:], mode) for s in ('left', 'right')]
    held = [np.isin(s_, (1, 2, 3)) for s_ in S]
    lab = np.where(held[0] | held[1], np.maximum(np.where(held[0], S[0], 0), np.where(held[1], S[1], 0)), np.where((S[0] == 4) | (S[1] == 4), 4, 0))
    return lab, str(z['task']), t / max(int(z['T']) - 1, 1)
files = [f for f in sorted(glob.glob(str(H.parent / 'official/feat/*/*.npz'))) if len(np.load(f, allow_pickle=True)['X'])]
Z = np.load(H / 'big_v3.npz', allow_pickle=True); src = Z['src']; sel = np.where(src == 'demo')[0]; ZV, ZP, ZSP = Z['V'], Z['P'], Z['split']
assert len(sel) == len(files), (len(sel), len(files))
V, P, S, G, TK, SP = [], [], [], [], [], []
for i, f in zip(sel, files):
    lab, task, prog = labels(f); v = ZV[i].astype(np.float32); assert len(v) == len(lab)
    V.append(v); P.append(ZP[i].astype(np.float32)); S.append(lab); G.append(prog.astype(np.float32)); TK.append(task); SP.append(int(ZSP[i]))
SP = np.array(SP); TK = np.array(TK); tr = np.where(SP == 0)[0]
pm = np.concatenate([P[i] for i in tr]).mean(0); ps = np.concatenate([P[i] for i in tr]).std(0) + 1e-6
class Net(nn.Module):
    def __init__(s, d=256, L=4):
        super().__init__(); s.inp = nn.Linear(782, d); s.pos = nn.Embedding(W, d)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 4 * d, dropout=0.1, batch_first=True, norm_first=True), L); s.stage = nn.Linear(d, NS); s.prog = nn.Linear(d, 1)
    def forward(s, x):
        T = x.shape[1]; mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        h = s.enc(s.inp(x) + s.pos(torch.arange(T, device=x.device))[None], mask=mask); return s.stage(h), s.prog(h)[..., 0]
def feats(i): return np.c_[V[i], (P[i] - pm) / ps]
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); m = Net().to(dev); opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.05)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=a.steps, pct_start=0.05); tasks = sorted(set(TK)); by = {t: tr[TK[tr] == t] for t in tasks}
for step in range(a.steps):
    ids = np.concatenate([rng.choice(by[t], 64 // len(tasks) + 1) for t in tasks]); xs, ys, gs = [], [], []   # task-balanced batches
    for i in ids:
        e = rng.integers(1, len(V[i]) + 1); st = max(0, e - W); xs.append(feats(i)[st:e]); ys.append(S[i][st:e]); gs.append(G[i][st:e])
    T = max(len(x) for x in xs); X = np.zeros((len(xs), T, 782), np.float32); Y = np.full((len(xs), T), -1); Gp = np.full((len(xs), T), np.nan, np.float32)
    for b, (x, y, g) in enumerate(zip(xs, ys, gs)): X[b, T - len(x):] = x; Y[b, T - len(x):] = y; Gp[b, T - len(x):] = g
    X, Y, Gp = torch.tensor(X, device=dev), torch.tensor(Y, device=dev), torch.tensor(Gp, device=dev)
    sg, pr = m(X); k = Y >= 0; loss = F.cross_entropy(sg[k], Y[k]); k = ~torch.isnan(Gp); loss = loss + F.l1_loss(torch.sigmoid(pr[k]), Gp[k])
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
    if step % 2000 == 0: print(step, round(float(loss), 4), flush=True)
m.eval()
@torch.no_grad()
def predict(i):
    x = feats(i); xs = [x[max(0, e - W):e] for e in range(1, len(x) + 1)]; out_s, out_p = [], []
    for c0 in range(0, len(xs), 256):
        xb = xs[c0:c0 + 256]; T = max(len(v) for v in xb); X = np.zeros((len(xb), T, 782), np.float32)
        for b, v in enumerate(xb): X[b, T - len(v):] = v
        sg, pr = m(torch.tensor(X, device=dev)); out_s.append(sg[:, -1].argmax(1).cpu().numpy()); out_p.append(torch.sigmoid(pr[:, -1]).cpu().numpy())
    return np.concatenate(out_s), np.concatenate(out_p)
def summ(ids, Pd):
    y = np.concatenate([S[i] for i in ids]); p = np.concatenate([Pd[i][0] for i in ids]); e = np.concatenate([np.abs(Pd[i][1] - G[i]) for i in ids])
    return dict(episodes=len(ids), frames=len(y), acc=round(float((y == p).mean()), 4), progress_mae=round(float(e.mean()), 4),
                f1={NAMES[c]: round(float(2 * ((y == c) & (p == c)).sum() / max((y == c).sum() + (p == c).sum(), 1)), 3) for c in range(NS) if (y == c).any()},
                label_share={NAMES[c]: round(float((y == c).mean()), 3) for c in range(NS)})
R = dict(seed=a.seed, stages=NAMES, categories=CAT, n={s: int((SP == v).sum()) for s, v in (('train', 0), ('val', 1), ('test', 2))})
for sn, sv in (('val', 1), ('test', 2)):
    ids = np.where(SP == sv)[0]; Pd = {i: predict(i) for i in ids}
    R[sn] = dict(all=summ(ids, Pd), per_task={t: summ(ids[TK[ids] == t], Pd) for t in tasks}, per_category={c: summ(ids[np.isin(TK[ids], [t for t in tasks if CAT[t] == c])], Pd) for c in sorted(set(CAT.values()))})
    print(sn, json.dumps({c: (v['acc'], v['progress_mae']) for c, v in R[sn]['per_category'].items()}), flush=True)
json.dump(R, open(H / ('RESULT-shared-stage%s-s%d.json' % (a.tag, a.seed)), 'w'), indent=1)
