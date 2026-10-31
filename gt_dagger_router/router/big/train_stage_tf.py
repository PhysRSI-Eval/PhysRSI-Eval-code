"""Teacher-forced stage / progress heads on the official demos (--src astra_k1: Astra K1 episodes, sim-truth stage labels, v3 group split; progress unlabelled there) (user 10-31 02:2xZ; GT.md split 80 / 10 / 10 via big_v3.npz 'split').
Input per frame = PCA-768 DINOv3 + 14 proprio (state history; normalised on train) + one-hot of the PREVIOUS stage (6 stages + 'unknown').
  * tf      : trained and evaluated with the GT previous stage (teacher forcing);
  * free    : the same model evaluated free-running (previous stage = its own argmax at t-1) -> exposure-bias gap that DAgger closes;
  * nocond  : same network without the previous-stage input (reference).
  --merge_gl 1: lift merged into grasp ('grasp-lift', class 1);  --tol_frames k: tolerant F1 (+-k cached frames; K1 cache = every 4th step, so 1 ~ +-5 steps).
  --tstep 1 : the env step index / 400 as an input (placed vs approach look alike; the step counter separates early from late).
  --placed_succ 1: placed label only in successful episodes (label-definition fix; 70 % of K1 placed frames are in failed episodes).
  --ss p    : scheduled sampling for the conditioned model (own previous prediction with prob p in training; cheap offline DAgger proxy).
Caveat: the demo 'GT' stage is the proprio-derived label of build_big (approach / grasp / lift / placed; carry frames unknown), not a sim
truth stage (the demos carry no stage labels, GT.md 8).  Causal transformer d256 L4, window 32.  -> RESULT-stage-tf.json"""
import argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap = argparse.ArgumentParser(); ap.add_argument('--data', default='big_v3.npz'); ap.add_argument('--steps', type=int, default=8000); ap.add_argument('--device', default='cuda:0'); ap.add_argument('--src', default='demo'); ap.add_argument('--ss', type=float, default=0.0); ap.add_argument('--merge_gl', type=int, default=0); ap.add_argument('--tol_frames', type=int, default=1); ap.add_argument('--tstep', type=int, default=0); ap.add_argument('--placed_succ', type=int, default=0); ap.add_argument('--out', default='RESULT-stage-tf.json')
a = ap.parse_args(); H_ = Path(__file__).resolve().parent; dev = torch.device(a.device); W, NS = 32, 6
Z = np.load(a.data, allow_pickle=True); sel = np.where(Z['src'] == a.src)[0]
ZV, ZP, ZS, ZG = Z['V'], Z['P'], Z['stage'], Z['prog']   # each Z[key] access decompresses the whole array: read once
V = [ZV[i].astype(np.float32) for i in sel]; P = [ZP[i] for i in sel]; S = [np.where(ZS[i].astype(int) == 2, 1, ZS[i].astype(int)) if a.merge_gl else ZS[i].astype(int) for i in sel]   # --merge_gl: lift (1-3 cm) -> grasp ('grasp-lift')
G = [ZG[i].astype(np.float32) for i in sel]; SP = Z['split'][sel].astype(int)
TS = [np.zeros(len(v), np.float32) for v in V]
if a.placed_succ:   # --placed_succ 1: 'placed' (5) kept only in natively successful episodes; elsewhere (object left near the target, 70 % of placed frames) -> unknown
    SU = Z['succ'][sel].astype(float); S = [np.where((s_ == 5) & (su != 1), -1, s_) for s_, su in zip(S, SU)]
if a.tstep:   # --tstep 1: env step index / 400 as an extra input (deployable: the step counter is known at deploy); K1 steps from stage_k1_v2 meta
    assert a.src == 'astra_k1'; K = np.load(H_ / '../stage/stage_k1_v2.npz', allow_pickle=True); KM = json.loads(str(K['meta'])); KX = K['X']
    TS = [np.asarray(m_['steps'], np.float32) / 400.0 for x_, m_ in zip(KX, KM) if len(x_)]; assert [len(t) for t in TS] == [len(v) for v in V]
tr = np.where(SP == 0)[0]; pm = np.concatenate([P[i] for i in tr]).mean(0); ps = np.concatenate([P[i] for i in tr]).std(0) + 1e-6
def prev_onehot(prev):
    o = np.zeros((len(prev), NS + 1), np.float32); o[np.arange(len(prev)), np.where(np.asarray(prev) >= 0, prev, NS)] = 1; return o
def gt_prev(s): return np.r_[-1, s[:-1]]
class Net(nn.Module):
    def __init__(s, cond, d=256, L=4):
        super().__init__(); s.cond = cond; s.inp = nn.Linear(768 + 14 + a.tstep + (NS + 1) * cond, d); s.pos = nn.Embedding(W, d)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 4 * d, dropout=0.1, batch_first=True, norm_first=True), L); s.stage = nn.Linear(d, NS); s.prog = nn.Linear(d, 1)
    def forward(s, x):
        T = x.shape[1]; mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        h = s.enc(s.inp(x) + s.pos(torch.arange(T, device=x.device))[None], mask=mask); return s.stage(h), s.prog(h)[..., 0]
def feats(i, prev, cond):
    x = np.c_[V[i], (P[i] - pm) / ps] if not a.tstep else np.c_[V[i], (P[i] - pm) / ps, TS[i]]; return np.c_[x, prev_onehot(prev)] if cond else x
def train(cond):
    torch.manual_seed(0); rng = np.random.default_rng(0); m = Net(cond).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=a.steps, pct_start=0.05)
    for step in range(a.steps):
        ids = rng.choice(tr, 64); xs, ys, gs = [], [], []
        for i in ids:
            e = rng.integers(1, len(V[i]) + 1); st = max(0, e - W); x = feats(i, gt_prev(S[i]), cond)[st:e]; xs.append(x); ys.append(S[i][st:e]); gs.append(G[i][st:e])
        T = max(len(x) for x in xs); X = np.zeros((64, T, xs[0].shape[1]), np.float32); Y = np.full((64, T), -1); Gp = np.full((64, T), np.nan, np.float32)
        for b, (x, y, g) in enumerate(zip(xs, ys, gs)): X[b, T - len(x):] = x; Y[b, T - len(x):] = y; Gp[b, T - len(x):] = g
        X, Y, Gp = torch.tensor(X, device=dev), torch.tensor(Y, device=dev), torch.tensor(Gp, device=dev)
        if cond and a.ss > 0:   # scheduled sampling: with prob ss per frame the previous-stage input = the model's own prediction at t-1 (one no-grad pass)
            with torch.no_grad(): own = m(X)[0].argmax(-1)
            own = torch.cat([torch.full_like(own[:, :1], NS), own[:, :-1]], 1); use = torch.rand(own.shape, device=dev) < a.ss
            X = X.clone(); oh = F.one_hot(own, NS + 1).float(); X[..., -(NS + 1):] = torch.where(use[..., None], oh, X[..., -(NS + 1):])
        sg, pr = m(X); k = Y >= 0; loss = F.cross_entropy(sg[k], Y[k]); k = ~torch.isnan(Gp); loss = loss + (F.l1_loss(torch.sigmoid(pr[k]), Gp[k]) if k.any() else 0)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        if step % 2000 == 0: print(cond, step, round(float(loss), 4), flush=True)
    return m.eval()
@torch.no_grad()
def evaluate(m, ids, mode):
    ys, ps_, pe, yt = [], [], [], []
    for i in ids:
        L = len(V[i]); prev = gt_prev(S[i]) if mode == 'tf' else np.full(L, -1); own = []
        for e in range(1, L + 1):
            if mode == 'free': prev[e - 1] = own[-1] if own else -1
            x = feats(i, prev, m.cond)[max(0, e - W):e]; sg, pr = m(torch.tensor(x[None], device=dev)); own.append(int(sg[0, -1].argmax()))
            if S[i][e - 1] >= 0:
                ys.append(S[i][e - 1]); ps_.append(own[-1]); w_ = S[i][max(0, e - 1 - a.tol_frames):e + a.tol_frames]; yt.append(own[-1] if own[-1] in w_ else S[i][e - 1])   # tolerant: a label within +-tol frames
            if not np.isnan(G[i][e - 1]): pe.append(abs(float(torch.sigmoid(pr[0, -1])) - G[i][e - 1]))
    y, p, ytl = np.array(ys), np.array(ps_), np.array(yt)
    f1 = lambda yy: {c: round(float(2 * ((yy == c) & (p == c)).sum() / max((yy == c).sum() + (p == c).sum(), 1)), 3) for c in range(NS) if (yy == c).any()}
    return dict(frames=len(y), acc=round(float((y == p).mean()), 4), f1={c: round(float(2 * ((y == c) & (p == c)).sum() / max((y == c).sum() + (p == c).sum(), 1)), 3) for c in range(NS) if (y == c).any()},
                acc_tol=round(float((ytl == p).mean()), 4), f1_tol=f1(ytl), progress_mae=round(float(np.mean(pe)), 4) if pe else None)
R = dict(n_demos={s: int((SP == v).sum()) for s, v in (('train', 0), ('val', 1), ('test', 2))}, stage_names=['approach', 'grasp', 'lift', 'transport', 'align', 'placed'])
m0, m1 = train(0), train(1)
for sn, sv in (('val', 1), ('test', 2)):
    ids = np.where(SP == sv)[0]; R[sn] = dict(nocond=evaluate(m0, ids, 'nocond'), tf=evaluate(m1, ids, 'tf'), free=evaluate(m1, ids, 'free'))
    R[sn]['exposure_gap_acc'] = round(R[sn]['tf']['acc'] - R[sn]['free']['acc'], 4); print(sn, json.dumps(R[sn]), flush=True)
json.dump(R, open(a.out, 'w'), indent=1)
