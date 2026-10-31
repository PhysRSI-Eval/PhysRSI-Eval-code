"""Drop-hazard head (dispatcher 10-31): label = the charger is dropped within the next K env steps (K = 8 / 16) vs not, on BC-rollout rows
(big_dagger_*.npz from ingest_dagger.py: V PCA-768, P 14 proprio, tstep, drop_step = first charger_not_held step or -1). Frames at / after
the drop are excluded. Budget rollouts (no drop) are all negatives. Causal transformer (d128 L2, window 16 rows; rows are ~4 steps apart),
trained on split-0 rollouts, evaluated on held-out rollouts (split 1 + 2; split = ctx group of the rollout's start episode, so the
held-out rollouts are new rollout IDs AND new start contexts). Deploy inputs only.
Reports: frame AUROC; AUROC per 8-step bin of steps-since-start; lead time at a fixed false-alarm rate on held-out BUDGET rollouts
(threshold = the q-quantile of hazard over budget frames, q = 0.95 and 0.90; rollout-level: fraction of budget rollouts that ever flag);
detection = flagged at least once before the drop; lead = drop step - first flag step.
--no_vision 1: visual input zeroed (proprio + steps only; attribution of the hazard signal).
python train_drop_hazard.py --K 8 --data big_dagger_dag12f.npz [...] -> RESULT-drop-hazard-K8.json"""
import argparse, json, sys
for _i, _x in enumerate(sys.argv[:-1]):   # tags start with '-' ('-bal', '-d1'): argparse would read them as options
    if _x == '--tag': sys.argv[_i:_i + 2] = ['--tag=' + sys.argv[_i + 1]]; break
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap = argparse.ArgumentParser(); ap.add_argument('--K', type=int, default=8); ap.add_argument('--data', nargs='+', required=True); ap.add_argument('--steps', type=int, default=3000)
ap.add_argument('--device', default='cuda:0'); ap.add_argument('--tag', default=''); ap.add_argument('--budget_frac', type=float, default=0.0); ap.add_argument('--no_vision', type=int, default=0); ap.add_argument('--seed', type=int, default=0); a = ap.parse_args(); dev = torch.device(a.device); W = 16
V, P, T, DS, SP, JOB = [], [], [], [], [], []
for f in a.data:
    Z = np.load(f, allow_pickle=True); Z = {k: Z[k] for k in ('V', 'P', 'tstep', 'drop_step', 'split', 'job', 'outcome')}
    for i in range(len(Z['V'])):
        if Z['outcome'][i] not in ('charger_dropped', 'budget'): continue
        V.append(Z['V'][i].astype(np.float32)); P.append(Z['P'][i].astype(np.float32)); T.append(Z['tstep'][i].astype(int)); DS.append(int(Z['drop_step'][i])); SP.append(int(Z['split'][i])); JOB.append(str(Z['job'][i]))
SP = np.array(SP); tr = np.where(SP == 0)[0]; te = np.where(SP != 0)[0]; trb = np.array([i for i in tr if DS[i] < 0]); trd = np.array([i for i in tr if DS[i] >= 0])   # --budget_frac: share of budget rollouts per batch (19 / 104 train rollouts are budget)
pm = np.concatenate([P[i] for i in tr]).mean(0); ps = np.concatenate([P[i] for i in tr]).std(0) + 1e-6
def labels(i):
    t, ds = T[i], DS[i]
    if ds < 0: return np.zeros(len(t)), np.ones(len(t), bool)
    return ((t >= ds - a.K) & (t < ds)).astype(float), t < ds
class Net(nn.Module):
    def __init__(s, d=128, L=2):
        super().__init__(); s.inp = nn.Linear(768 + 14 + 1, d); s.pos = nn.Embedding(W, d)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.2, batch_first=True, norm_first=True), L); s.out = nn.Linear(d, 1)
    def forward(s, x):
        T_ = x.shape[1]; mask = torch.triu(torch.ones(T_, T_, dtype=torch.bool, device=x.device), 1); return s.out(s.enc(s.inp(x) + s.pos(torch.arange(T_, device=x.device))[None], mask=mask))[..., 0]
def feats(i): return np.c_[V[i] * (0 if a.no_vision else 1), (P[i] - pm) / ps, (T[i] - T[i][0])[:, None] / 100.0]   # + steps since the rollout start
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); m = Net().to(dev); opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.05)
pos_w = torch.tensor(float(sum((1 - labels(i)[0])[labels(i)[1]].sum() for i in tr) / max(sum(labels(i)[0].sum() for i in tr), 1)), device=dev)
for step in range(a.steps):
    ids = rng.choice(tr, 32) if not a.budget_frac else np.where(rng.random(32) < a.budget_frac, rng.choice(trb, 32), rng.choice(trd, 32)); xs, ys, ms = [], [], []
    for i in ids:
        y, ok = labels(i); e = rng.integers(1, len(y) + 1); st = max(0, e - W); xs.append(feats(i)[st:e]); ys.append(y[st:e]); ms.append(ok[st:e])
    Tm = max(len(x) for x in xs); X = np.zeros((32, Tm, 783), np.float32); Y = np.zeros((32, Tm), np.float32); M = np.zeros((32, Tm), bool)
    for b, (x, y, k) in enumerate(zip(xs, ys, ms)): X[b, Tm - len(x):] = x; Y[b, Tm - len(x):] = y; M[b, Tm - len(x):] = k
    lo = m(torch.tensor(X, device=dev)); k = torch.tensor(M, device=dev)
    loss = F.binary_cross_entropy_with_logits(lo[k], torch.tensor(Y, device=dev)[k], pos_weight=pos_w); opt.zero_grad(); loss.backward(); opt.step()
m.eval()
@torch.no_grad()
def score(i):
    x = feats(i); xs = [x[max(0, e - W):e] for e in range(1, len(x) + 1)]; Tm = max(len(v) for v in xs); X = np.zeros((len(xs), Tm, 783), np.float32)
    for b, v in enumerate(xs): X[b, Tm - len(v):] = v
    return m(torch.tensor(X, device=dev))[:, -1].cpu().numpy()
def auroc(y, p):
    y, p = np.asarray(y), np.asarray(p); n1, n0 = (y == 1).sum(), (y == 0).sum()
    if not n1 or not n0: return None
    r = np.argsort(np.argsort(p)) + 1; return round(float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), 3)
S = {i: score(i) for i in te}; yy, pp, bins = [], [], []
for i in te:
    y, ok = labels(i); yy += list(y[ok]); pp += list(S[i][ok]); bins += list(((T[i] - T[i][0]) // 8)[ok])
yy, pp, bins = np.array(yy), np.array(pp), np.array(bins)
res = dict(K=a.K, data=a.data, n_train_rollouts=len(tr), n_heldout=dict(drop=int(sum(DS[i] >= 0 for i in te)), budget=int(sum(DS[i] < 0 for i in te))),
           frame_auroc=auroc(yy, pp), auroc_by_steps_since_start={'%d-%d' % (8 * b, 8 * b + 7): dict(n_pos=int((yy[bins == b] == 1).sum()), n=int((bins == b).sum()), auroc=auroc(yy[bins == b], pp[bins == b])) for b in sorted(set(bins)) if b < 16})
bud = [i for i in te if DS[i] < 0]; drp = [i for i in te if DS[i] >= 0]; bf = np.concatenate([S[i] for i in bud]) if bud else np.zeros(1)
for q in (0.95, 0.90):
    thr = float(np.quantile(bf, q)); leads = []; det = 0
    for i in drp:
        f = np.where((S[i] > thr) & (T[i] < DS[i]))[0]
        if len(f): det += 1; leads.append(int(DS[i] - T[i][f[0]]))
    res['far_%.2f' % (1 - q)] = dict(threshold=round(thr, 3), budget_rollouts_flagged=int(sum((S[i] > thr).any() for i in bud)), budget_rollouts=len(bud), drops_detected=det, drops=len(drp),
                                     lead_steps=dict(median=float(np.median(leads)) if leads else None, p25=float(np.percentile(leads, 25)) if leads else None, p75=float(np.percentile(leads, 75)) if leads else None),
                                     drop_steps_after_start_median=float(np.median([DS[i] - T[i][0] for i in drp])) if drp else None)
json.dump(res, open('RESULT-drop-hazard-K%d%s.json' % (a.K, a.tag), 'w'), indent=1); print(json.dumps(res))
