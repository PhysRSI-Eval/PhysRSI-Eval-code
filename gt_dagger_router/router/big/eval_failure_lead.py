"""Does the failure-per-stage head flag the BC-rollout drop states BEFORE the drop?  python eval_failure_lead.py MODEL.pt DAGGER.npz BASE.npz TAG
 * p_fail(frame) = the fail LOGIT of the frame's predicted stage (v1 of this script used the sigmoid: saturated at 1.0, threshold 1.0 -> invalid) of a train_big_v3 model (causal window, deploy inputs only).
 * threshold: 90th percentile of p_fail over the frames of SUCCESSFUL Astra K1 val episodes (per-frame false-alarm rate 10 % there).
 * drop rollouts (charger_not_held appears): first flagged step vs the first charger_not_held step -> lead time in env steps;
   budget rollouts (no drop; also failures) report the flag rate.  Rollouts of every split are reported, held-out (split 1 / 2) separately:
   a model trained WITH the dagger rows has seen the train-split rollouts (and their all-failure label), so only held-out rows are evidence;
   a model trained WITHOUT them (v3m) is a transfer test.  Also the drop-vs-budget AUROC of max p_fail before the drop / over the rollout.
 -> RESULT-failure-lead-TAG.json"""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H.parent)); import router_v4 as R4   # noqa: E402
mp, dp, bp, tag = sys.argv[1:5]; dev = torch.device('cuda:0')
class NetV3(R4.Net):
    def __init__(s, d, L, e, W):
        super().__init__(d, L, e, W); s.take = nn.Linear(d, 4); s.fail = nn.Linear(d, 6)
    def forward(s, x):
        T = x.shape[1]; mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1); h = s.inp(x) + s.pos(torch.arange(T, device=x.device))[None]
        for b in s.blocks: h = b(h, mask)
        h = s.n(h); return s.stage(h), s.fail(h)
ck = R4.load_ck(mp); a = ck['args']; m = NetV3(a['d'], a['layers'], a['experts'], a['W']).to(dev); m.load_state_dict(ck['state_dict']); m.eval()
pm, ps = np.asarray(ck['pm'], np.float32), np.asarray(ck['ps'], np.float32); Wn = a['W']
@torch.no_grad()
def pfail(V, P):
    x = np.c_[V.astype(np.float32), (P - pm) / ps]; xs = [x[max(0, e - Wn):e] for e in range(1, len(x) + 1)]; out = []
    for c0 in range(0, len(xs), 256):
        xb = xs[c0:c0 + 256]; T = max(len(v) for v in xb); X = np.zeros((len(xb), T, x.shape[1]), np.float32)
        for b, v in enumerate(xb): X[b, T - len(v):] = v
        sg, fl = m(torch.tensor(X, device=dev)); sg, fl = sg[:, -1], fl[:, -1]; out.append(fl.gather(1, sg.argmax(1, keepdim=True))[:, 0].float().cpu().numpy())
    return np.concatenate(out)
B = np.load(bp, allow_pickle=True); src, spl, succ = B['src'], B['split'].astype(int), B['succ'].astype(float); BV, BP = B['V'], B['P']
ref = np.concatenate([pfail(BV[i], BP[i]) for i in np.where((src == 'astra_k1') & (spl == 1) & (succ == 1))[0]]); thr = float(np.quantile(ref, 0.9))
D = np.load(dp, allow_pickle=True); D = {k: D[k] for k in D.files}; rows = []
for i in range(len(D['V'])):
    p = pfail(D['V'][i], D['P'][i]); t = D['tstep'][i].astype(int); ds = int(D['drop_step'][i]); pre = t < ds if ds >= 0 else np.ones(len(t), bool)
    fl = np.where((p > thr) & pre)[0]
    rows.append(dict(job=str(D['job'][i]), split=int(D['split'][i]), outcome=str(D['outcome'][i]), drop_step=ds, start=int(t[0]), first_flag=int(t[fl[0]]) if len(fl) else None,
                     lead=(ds - int(t[fl[0]])) if (ds >= 0 and len(fl)) else None, max_p_pre=float(p[pre].max()) if pre.any() else None,
                     max_p_first16=float(p[t < t[0] + 16].max()), drop_after16=bool(ds < 0 or ds >= t[0] + 16)))
def summ(rs):
    dr = [r for r in rs if r['drop_step'] >= 0]; bu = [r for r in rs if r['outcome'] == 'budget']; leads = [r['lead'] for r in dr if r['lead'] is not None]
    dr16 = [r for r in dr if r['drop_after16']]   # length-matched: max over the first 16 steps after the start, drops after step 16 only
    y = [1] * len(dr16) + [0] * len(bu); s_ = [r['max_p_first16'] for r in dr16 + bu]
    au = None
    if dr16 and bu:
        o = np.argsort(np.argsort(s_)) + 1; au = round(float((o[:len(dr16)].sum() - len(dr16) * (len(dr16) + 1) / 2) / (len(dr16) * len(bu))), 3)
    return dict(drop_rollouts=len(dr), flagged_before_drop=sum(r['lead'] is not None for r in dr), lead_steps=dict(median=float(np.median(leads)) if leads else None,
                p25=float(np.percentile(leads, 25)) if leads else None, p75=float(np.percentile(leads, 75)) if leads else None),
                drop_steps_after_start=float(np.median([r['drop_step'] - r['start'] for r in dr])) if dr else None,
                budget_rollouts=len(bu), budget_flagged=sum(r['first_flag'] is not None for r in bu), auroc_drop_vs_budget_first16=au, n_drop_after16=len(dr16))
res = dict(model=mp, threshold=round(thr, 4), threshold_def='90th pct of p_fail on successful Astra K1 val frames', all=summ(rows),
           heldout=summ([r for r in rows if r['split'] != 0]), train_split=summ([r for r in rows if r['split'] == 0]), rollouts=rows)
json.dump(res, open(H / ('RESULT-failure-lead-%s.json' % tag), 'w'), indent=1); print(json.dumps({k: res[k] for k in ('threshold', 'all', 'heldout', 'train_split')}))
