"""Dedup refit and margin check (bottleneck line, 16:4xZ; dispatcher). Zero quota, CPU only, nothing armed.
(1) gate v2: deploy-kind rows of gate/DATASET-v2.json deduplicated to one row per (kind, context, takeover step) (first by episode name, as
    ROUTER-v0 Amendment B), the same grouped 5-fold CV / features / model / 0.65 threshold as gate/train_eval_v2.py -> margins; refit on the
    dedup rows -> rsi_v2/policies/gate_v2.json (dedup_refit_required false) ONLY if routed > always-Astra and > always-intervene and harmful <= 10 %.
(2) ROUTER-v0: re-run its grouped 5-fold CV exactly as gate/train_router_v0.py (rows already deduplicated by Amendment B; CPU here) and report
    margins overall and on the plug states that are in macro-tools-20261027/c1/unique-starts.json (the 60-state rg3f pool after dedup), plus a
    threshold sweep (0.5 / 0.65 / 0.8) as information only (the frozen threshold stays 0.65).  -> out/DEDUP-REFIT-1029.json"""
import json, random, sys
import numpy as np
G = '/lizekai/RSIws/experiments/macro-tools-20261027/gate'; H = '/lizekai/RSIws/experiments/harness-v2-20261028'; sys.path.insert(0, G)
U = set(json.load(open('/lizekai/RSIws/experiments/macro-tools-20261027/c1/unique-starts.json'))['unique']); OUT = {}
# ---------------- (1) gate v2
import train_eval_v2 as TE
from sklearn.linear_model import LogisticRegression
seen = set(); rows = []
for r in sorted([r for r in TE.D if r['kind'] in TE.DEPLOY], key=lambda r: (r['kind'], r['episode'])):
    k = (r['kind'], r['ctx'], r['t'])
    if k in seen: continue
    seen.add(k); rows.append(r)
ev = TE.evaluate(rows, 'deploy kinds, deduplicated'); s = ev['success']
ok = s['gate'] > s['always_astra'] and s['gate'] > s['always_intervene'] and ev['harmful'] <= 0.10 * max(ev['interventions'], 1)
OUT['gate_v2_dedup'] = dict(n=ev['n'], positives=ev['positives'], success=s, interventions=ev['interventions'], harmful=ev['harmful'], per_kind=ev['per_kind'], margins_hold=bool(ok))
if ok:
    X = np.array([TE.x_of(r) for r in rows]); y = np.array([int(r['int_success'] > r['astra_success']) for r in rows]); m = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000).fit(X, y)
    A = json.load(open(H + '/rsi_v2/policies/gate_v2.json')); old = dict(coef=A['coef'], intercept=A['intercept'], n_train=A['n_train'])
    A.update(coef=[round(float(c), 8) for c in m.coef_[0]], intercept=round(float(m.intercept_[0]), 8), n_train=len(rows), positives=int(y.sum()), dedup_refit_required=False,
             cv_primary_dedup=OUT['gate_v2_dedup'], previous_fit_with_duplicates=old, dedup_note='16:4xZ refit on deduplicated rows (one per kind, context, step); CV margins hold.')
    json.dump(A, open(H + '/rsi_v2/policies/gate_v2.json', 'w'), indent=1); OUT['gate_v2_refit'] = 'written (dedup_refit_required false)'
else:
    OUT['gate_v2_refit'] = 'NOT written: margins do not hold after dedup; gate v2 stays log-only'
# ---------------- (2) ROUTER-v0 CV re-run on CPU
sys.argv = [sys.argv[0], '--gpu', '-1', '--tasks', 'plug_in_charger']
src = open(G + '/train_router_v0.py').read().split("if __name__ == '__main__':")[0]; ns = {}; exec(compile(src, G + '/train_router_v0.py', 'exec'), ns)
import torch
D = ns['D']; dev = torch.device('cpu')
for r in D: r['y'] = int(bool(r['int_success']) and not r['astra_success'] and not r['abstained'])
seen = set(); idx = []
for i in sorted([i for i, r in enumerate(D) if r['kind'] in {'rg3f'}], key=lambda i: (D[i]['kind'], D[i]['episode'])):
    k = (D[i]['kind'], D[i]['ctx'], D[i]['t'], None)
    if k in seen: continue
    seen.add(k); idx.append(i)
ctx = sorted({D[i]['ctx'] for i in idx}); random.Random(20261029).shuffle(ctx); fold = {c: k % 5 for k, c in enumerate(ctx)}
p_r = np.zeros(len(idx)); pos = {i: j for j, i in enumerate(idx)}
for k in range(5):
    tr = [i for i in idx if fold[D[i]['ctx']] != k]; te = [i for i in idx if fold[D[i]['ctx']] == k]
    if not te: continue
    for i, a_ in zip(te, ns['fit_predict'](tr, te, D, dev)): p_r[pos[i]] = a_
def margins(sel, thr):
    R = [D[i] for i in sel]; p = np.array([p_r[pos[i]] for i in sel]); ab = np.array([r['abstained'] for r in R]); go = (p > thr) & ~ab
    sa = np.array([r['astra_success'] for r in R], float); si = np.where(ab, sa, np.array([r['int_success'] for r in R], float)); sg = np.where(go, si, sa)
    return dict(n=len(R), positives=int(sum(r['y'] for r in R)), routed=round(float(sg.mean()), 3), always_astra=round(float(sa.mean()), 3), always_executor=round(float(si.mean()), 3),
                takeovers=int(go.sum()), harmful=int(sum(g and r['astra_success'] and not r['int_success'] for g, r in zip(go, R))))
sub = [i for i in idx if D[i]['episode'] in U]
OUT['router_v0_cv_cpu'] = dict(all={str(t): margins(idx, t) for t in (0.5, 0.65, 0.8)}, unique_60pool={str(t): margins(sub, t) for t in (0.5, 0.65, 0.8)},
                               judged_gpu=json.load(open(G + '/RESULT-router-v0.json'))['router']['success'])
m = OUT['router_v0_cv_cpu']
OUT['router_v0_margins_hold'] = {k: bool(v['0.65']['routed'] > v['0.65']['always_astra'] and v['0.65']['routed'] > v['0.65']['always_executor']) for k, v in (('all', m['all']), ('unique_60pool', m['unique_60pool']))}
json.dump(OUT, open(H + '/out/DEDUP-REFIT-1029.json', 'w'), indent=1); print(json.dumps(OUT, indent=1)[:3500])
