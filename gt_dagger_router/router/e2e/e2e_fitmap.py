"""M-R2 end-to-end offline check on the plug fitmap (v1: C1 o7 truth-ceiling VS outcomes, rg3f comp, Astra archived; 84 unique states).
Policies: always-Astra, always-rg3f, always-VS, oracle, RULE (tool table v1: lift-time v4 abstain -> VS; early hold t <= 95 and v4 ok ->
VS if the VS head... -> here: v4 abstain -> VS, else VS), and LEARNED (3-way tool head: Astra / rg3f / VS, MLP on [PCA-32 of the DINOv3
handover features, t/400, v4-abstain flag], grouped 5-fold CV by context; label = best method in the order VS, rg3f, Astra, none -> Astra).
VS outcomes are a truth-keypoint ceiling (the deploy KPNet version is pending), so the routed numbers are an upper bound of the tool choice.
 -> e2e/RESULT-e2e-fitmap.json"""
import json, random
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
H = Path(__file__).resolve().parents[1]; F = json.load(open(H / 'fitmap/fitmap.json'))
D = json.load(open(H.parent / 'gate/ROUTER-DATA-v0.json'))['rows']; X = np.load(H.parent / 'gate/ROUTER-DATA-v0.npz')['dino'].reshape(len(D), -1)
feat = {}
for r, x in zip(D, X):
    if r['task'] == 'plug_in_charger': feat.setdefault((r['ctx'], r['t']), x)
rows = [e for e in F if e.get('vs') is not None and e.get('astra') is not None]
def outc(e, m): return bool(e.get({'Astra': 'astra', 'rg3f': 'rg3f', 'VS': 'vs'}[m]))
pol = {m: [outc(e, m) for e in rows] for m in ('Astra', 'rg3f', 'VS')}
res = dict(n=len(rows), always={m: round(float(np.mean(v)), 3) for m, v in pol.items()}, oracle=round(float(np.mean([any(outc(e, m) for m in ('Astra', 'rg3f', 'VS')) for e in rows])), 3))
# RULE: tool table v1 (VS dominates on this map: VS for every held state; harm = states where Astra alone succeeded and VS failed)
rule = ['VS' for e in rows]; res['rule_VS_all'] = dict(success=round(float(np.mean([outc(e, m) for e, m in zip(rows, rule)])), 3), harmful=int(sum(e['astra'] and not e['vs'] for e in rows)))
# LEARNED: 3-way tool head, grouped CV
lab = np.array([0 if not (e['vs'] or e.get('rg3f') or e['astra']) else (2 if e['vs'] else 1 if e.get('rg3f') else 0) for e in rows])   # 0 Astra, 1 rg3f, 2 VS
have = np.array([(e['ctx'], e['t']) in feat for e in rows]); Xs = np.array([feat.get((e['ctx'], e['t']), np.zeros(X.shape[1])) for e in rows], np.float32)
extra = np.array([[e['t'] / 400.0, float(bool(e.get('lift_abstain')))] for e in rows])
ctx = [e['ctx'] for e in rows]; cs = sorted(set(ctx)); random.Random(20261031).shuffle(cs); fold = {c: i % 5 for i, c in enumerate(cs)}; fo = np.array([fold[c] for c in ctx])
pred = np.zeros(len(rows), int)
for k in range(5):
    tr, te = np.where(fo != k)[0], np.where(fo == k)[0]
    p = PCA(32, random_state=0).fit(Xs[tr][have[tr]]); Z = np.c_[p.transform(Xs) * have[:, None], extra]
    if len(set(lab[tr])) < 2: pred[te] = lab[tr][0]; continue
    m = LogisticRegression(C=0.3, max_iter=4000, class_weight='balanced').fit(Z[tr], lab[tr]); pred[te] = m.predict(Z[te])
name = ['Astra', 'rg3f', 'VS']; ch = [name[i] for i in pred]
res['learned_tool_head'] = dict(success=round(float(np.mean([outc(e, m) for e, m in zip(rows, ch)])), 3), choices={m: ch.count(m) for m in name},
                                harmful=int(sum(e['astra'] and not outc(e, m) for e, m in zip(rows, ch))), features_missing=int((~have).sum()))
# fit by subset
def sub(sel): return dict(n=int(sum(sel)), Astra=round(float(np.mean([e['astra'] for e, s in zip(rows, sel) if s])), 3) if any(sel) else None,
                          rg3f=round(float(np.mean([bool(e.get('rg3f')) for e, s in zip(rows, sel) if s])), 3) if any(sel) else None,
                          VS=round(float(np.mean([e['vs'] for e, s in zip(rows, sel) if s])), 3) if any(sel) else None)
res['by_subset'] = {'t<=95 v4ok': sub([e['t'] <= 95 and not e.get('lift_abstain') for e in rows]), 't<=95 v4abstain': sub([e['t'] <= 95 and bool(e.get('lift_abstain')) for e in rows]),
                    't>95': sub([e['t'] > 95 for e in rows])}
json.dump(res, open(H / 'e2e/RESULT-e2e-fitmap.json', 'w'), indent=1); print(json.dumps(res, indent=1))
