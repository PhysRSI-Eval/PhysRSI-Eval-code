"""M-R2 offline check of the wired router v4 (router_v4.RouterV4): (1) full RGB path: task ID from raw first frames of archived episodes
(one per task, 42; in-sample for the task-ID head = wiring check), (2) BIG stage head through step_feat on the held-out K1 episodes
(must reproduce the trainer's dev stage number), (3) routed success on the plug fitmap states (dev groups = held-out; train groups
reported as in-sample) vs always-Astra / always-rg3f / always-VS, with the v4-abstain -> VS rule.  -> e2e/RESULT-e2e-v4-<model>.json  (argv[1] = model file in big/)"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
H = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(H)); import router_v4 as R4   # noqa: E402
MOD = sys.argv[1] if len(sys.argv) > 1 else 'model-h200-v2-dense-d512L8.pt'; R = R4.RouterV4('cuda:0', big=H / 'big' / MOD); res = dict(model=MOD)
# (1) task ID through the full RGB path
E = json.load(open(H / 'taskid/episode_index.json')); NAT = Path('/lizekai/RSIws/artifacts/native'); first = {}
for e in E: first.setdefault(e['task'], e)
ok, okt = [], []
for t, e in sorted(first.items()):
    o = NAT / e['episode'] / e['inner'] / 'observations'
    ims = {c: np.asarray(Image.open(o / ('000000-%s.png' % c)).convert('RGB')) for c in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')}
    R.reset(); R.step(ims, dict(g=(1.0, 1.0), z=(0.0, 0.0), xy=((0, 0), (0, 0))), 0); ok.append(R.task == t)
    if R.ttx is not None:
        ins = json.load(open(NAT / e['episode'] / 'reset.json')).get('instruction', ''); R.reset(ins); R.step(ims, dict(g=(1.0, 1.0), z=(0.0, 0.0), xy=((0, 0), (0, 0))), 0); okt.append(R.task == t)
res['taskid_rgb_path'] = dict(n=len(ok), acc=round(float(np.mean(ok)), 3))
if okt: res['taskid_text_rgb_path'] = dict(n=len(okt), acc=round(float(np.mean(okt)), 3))
# (2) stage through step_feat on held-out K1 episodes
Z = np.load(H / 'big/big_v2.npz', allow_pickle=True); Z = {k: Z[k] for k in ('V', 'P', 'stage', 'src', 'dev')}; src, dev = Z['src'], Z['dev'].astype(bool); y, p = [], []   # read each array once
for i in np.where((src == 'astra_k1') & dev)[0]:
    V, P, S = Z['V'][i].astype(np.float32), Z['P'][i], Z['stage'][i]
    for f in range(len(V)):
        if S[f] < 0: continue
        y.append(int(S[f])); p.append(R4.STAGES.index(R.step_feat(V[:f + 1], P[:f + 1])['stage']))
y, p = np.array(y), np.array(p); res['k1_stage_dev'] = dict(frames=len(y), acc=round(float((y == p).mean()), 3), f1_lift=round(float(2 * ((y == 2) & (p == 2)).sum() / max((y == 2).sum() + (p == 2).sum(), 1)), 3))
# (3) plug fitmap states, same row order as big/build_big_v2.py
F = {(e['ctx'], e['t']): e for e in json.load(open(H / 'fitmap/fitmap.json')) if e.get('vs') is not None}
D = json.load(open(H.parent / 'gate/ROUTER-DATA-v0.json'))['rows']; seen, rows = set(), []
for r in D:
    if r['task'] != 'plug_in_charger' or (r['ctx'], r['t']) in seen or (r['ctx'], r['t']) not in F: continue
    seen.add((r['ctx'], r['t'])); rows.append(F[(r['ctx'], r['t'])])
idx = np.where(src == 'fitmap_plug_tool')[0]; assert len(idx) == len(rows)
R.task = 'plug_in_charger'; key = {'Astra': 'astra', 'rg3f': 'rg3f', 'VS': 'vs'}; out = {}
for name, sel in (('dev_heldout', True), ('train_insample', False)):
    ch, rs = [], []
    for i, e in zip(idx, rows):
        if bool(dev[i]) != sel: continue
        o = R.step_feat(Z['V'][i].astype(np.float32), Z['P'][i], v4_abstain=bool(e.get('lift_abstain')))
        m = o['method'] if o['stage'] in ('lift', 'transport', 'align') else R4.TOOLS[int(np.argmax(o['tool_p']))]   # these are held states by construction
        ch.append(m); rs.append(e)
    out[name] = dict(n=len(rs), routed=round(float(np.mean([bool(e.get(key[m])) for e, m in zip(rs, ch)])), 3) if rs else None,
                     choices={t: ch.count(t) for t in R4.TOOLS}, harmful=int(sum(bool(e.get('astra')) and not e.get(key[m]) for e, m in zip(rs, ch))),
                     **{'always_' + t: round(float(np.mean([bool(e.get(key[t])) for e in rs])), 3) if rs else None for t in R4.TOOLS},
                     oracle=round(float(np.mean([any(e.get(key[t]) for t in R4.TOOLS) for e in rs])), 3) if rs else None)
res['plug_fitmap'] = out; res['note'] = 'VS outcomes = truth-keypoint ceiling; fitmap rows were trained with P = 0 (single frame), deploy P is real (shift)'
json.dump(res, open(H / ('e2e/RESULT-e2e-v4-%s.json' % MOD[:-3]), 'w'), indent=1); print(json.dumps(res))
