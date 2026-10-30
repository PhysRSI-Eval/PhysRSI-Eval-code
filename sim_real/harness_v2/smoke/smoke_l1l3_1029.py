"""Smoke for the 10-29 harness additions (zero quota, no simulator, no model calls):
 1 L1 specs: insert_charger and segment_small listed with cards, predict_outcome registered but not listed; contract on every new reply.
 2 insert_charger without backend / without in-hand service -> abstain, 0 steps; L3 guard blocks an automatic call, allows explicit and policy token.
 3 segment_small on stored SSEG TEST views (coin wrist, coin head, mahjong head) vs the truth label masks; key -> not available.
 4 L3 gate v2: decide() from the archived per-step history of every rg3f_comp row reproduces the logistic on the dataset features.
 5 Harness glue: offer mode surfaces one availability fact (advice filter, rate limit), log mode surfaces nothing, auto without approval -> offer.
  /opt/PhysRSI/ENV/robodojo/bin/python smoke/smoke_l1l3_1029.py -> out/SMOKE-l1l3-1029.json"""
import json, sys, glob, random
from pathlib import Path
import numpy as np
H = Path('/lizekai/RSIws/experiments/harness-v2-20261028'); sys.path.insert(0, str(H))
from rsi_v2 import registry as L1, surfacing as L3, policy as P, harness as HN
from rsi_v2.contract import check
R = {}
names = [s['name'] for s in L1.specs()]
R['1_specs'] = dict(listed=names, insert_charger='insert_charger' in names, segment_small='segment_small' in names, predict_outcome_hidden='predict_outcome' not in names and 'predict_outcome' in L1.REGISTRY)
ctx = L1.Ctx(images={}, states=[0.0] * 14, eef={'left': ([0, 0, 1], [1, 0, 0, 0]), 'right': ([0, 0, 1], [1, 0, 0, 0])}, last_grip_cmd={'left': 0.0}, cfg={})
ctx.states[6] = 0.3
r = L1.call('insert_charger', ctx); R['2_insert_no_service'] = dict(reply=r, contract=check(r), ok=r['abstain'] and r['steps_used'] == 0)
ctx.cfg['infer_url_v4'] = 'http://127.0.0.1:9'   # unreachable on purpose
r = L1.call('insert_charger', ctx); R['2_insert_unreachable'] = dict(note=r['note'], ok=r['abstain'] and r['steps_used'] == 0)
R['2_guard'] = dict(auto=L3.allow_call('insert_charger', {}, False)[0], explicit=L3.allow_call('insert_charger', {}, True)[0], policy=L3.allow_call('insert_charger', {}, L3.POLICY_TOKEN)[0])
# 3 segment_small vs labels
SS = Path('/lizekai/RSIws/experiments/smallseg-20261029'); seg = []
lab = sorted(glob.glob(str(SS / 'labels/ctx-*.json'))); random.Random(0).shuffle(lab)
want = {('coin', 'cam_left_wrist'): 0, ('coin', 'cam_right_wrist'): 0, ('coin', 'cam_high'): 0, ('mahjong', 'cam_high'): 0}
for f in lab:
    d = json.load(open(f))
    if d['split'] != 'TEST': continue
    for v in d['views']:
        objs = [o for o in (v.get('instances') or v.get('objects') or []) if (o.get('visible') or o.get('visible_px', 1))]
        cls = {o.get('cls') or o.get('class') for o in objs}
        for (ob, cam) in want:
            if want[(ob, cam)] >= 2 or v.get('cam') != cam or ob not in cls: continue
            img = SS / 'data' / v['image']; c2 = L1.Ctx(images={cam: img}, states=[0.0] * 14, eef={}, cfg={'device': 'cpu'})
            r = L1.call('segment_small', c2, object=ob, camera=cam); want[(ob, cam)] += 1
            seg.append(dict(view=v['image'], object=ob, camera=cam, n_truth=sum((o.get('cls') or o.get('class')) == ob for o in objs), n_found=(r['value'] or {}).get('n_instances'), contract=check(r), note=r['note'][:90]))
    if all(x >= 2 for x in want.values()): break
c3 = L1.Ctx(images={'cam_high': np.zeros((480, 640, 3), np.uint8)}, states=[0.0] * 14, eef={}, cfg={'device': 'cpu'})
R['3_segment'] = dict(views=seg, key_not_available=L1.call('segment_small', c3, object='key', camera='cam_high')['abstain'])
# 4 gate v2 reproduction from archived per-step history
sys.path.insert(0, '/lizekai/RSIws/experiments/macro-tools-20261027/gate'); import build_dataset_v2 as BD, train_eval_v2 as TE
G = P.GateV2(); rows = [r for r in TE.D if r['kind'] == 'rg3f_comp']; diffs = []; dec = {}
for r in rows:
    inner = [p for p in (BD.NAT / r['episode']).iterdir() if p.is_dir() and (p / 'action_000000.json').exists()][0]; acts = []
    for k in range(r['t'] + 1):
        f = inner / ('action_%06d.json' % k)
        if not f.exists(): break
        acts.append(json.load(open(f)))
    S = [a['obs']['states'] for a in acts if (a.get('obs') or {}).get('states')]; A = [a['executed_action'] for a in acts]
    d = G.decide(dict(task=r['task'], step=r['t'], step_limit=400, states_hist=S, cmd_hist=A, executor='insert_charger'))
    x = np.asarray(TE.x_of(r)); p_ref = float(1 / (1 + np.exp(-(G.w @ x + G.b))))
    diffs.append(abs((d['p'] if d['p'] is not None else -1) - p_ref)); dec[d['decision']] = dec.get(d['decision'], 0) + 1
R['4_gate'] = dict(n=len(rows), max_abs_p_diff=round(max(diffs), 5), decisions=dec, ok=max(diffs) < 1e-3)
# 5 harness glue on one archived rg3f state (per-step history callback)
r0 = rows[0]; inner = [p for p in (BD.NAT / r0['episode']).iterdir() if p.is_dir() and (p / 'action_000000.json').exists()][0]
acts = [json.load(open(inner / ('action_%06d.json' % k))) for k in range(r0['t'] + 1) if (inner / ('action_%06d.json' % k)).exists()]
S = [a['obs']['states'] for a in acts if (a.get('obs') or {}).get('states')]; A = [a['executed_action'] for a in acts]
out5 = {}
for mode in ('log', 'offer', 'auto'):
    h = HN.Harness(dict(task='plug_in_charger', native_step_limit=400, step_history=lambda: (S, A), l3_policy=dict(name='gate_v2', mode=mode, executor='insert_charger')))
    c = L1.Ctx(images={}, states=S[-1], eef={'left': ([0, 0, 1], [1, 0, 0, 0]), 'right': ([0, 0, 1], [1, 0, 0, 0])}, step=r0['t'], max_steps=800, cfg={})
    items = h.after_act(c, {}); items2 = h.after_act(c, {})
    out5[mode] = dict(items=[i['text'] for i in items], second_turn=len(items2), mode_used=h.pol_mode, log=h.pol_log[:3], filter_log=h.gate.log)
R['5_glue'] = out5
json.dump(R, open(H / 'out/SMOKE-l1l3-1029.json', 'w'), indent=1, default=str)
print(json.dumps({k: (v if k != '3_segment' else v) for k, v in R.items() if k != '1_specs'}, default=str)[:3500]); print(R['1_specs'])
