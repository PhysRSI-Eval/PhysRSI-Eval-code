"""ROUTER-RGB-v1 sample list (PREREG router-rgb/PREREG-ROUTER-RGB-v1.md): astra_v1 episodes, one sample every 8 steps, proprio from
astra_v1/arrays, frames from artifacts/native observations (640x480 PNG), labels by the frozen proprio rule. -> samples.json"""
import json, os
from pathlib import Path
import numpy as np
A = Path('/lizekai/RSIws/experiments/mva-20261029/astra_v1'); NAT = Path('/lizekai/RSIws/artifacts/native'); OUT = Path(__file__).resolve().parent
C1 = {'plug_in_charger', 'insert_key', 'insert_tubes', 'deposit_coin', 'fill_pen_holder', 'store_tools_in_toolbox'}
ARMS = ('left', 'right'); S = []; skipped = {}
for line in open(A / 'MANIFEST.jsonl'):
    m = json.loads(line)
    if not m.get('valid'): continue
    ep, task = m['episode'], m['task']; af = A / 'arrays' / task / ('%s.npz' % ep)
    inner = [p for p in (NAT / ep).iterdir() if p.is_dir() and (p / 'action_000000.json').exists() and (p / 'observations').is_dir()] if (NAT / ep).is_dir() else []
    if not af.exists() or len(inner) != 1: skipped[ep] = 'no arrays / observations'; continue
    d = np.load(af); T = int(d['T']); obs = inner[0] / 'observations'
    g = {a: d['state/%s_ee_joint_states' % a][:, 0] for a in ARMS}; c = {a: d['action/%s_ee_joint_states' % a][:, 0] for a in ARMS}; z = {a: d['state/%s_ee_poses' % a][:, 2] for a in ARMS}
    zc = {a: None for a in ARMS}; lab = np.zeros(T, bool)
    for t in range(T):
        for a in ARMS:
            closed = c[a][t - 1] < 0.5 if t > 0 else False
            if closed and zc[a] is None: zc[a] = z[a][t]
            if not closed: zc[a] = None
            held = closed and (task == 'deposit_coin' or 0.02 < g[a][t] < 0.9)
            if task in C1 and held and zc[a] is not None and z[a][t] - zc[a] >= 0.02: lab[t] = True
    for t in range(0, T, 8):
        fs = [obs / ('%06d-%s.png' % (t, cam)) for cam in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')]
        if not all(f.exists() for f in fs): continue
        prop = []
        for a in ARMS: prop += list(map(float, d['state/%s_arm_joint_states' % a][t])) + [float(g[a][t]), float(c[a][t - 1]) if t > 0 else 1.0, float(z[a][t])]
        S.append(dict(episode=ep, task=task, ctx='-'.join(ep.split('-')[:3]), t=t, c1=task in C1, label=int(lab[t]), prop=prop + [t / 600.0], frames=[str(f) for f in fs]))
json.dump(dict(n=len(S), skipped=skipped, samples=S), open(OUT / 'samples.json', 'w'))
import collections
print(len(S), 'pos', sum(s['label'] for s in S), dict(collections.Counter((s['task'], s['label']) for s in S)), 'skipped', len(skipped))
