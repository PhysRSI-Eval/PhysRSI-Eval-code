"""Stage v2 data: + per-frame deploy-legal proprio for each packed episode (astra_v1 arrays): per arm gripper reading, last gripper
command, steps since the close / 100, eef z, eef z velocity and xy speed over 4 steps (x100), z change since the close (x10).
  python add_prop.py IN.npz OUT.npz"""
import json, sys
from pathlib import Path
import numpy as np
A = Path('/lizekai/RSIws/experiments/mva-20261029/astra_v1'); Z = np.load(sys.argv[1], allow_pickle=True); META = json.loads(str(Z['meta'])); P = []
for m in META:
    d = np.load(A / 'arrays' / m['task'] / ('%s.npz' % m['episode'])); T = int(d['T']); rows = []
    feats = {}
    for arm in ('left', 'right'):
        g = d['state/%s_ee_joint_states' % arm][:, 0]; c = d['action/%s_ee_joint_states' % arm][:, 0]; p = d['state/%s_ee_poses' % arm][:, :3]
        since = np.full(T, -1.0); zc = np.zeros(T); cur = None
        for t in range(T):
            closed = t > 0 and c[t - 1] < 0.5
            if closed and cur is None: cur = t
            if not closed: cur = None
            since[t] = (t - cur) / 100.0 if cur is not None else -1.0; zc[t] = (p[t, 2] - p[cur, 2]) * 10 if cur is not None else 0.0
        feats[arm] = (g, c, since, p, zc)
    for t in m['steps']:
        r = []
        for arm in ('left', 'right'):
            g, c, since, p, zc = feats[arm]; t0 = max(t - 4, 0)
            r += [g[t], c[t - 1] if t > 0 else 1.0, since[t], p[t, 2], (p[t, 2] - p[t0, 2]) * 100 / 4, np.linalg.norm(p[t, :2] - p[t0, :2]) * 100 / 4, zc[t]]
        rows.append(r)
    P.append(np.asarray(rows, np.float32))
np.savez_compressed(sys.argv[2], X=Z['X'], Y=Z['Y'], P=np.array(P, dtype=object), meta=Z['meta']); print(len(P))
