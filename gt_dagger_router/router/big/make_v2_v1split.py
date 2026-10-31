"""K1 stage drop check (.830 v1 -> .750 v2): the source-balanced sampling is identical by construction (4 sources, bs // 4 each in both),
but the dev split changed (groups reshuffled after the demo count went 807 -> 900: K1 dev 100 -> 126 episodes, 21 / 37 v1 dev contexts
overlap). One change: big_v2.npz with v1's dev flags per group (groups absent from v1 -> train).  -> big_v2_v1split.npz"""
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; a = np.load(H / 'big_v1.npz', allow_pickle=True); b = np.load(H / 'big_v2.npz', allow_pickle=True)
d1 = {g: bool(d) for g, d in zip(a['group'], a['dev'])}; out = {k: b[k] for k in b.files}; out['dev'] = np.array([d1.get(g, False) for g in b['group']], dtype=object)
np.savez_compressed(H / 'big_v2_v1split.npz', **out); s = b['src']; dv = out['dev'].astype(bool); print({k: (int((s == k).sum()), int((dv & (s == k)).sum())) for k in sorted(set(s))})
