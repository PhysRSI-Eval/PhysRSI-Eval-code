"""Adopt 'grasp-lift' (dispatcher 10-31, after RESULT-stage-k1-merge-gl.json: merged F1 .815 / .835 val / test, other stages unchanged):
big_v3.npz with stage label 2 (lift, object raised 1-3 cm) mapped to 1 (grasp) for every source.  -> big_v3m.npz"""
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; Z = np.load(H / 'big_v3.npz', allow_pickle=True); out = {k: Z[k] for k in Z.files}
out['stage'] = np.array([np.where(s == 2, 1, s).astype(np.int8) for s in out['stage']], dtype=object); np.savez_compressed(H / 'big_v3m.npz', **out); print('ok', len(out['stage']))
