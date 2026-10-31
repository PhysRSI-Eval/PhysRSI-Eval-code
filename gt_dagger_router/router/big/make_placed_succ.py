"""Adopt the 'placed' label fix (RESULT-stage-k1-placed-succ.json: placed F1 val .33 -> .58, test .65 -> .82): Astra K1 rows keep stage 5
only in natively successful episodes (else -1 = unknown; 70 % of K1 placed frames were in failed episodes = object left near the target).
Demo rows unchanged.  python make_placed_succ.py IN.npz OUT.npz"""
import sys
import numpy as np
Z = np.load(sys.argv[1], allow_pickle=True); out = {k: Z[k] for k in Z.files}; src, su = out['src'], out['succ'].astype(float)
out['stage'] = np.array([np.where((s == 5) & (su[i] != 1), -1, s).astype(np.int8) if src[i] == 'astra_k1' else s for i, s in enumerate(out['stage'])], dtype=object)
np.savez_compressed(sys.argv[2], **out); print('ok', sys.argv[2])
