"""Pack the stage shards into one compact array file for the cluster run and the H200 push: stage_k1_v1.npz (X object array of (L,3,1536)
float16 per episode, Y int8, meta JSON)."""
import glob, json
import numpy as np
X, Y, M = [], [], []
for f in sorted(glob.glob('feat/shard-*.npz')):
    z = np.load(f, allow_pickle=True); X += list(z['X']); Y += list(z['Y']); M += json.load(open(f[:-4] + '.json'))
np.savez_compressed('stage_k1_v1.npz', X=np.array(X, dtype=object), Y=np.array(Y, dtype=object), meta=json.dumps(M)); print(len(X), sum(len(y) for y in Y))
