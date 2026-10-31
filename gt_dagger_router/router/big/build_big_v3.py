"""ROUTER-BIG v3 dataset (GT directive 10-31) = big_v2.npz features unchanged + two GT-correctness changes:
 (1) fixed episode split from GT.md 6.1 for the official demos: episode 0-79 train, 80-89 val, 90-99 test (per task); other sources:
     groups (ctx / episode) shuffled with random.Random(20261031), 80 / 10 / 10 -> 'split' (0 train, 1 val, 2 test).
 (2) demos carry NO success label (GT.md 8: the files have no explicit success label; do not mark every demo as success) ->
     success / failure heads learn only from Astra native outcomes (K1 stage episodes). Demo stage stays the proprio-derived label (flagged).
 Recovery label ('take') unchanged: 0 Astra, 1 rg3f, 2 VS (fitmap v1); class 3 = BC policy is reserved for the DAgger rollouts.  -> big_v3.npz"""
import random, re
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; Z = np.load(H / 'big_v2.npz', allow_pickle=True); out = {k: Z[k] for k in Z.files}
src, grp = Z['src'], Z['group']; split = np.zeros(len(src), np.int8)
og = sorted({g for g, s in zip(grp, src) if s != 'demo'}); random.Random(20261031).shuffle(og); n = len(og)
gs = {g: 0 if i < 0.8 * n else 1 if i < 0.9 * n else 2 for i, g in enumerate(og)}
succ = np.array(Z['succ'], dtype=object)
for i, (g, s) in enumerate(zip(grp, src)):
    if s == 'demo':
        e = int(re.match(r'demo:episode_(\d+)', g).group(1)); split[i] = 0 if e < 80 else 1 if e < 90 else 2; succ[i] = np.nan
    else: split[i] = gs[g]
out['split'] = split; out['succ'] = succ
np.savez_compressed(H / 'big_v3.npz', **out)
import collections; print({s: collections.Counter(split[src == s].tolist()) for s in sorted(set(src))})
