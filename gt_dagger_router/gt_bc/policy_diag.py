"""Diagnose the policy_smoke gap (harness path 0.09-0.41 rad vs teacher-forced 0.012): on VAL episode 0 of each K1 task at t = 0, 40, 80:
(a) training path: bcnet.sample with cached features + GT history; (b) policy path with the GT state/action history injected;
(c) policy path as the harness calls it (history = its own previous chunk, assumed executed). Same checkpoint."""
import io, json, sys
from pathlib import Path
import h5py, numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent)); import policy as P, bcnet as B  # noqa: E402
ck = sys.argv[1]; H = P.load(ck); pi = H.pi; split = json.load(open(B.G / 'split.json')); nm = pi.nm; out = {}
CAM = {'cam_high': 'cam_head', 'cam_left_wrist': 'cam_left_wrist', 'cam_right_wrist': 'cam_right_wrist'}
for task in B.K1:
    ep = split[task]['val'][0]; E = B.load_episode(task, ep); f = h5py.File('/lizekai/RSIws/data/official/sim_train/%s/%s.hdf5' % (task, ep), 'r'); res = {}
    for t in (0, 40, 80):
        i = t // 2; X, s, h, hm, y, m = B.sample(E, i, nm, pi.hist, pi.H, pi.target); T_ = lambda x: torch.from_numpy(np.asarray(x, np.float32)[None]).to(pi.dev)
        with torch.no_grad(): p = pi.net(T_(X), T_(s), T_(h), T_(hm), torch.tensor([B.K1.index(task)], device=pi.dev)).cpu().numpy()[0]
        ea = float(np.abs(nm.dec_y(p, E['state'][t], pi.target) - nm.dec_y(y, E['state'][t], pi.target))[:, B.ARMM > 0].mean())
        ims = {a: np.asarray(Image.open(io.BytesIO(f['vision/%s/colors' % b][t])).convert('RGB')) for a, b in CAM.items()}
        pi.reset(); pi.S = [E['state'][max(0, t - j)] for j in range(pi.hist, 0, -1)]; pi.A = [E['action'][t - j] for j in range(min(t, pi.hist), 0, -1)]
        cb = pi(dict(task=task, state=E['state'][t], images=ims)); n = min(pi.H, E['T'] - t); eb = float(np.abs(cb[:n] - E['action'][t:t + n])[:, B.ARMM > 0].mean())
        res[t] = dict(train_path=round(ea, 4), policy_gt_history=round(eb, 4))
    out[task] = res
print(json.dumps(out))
