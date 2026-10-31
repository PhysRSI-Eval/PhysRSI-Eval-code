"""Smoke test of policy.py (C1 harness contract) in the sim driver python: load(ckpt) -> act(obs) on 3 official VAL frames of each K1 task
built exactly as the harness will (480x640 RGB from the HDF5 JPEGs, state14, instruction, step). Checks: output kind/shape, chunk vs GT
action error, and the recomputed DINOv3 features vs the router cache (same function, same weights) -> one JSON line."""
import io, json, sys, time
from pathlib import Path
import h5py, numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent)); import importlib, bcnet as B  # noqa: E402
P = importlib.import_module(sys.argv[2] if len(sys.argv) > 2 else 'policy'); T0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0; EXN = int(sys.argv[4]) if len(sys.argv) > 4 else 8
ck = sys.argv[1]; pi = P.load(ck, EXN); split = json.load(open(B.G / 'split.json')); out = {}
F = ['left_arm_joint_states', 'left_ee_joint_states', 'right_arm_joint_states', 'right_ee_joint_states']; CAM = {'cam_high': 'cam_head', 'cam_left_wrist': 'cam_left_wrist', 'cam_right_wrist': 'cam_right_wrist'}
for task in B.K1:
    ep = split[task]['val'][0]; f = h5py.File('/lizekai/RSIws/data/official/sim_train/%s/%s.hdf5' % (task, ep), 'r'); st = np.concatenate([f['state/' + k][:] for k in F], -1); ac = np.concatenate([f['action/' + k][:] for k in F], -1)
    ins = f['instruction'][()].decode(); z = np.load(B.FEAT / task / (ep + '.npz')); e = []; fe = []; t0 = time.time()
    pi.step = None
    for t in range(T0, T0 + 88, EXN):  # 与 harness 同节奏：每 exec_n = 8 步查询一次，act() 内部把上一块前 8 步记为已执行
        ims = {a: np.asarray(Image.open(io.BytesIO(f['vision/%s/colors' % b][t])).convert('RGB')) for a, b in CAM.items()}
        r = pi.act(dict(images=ims, state=st[t], eef=None, instruction=ins, step=t)); assert r['kind'] == 'joint' and r['actions'].shape[1] == 14
        if t not in (T0, T0 + 40, T0 + 80): continue
        H = min(len(r['actions']), len(ac) - t); e.append(float(np.abs(r['actions'][:H] - ac[t:t + H])[:, B.ARMM > 0].mean()))
        Xr = P.RR.dino_features(pi.pi.vit, pi.pi.nreg, ims, pi.pi.dev); fe.append(float(np.abs(Xr - z['X'][t // 2].astype(np.float32)).max()))
    out[task] = dict(chunk_arm_rad=[round(x, 4) for x in e], feat_maxdiff_vs_cache=round(max(fe), 4), shape=list(r['actions'].shape), sec_per_call=round((time.time() - t0) / 11, 2))
print(json.dumps(out))
