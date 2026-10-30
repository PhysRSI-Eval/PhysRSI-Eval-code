"""finepose CAD label renderer (10-29, user: train the segmentation/localisation model directly from the key and lock CAD). Rasterises the exported
CAD meshes (cad/key.npz, cad/lock.npz) at given poses into a camera (USD convention: looks along -Z, +Y up) with a z-buffer -> label image
0 background / 1 lock / 2 key (+ per-pixel depth). Robot fingers are not modelled (they occlude the key bow in some views).
Key pose = the sim key frame (bbox centre, cl = (0.0021, 0, -0.02135) from the asset origin); lock pose = asset origin (bottom centre)."""
import numpy as np
from pathlib import Path
C = Path(__file__).resolve().parent.parent / 'cad'
KEY = {k: v for k, v in np.load(C / 'key.npz').items()}; LOCK = {k: v for k, v in np.load(C / 'lock.npz').items()};  # eager (lazy NpzFile breaks under fork)
KEY_CL = np.array([0.0021, 0.0, -0.02135])
KEYHOLE_KP = np.array([[-0.002, 0, 0.055], [0.012, 0, 0.055], [-0.0064, 0, 0.055]])  # lock frame: round-hole centre, slot end (+x), far rim of the round hole


def _cam(Kin, T, P):
    pc = (np.linalg.inv(T) @ np.c_[P, np.ones(len(P))].T).T[:, :3]; z = -pc[:, 2]
    return np.c_[Kin[0, 0] * pc[:, 0] / z + Kin[0, 2], -Kin[1, 1] * pc[:, 1] / z + Kin[1, 2]], z


def render(Kin, T_cam, objs, W=640, H=480):
    """objs: [(label, vertices_world (N,3), faces (M,3))] -> (labels HxW uint8, depth HxW)."""
    Kin = np.asarray(Kin, float); T_cam = np.asarray(T_cam, float); lab = np.zeros((H, W), np.uint8); dep = np.full((H, W), np.inf)
    for label, Vw, Fc in objs:
        uv, z = _cam(Kin, T_cam, Vw)
        for f in Fc:
            if np.any(z[f] <= 1e-4): continue
            p = uv[f]; x0, y0 = np.floor(p.min(0)).astype(int); x1, y1 = np.ceil(p.max(0)).astype(int)
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W - 1), min(y1, H - 1)
            if x0 > x1 or y0 > y1: continue
            xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5); a, b, c = p
            den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(den) < 1e-12: continue
            w0 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / den; w1 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / den; w2 = 1 - w0 - w1
            m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not m.any(): continue
            zz = 1.0 / (w0 / z[f[0]] + w1 / z[f[1]] + w2 / z[f[2]]); sub = dep[y0:y1 + 1, x0:x1 + 1]; upd = m & (zz < sub)
            sub[upd] = zz[upd]; lab[y0:y1 + 1, x0:x1 + 1][upd] = label
    return lab, dep


def key_world(K_pose): K_pose = np.asarray(K_pose); return (K_pose[:3, :3] @ (KEY['vertices'] - KEY_CL).T).T + K_pose[:3, 3]
def lock_world(T_lock): T_lock = np.asarray(T_lock); return (T_lock[:3, :3] @ LOCK['vertices'].T).T + T_lock[:3, 3]
def labels_for(rec, T_lock):
    return render(rec['intrinsics'], rec['extrinsics_cam_to_world'], [(1, lock_world(T_lock), LOCK['faces']), (2, key_world(rec['key_pose']), KEY['faces'])])
