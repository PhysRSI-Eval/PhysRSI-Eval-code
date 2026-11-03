"""in-hand pose v4 shared geometry (2026-10-28, post-training line). Everything is expressed in the link6 frame of the holding arm.
- wrist camera: fixed mount in link6 (objects_v2.json wrist_cam, as geomM), image used at 320 x 240;
- head camera: static calibration (intrinsics + OpenGL cam-to-world), image crop of 0.24 m at the TCP depth centred on the FK-projected TCP,
  resized to 160 x 160. Link6 pose L comes from joints + robot kinematics (deploy-time proprio), never the replay camera RPC.
- CAD prior: cad/objects_cad_v4.json (asset mesh in the metadata frame); axial objects (test_tube, mallet) use 16 points on the axis."""
import json, os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

V4 = Path(os.environ.get('PHYSRSI_INHAND_V4', '/lizekai/RSIws/experiments/inhand-pose-20261026/v4'))
CAM = json.load(open(os.environ.get('PHYSRSI_WRIST_CALIB', '/lizekai/RSIws/experiments/inhand-pose-20261026/train_v2/objects_v2.json')))['wrist_cam']
R_LC = R.from_rotvec(CAM['rotvec_link6_cam']).as_matrix(); T_LC = np.asarray(CAM['t_link6_cam'], float)
FXW, CXW, CYW = CAM['fx_320'], CAM['cx_320'], CAM['cy_320']          # wrist intrinsics at 320 x 240
TCP = np.array([0.14, 0.0, 0.0]); HEAD_M = 0.24; HS = 160; WW, WH = 320, 240; K = 16
AXIAL = ('test_tube', 'mallet')
HEAD_CAM = None


def head_cam():
    global HEAD_CAM
    if HEAD_CAM is None: HEAD_CAM = json.load(open(V4 / 'cad' / 'head_cam.json'))
    return HEAD_CAM


def category(variant): return variant.rsplit('_', 1)[0]


def load_cad():
    d = json.load(open(V4 / 'cad' / 'objects_cad_v4.json')); out = {}
    for var, c in d.items():
        lo, hi = np.asarray(c['bbox_lo']), np.asarray(c['bbox_hi']); diag = float(np.linalg.norm(hi - lo)); ctr = (lo + hi) / 2
        ax = bool(category(var) in AXIAL)
        if ax:  # symmetry axis = long bbox axis through the bbox centre
            i = int(np.argmax(hi - lo)); kp = np.tile(ctr, (K, 1)); kp[:, i] = np.linspace(lo[i], hi[i], K); axis = np.eye(3)[i]
        else:
            kp = np.asarray(c['keypoints'], float); axis = None
        pts = np.asarray(c['surface_points'], float)[:512]
        out[var] = dict(kp=kp, pts=pts, diag=diag, ctr=ctr, axes=np.asarray(c['principal_axes'], float), axial=ax, axis=axis)
    return out


def T_of(p, q):
    M = np.eye(4); M[:3, :3] = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); M[:3, 3] = p; return M


def wrist_uv(P):
    """link6-frame points (N,3) -> wrist pixels at 320x240 (N,2), depth (N,)."""
    pc = (np.asarray(P, float) - T_LC) @ R_LC; z = pc[:, 2]
    return np.stack([FXW * pc[:, 0] / np.maximum(z, 1e-4) + CXW, FXW * pc[:, 1] / np.maximum(z, 1e-4) + CYW], 1), z


def head_full_uv(Pw):
    """world points -> head pixels at 640x480, depth."""
    hc = head_cam(); Kh = np.asarray(hc['intrinsics']); Tw2c = np.linalg.inv(np.asarray(hc['extrinsics_cam_to_world']))
    q = np.asarray(Pw, float) @ Tw2c[:3, :3].T + Tw2c[:3, 3]; z = -q[:, 2]
    return np.stack([Kh[0, 0] * q[:, 0] / z + Kh[0, 2], -Kh[1, 1] * q[:, 1] / z + Kh[1, 2]], 1), z


def head_crop(L):
    """crop (x0, y0, S) in 640x480 pixels centred on the projected TCP; S covers HEAD_M metres at the TCP depth."""
    tcp = L[:3, :3] @ TCP + L[:3, 3]; uv, z = head_full_uv(tcp[None]); f = np.asarray(head_cam()['intrinsics'])[0, 0]
    S = float(f * HEAD_M / max(z[0], 0.2)); return float(uv[0, 0] - S / 2), float(uv[0, 1] - S / 2), S


def head_uv(P, L, crop):
    """link6-frame points -> head crop pixels (HS x HS)."""
    Pw = np.asarray(P, float) @ L[:3, :3].T + L[:3, 3]; uv, z = head_full_uv(Pw); x0, y0, S = crop
    return (uv - np.array([x0, y0])) * (HS / S), z


def rays_wrist(st=8):
    """unit ray directions (link6 frame) for the wrist feature grid at stride st and the camera origin."""
    ys, xs = np.mgrid[0:WH // st, 0:WW // st]; u = (xs + 0.5) * st; v = (ys + 0.5) * st
    d = np.stack([(u - CXW) / FXW, (v - CYW) / FXW, np.ones_like(u, float)], -1) @ R_LC.T
    return d / np.linalg.norm(d, axis=-1, keepdims=True), T_LC.copy()


def rays_head(L, crop, st=8):
    hc = head_cam(); Kh = np.asarray(hc['intrinsics']); C2W = np.asarray(hc['extrinsics_cam_to_world']); x0, y0, S = crop
    n = HS // st; ys, xs = np.mgrid[0:n, 0:n]; u = x0 + (xs + 0.5) * st * S / HS; v = y0 + (ys + 0.5) * st * S / HS
    dc = np.stack([(u - Kh[0, 2]) / Kh[0, 0], -(v - Kh[1, 2]) / Kh[1, 1], -np.ones_like(u, float)], -1)  # OpenGL camera
    dw = dc @ C2W[:3, :3].T; dl = dw @ L[:3, :3]; o = L[:3, :3].T @ (C2W[:3, 3] - L[:3, 3])
    return dl / np.linalg.norm(dl, axis=-1, keepdims=True), o


def rot_err_deg(Rp, Rt, axis=None):
    if axis is not None:
        a, b = Rp @ axis, Rt @ axis; return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))
    return float(np.degrees((R.from_matrix(Rp) * R.from_matrix(Rt).inv()).magnitude()))
