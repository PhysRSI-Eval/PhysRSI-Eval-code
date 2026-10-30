"""keypoint factory API (2026-10-29, post-training line). Deploy inputs only: RGB images, camera calibration (FK link6 x mount, static
head), the trained KPNet (kpnet3.Detector), and the instance's keypoint table (names).
  detect(ckpt, images)                    -> per image [(u, v, vis_prob, peak)] x K
  triangulate(dets, cams, min_vis=0.5)    -> (K, 3) world points (NaN where < 2 views see the point), per-point ray residual (m)
  plug_relative(P, outlet)                -> prong-tip -> hole-entry error for outlet k: dxy (m, world xy, mean of the two prongs),
                                             dz (tip height above the hole entry, m), dyaw (deg, prong pair vs hole pair, mod 180), pairing
Camera convention: OpenGL (looks along -Z, +Y up); a camera is (K 3x3, T_cam_to_world 4x4).
Keypoint order for plug_in_charger (make_labels_kp.NAMES): tip_l, tip_r, root_l, root_r, face_c, then per outlet k: hole_a, hole_b, centre."""
import math
import numpy as np

_DET = {}


def detect(ckpt, images, gpu=0):
    import sys
    sys.path.insert(0, '/lizekai/RSIws/experiments/rung1-newtargets-20261027/src'); import kpnet3
    if ckpt not in _DET: _DET[ckpt] = kpnet3.Detector(ckpt, gpu)
    return [_DET[ckpt](im) for im in images]


def ray(K, T, uv):
    K = np.asarray(K); T = np.asarray(T); d = np.array([(uv[0] - K[0][2]) / K[0][0], -(uv[1] - K[1][2]) / K[1][1], -1.0]); d = T[:3, :3] @ d
    return T[:3, 3], d / np.linalg.norm(d)


def triangulate(dets, cams, min_vis=0.5):
    nk = len(dets[0]); P = np.full((nk, 3), np.nan); res = np.full(nk, np.nan)
    for k in range(nk):
        rays = [ray(Kc, Tc, d[k][:2]) for d, (Kc, Tc) in zip(dets, cams) if d[k][2] >= min_vis]
        if len(rays) < 2: continue
        A = np.zeros((3, 3)); b = np.zeros(3)
        for o, dv in rays: M = np.eye(3) - np.outer(dv, dv); A += M; b += M @ o
        if np.linalg.cond(A) > 1e8: continue
        X = np.linalg.solve(A, b); P[k] = X; res[k] = float(np.mean([np.linalg.norm((np.eye(3) - np.outer(dv, dv)) @ (X - o)) for o, dv in rays]))
    return P, res


def plug_relative(P, outlet):
    tips = P[[0, 1]]; holes = P[[5 + 3 * outlet, 6 + 3 * outlet]]
    if np.isnan(tips).any() or np.isnan(holes).any(): return None
    best = min(((0, 1), (1, 0)), key=lambda pr: np.linalg.norm(tips[0, :2] - holes[pr[0], :2]) + np.linalg.norm(tips[1, :2] - holes[pr[1], :2]))
    H = holes[list(best)]; dxy = (H[:, :2] - tips[:, :2]).mean(0); dz = float((tips[:, 2] - H[:, 2]).mean())
    vt = tips[1, :2] - tips[0, :2]; vh = H[1, :2] - H[0, :2]; a = math.degrees(math.atan2(vh[1], vh[0]) - math.atan2(vt[1], vt[0])); a = (a + 90) % 180 - 90
    return dict(dxy_m=dxy.tolist(), dz_m=dz, dyaw_deg=a, pairing=list(best))
