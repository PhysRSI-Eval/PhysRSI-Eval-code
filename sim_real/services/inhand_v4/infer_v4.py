"""In-hand pose v4, CHARGER ONLY (PREREG-IP-v4 verdict PARTIAL: charger PASS 0.50 mm / 0.60 deg on held-out layouts, coverage 97-99 %).
Deploy inputs: head RGB (cam_high, 640x480) + holding-arm wrist RGB + the holding arm's link6 pose (FK / eef) + gripper reading; the
charger CAD (deploy prior). Same call as infer_v3 (model_v3/pkg): infer(est, wrist, high, link6_pos, link6_quat_wxyz, gripper_state,
gripper_cmd, arm, category) -> dict, plus L1-format fields.
Output frame: TCP frame = link6 axes, origin link6 + R_link6 [0.14, 0, 0] (as v3). abstain = the frozen v4 flag (flip check, keypoint
residual, member spread; CALIB-IP-v4.json); flip_flag = the orientation part of it (flip margin or regression/solution disagreement).
    import sys; sys.path.insert(0, '/lizekai/RSIws/experiments/inhand-pose-20261026/v4/tool'); import infer_v4 as IV
    est = IV.load('cuda'); out = IV.infer(est, wrist='w.jpg', high='h.jpg', link6_pos=[...], link6_quat_wxyz=[...], gripper_state=0.4, arm='left')"""
import hashlib, json, sys, os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
V4 = Path(os.environ.get('PHYSRSI_INHAND_V4', '/lizekai/RSIws/experiments/inhand-pose-20261026/v4')); sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_v4 as LB  # noqa: E402
import solver_v4 as SV  # noqa: E402
import train_v4 as T  # noqa: E402
FROZEN = Path(__file__).with_name('FROZEN-v4-charger.json'); VARIANT = 'charger_0'; ST = 16


def _sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


class Est:
    pass


def load(device='cuda', verify=True):
    fz = json.load(open(FROZEN)); e = Est(); e.dev = torch.device(device); e.nets = []
    for m in fz['members']:
        p = V4 / m['path']
        if verify: assert _sha(p) == m['sha256'], 'sha256 mismatch %s' % p
        n = T.Net('dinov3', 8).to(e.dev); n.load_state_dict(torch.load(p, map_location=e.dev)['state']); n.eval(); e.nets.append(n)
    e.calib = json.load(open(V4 / 'results/CALIB-IP-v4.json')); c = LB.load_cad()[VARIANT]; e.c = c
    e.kp = torch.tensor((c['kp'] - c['ctr']) / c['diag'], dtype=torch.float32, device=e.dev)[None]; e.pts = torch.tensor((c['pts'] - c['ctr']) / c['diag'], dtype=torch.float32, device=e.dev)[None]
    rw, ow = LB.rays_wrist(ST); e.rw = torch.tensor(rw.reshape(1, -1, 3), dtype=torch.float32, device=e.dev); e.ow = torch.tensor(ow[None], dtype=torch.float32, device=e.dev)
    return e


def _img(x):
    return Image.open(x).convert('RGB') if isinstance(x, (str, Path)) else Image.fromarray(np.asarray(x, np.uint8)[..., :3])


@torch.no_grad()
def infer(est, wrist, high, link6_pos, link6_quat_wxyz, gripper_state, gripper_cmd=0.0, arm='left', category='charger'):
    if category != 'charger':
        return dict(value=None, uncertainty=None, abstain=True, steps_used=0, note='in-hand pose v4 is validated for the charger only')
    d = est.dev; L = LB.T_of(link6_pos, link6_quat_wxyz); wi = _img(wrist).resize((LB.WW, LB.WH), Image.BILINEAR); hi = _img(high); s = hi.size[0] / 640.0
    x0, y0, S = LB.head_crop(L); hc = hi.transform((LB.HS, LB.HS), Image.EXTENT, (x0 * s, y0 * s, (x0 + S) * s, (y0 + S) * s), Image.BILINEAR); crop = np.array([x0, y0, S], np.float32)
    w = torch.tensor(np.asarray(wi), device=d).permute(2, 0, 1)[None].float() / 255; h = torch.tensor(np.asarray(hc), device=d).permute(2, 0, 1)[None].float() / 255
    rh, oh = LB.rays_head(L, crop.astype(float), ST); prop = torch.tensor(np.concatenate([L[:3, :3].flatten(), L[:3, 3], [gripper_state]])[None], dtype=torch.float32, device=d)
    b = dict(w=w, h=h, rw=est.rw, ow=est.ow, rh=torch.tensor(rh.reshape(1, -1, 3), dtype=torch.float32, device=d), oh=torch.tensor(oh[None], dtype=torch.float32, device=d), kp=est.kp, pts=est.pts, prop=prop)
    P = []
    for n in est.nets:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=d.type == 'cuda'): uvw, uvh, pw, ph, sig, pose, hw, hh = T.run(n, b)
        P.append(dict(uvw=uvw.float().cpu().numpy(), uvh=uvh.float().cpu().numpy(), bw=(sig[..., 0].clamp(-3, 5).exp() * 2).float().cpu().numpy(),
                      bh=(sig[..., 1].clamp(-3, 5).exp() * 2).float().cpu().numpy(), R0=T.rot6_to_mat(pose[:, :6]).float().cpu().numpy(),
                      t0=(pose[:, 6:] * 0.1 + torch.tensor(LB.TCP, device=d, dtype=torch.float32)).float().cpu().numpy()))
    E = dict(uvw=np.mean([p['uvw'] for p in P], 0), uvh=np.mean([p['uvh'] for p in P], 0), bw=np.mean([p['bw'] for p in P], 0) + np.abs(P[0]['uvw'] - P[-1]['uvw']).sum(-1) / 2,
             bh=np.mean([p['bh'] for p in P], 0) + np.abs(P[0]['uvh'] - P[-1]['uvh']).sum(-1) / 2, R0=P[0]['R0'], t0=np.mean([p['t0'] for p in P], 0))
    hc_ = LB.head_cam(); wr = dict(R_LC=LB.R_LC, T_LC=LB.T_LC, fx=LB.FXW, cx=LB.CXW, cy=LB.CYW); f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64, device=d)  # noqa: E731
    c = est.c; geo = SV.Geo(L[None], crop[None].astype(float), hc_, wr, d)
    def solve(Q):
        return SV.solve(geo, f(c['kp'][None]), f(c['ctr'][None]), f(np.asarray(c['axes'])[None]), f(np.zeros((1, 3))), torch.zeros(1, dtype=torch.bool, device=d),
                        f(Q['uvw']), f(Q['bw']), f(Q['uvh']), f(Q['bh']), f(Q['R0']), f(Q['t0']))
    R, t, cst, mrg = [x.cpu().numpy()[0] for x in solve(E)]; Rm = [solve(p)[0].cpu().numpy()[0] for p in P]
    spread = max(LB.rot_err_deg(Rm[0], x) for x in Rm); regd = LB.rot_err_deg(R, P[0]['R0'][0])
    X = c['kp'] @ R.T + t; uw, zw = LB.wrist_uv(X); uh, _ = LB.head_uv(X, L, crop); ok = zw > 0.01
    rms = float(np.sqrt((((uw - E['uvw'][0]) ** 2).sum(-1)[ok].sum() + ((uh - E['uvh'][0]) ** 2).sum(-1).sum()) / (ok.sum() + len(uh))))
    k = est.calib; flip_flag = bool(mrg < k['m_flip'] or regd > 30 or spread > 30)
    abstain = bool(flip_flag or rms / k['tau_px'] > 1 or spread / k['tau_deg'] > 1)
    q = np.roll(__import__('scipy.spatial.transform', fromlist=['Rotation']).Rotation.from_matrix(R).as_quat(), 1)
    pos_tcp = (t - LB.TCP); val = dict(position_in_tcp_frame_m=[round(float(v), 5) for v in pos_tcp], quaternion_in_tcp_frame_wxyz=[round(float(v), 5) for v in q])
    unc = dict(keypoint_rms_px=round(rms, 2), member_spread_deg=round(float(spread), 2), flip_margin=round(float(mrg), 3) if np.isfinite(mrg) else None,
               regression_vs_solution_deg=round(float(regd), 2), tau_px=k['tau_px'], tau_deg=k['tau_deg'], m_flip=k['m_flip'])
    note = ('unsure: possible 180 deg flip of the charger in the hand' if flip_flag else ('unsure: keypoint fit or member agreement outside the calibrated range' if abstain
            else 'in-hand pose v4 (charger), 0 control steps'))
    return dict(value=val if not abstain else dict(position_in_tcp_frame_m=val['position_in_tcp_frame_m']) if not flip_flag else None, uncertainty=unc, abstain=abstain,
                flip_flag=flip_flag, steps_used=0, note=note, full=dict(R_object_in_link6=R.tolist(), t_object_in_link6_m=t.tolist(), **val))
