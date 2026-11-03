"""align_target v1 (harness v2 L1; capability line, 2026-10-29). Information form of align_and_insert: no motion, 0 control steps.

Given the target frame from locate_mm (power-strip outlets: strip origin xy, top z, yaw) and the held object's pose in the gripper from
inhand_pose (charger in the TCP frame), plus the current end-effector pose (FK / proprioception), returns for each outlet the gripper (link6)
poses that put the charger (a) on the outlet axis, prongs `hover_m` above the seated depth ("hover"), and (b) at the seated depth ("seated"),
with propagated uncertainty and abstain. Facts only.

Model (object CAD prior + one robot constant): seated charger pose in the strip frame S_k = [SEAT_R | (OFFS[k], OUT_Y, SEAT_P)] (from 6 native
successes), strip origin = top z - SOCKET_TOP; link6 target = T_strip @ S_k (+ hover along world z) @ inv(G), G = charger in link6 =
[R_q | TCP + p].
Uncertainty (1 sigma, at the charger): position = sqrt(sigma_loc^2 + sigma_inhand_pos^2 + (r_prong * sigma_inhand_rot)^2), axis = sigma_inhand_rot,
where sigma_loc comes from the locate result (fit residual -> rung-1 statistic) and the in-hand sigmas from the estimator (member spread when
no sigma is given). Abstain when an input abstains, or the predicted position sigma > max_pos_mm or axis sigma > max_axis_deg.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

TCP = np.array([0.14, 0.0, 0.0]); OFFS = (-0.0295, 0.0070, 0.0435); OUT_Y = 0.0064; SEAT_P = 0.0202; SOCKET_TOP = 0.02157
SEAT_R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
R_PRONG = 0.017   # charger origin -> prong tips (m): CAD bbox y_min -0.0152..-0.0169 (insertion axis = charger -y); lever arm for the in-hand rotation error


def _T(p, Rm): M = np.eye(4); M[:3, :3] = Rm; M[:3, 3] = p; return M
def _q(M): x, y, z, w = R.from_matrix(M[:3, :3]).as_quat(); return [round(float(w), 6), round(float(x), 6), round(float(y), 6), round(float(z), 6)]
def _rot_deg(A, B): return float(np.degrees((R.from_matrix(A[:3, :3]) * R.from_matrix(B[:3, :3]).inv()).magnitude()))


def out(value=None, uncertainty=None, abstain=True, steps_used=0, note=''):
    return dict(value=value, uncertainty=uncertainty, abstain=bool(abstain), steps_used=int(steps_used), note=str(note))


def strip_frame(x, y, top_z, yaw):
    return _T([x, y, top_z - SOCKET_TOP], R.from_euler('z', yaw).as_matrix())


def seated_object_poses(Ts, hover_m):
    outp = []
    for k, ox in enumerate(OFFS):
        S = _T([ox, OUT_Y, SEAT_P], SEAT_R); seat = Ts @ S; hov = seat.copy(); hov[2, 3] += hover_m; outp.append((k, seat, hov))
    return outp


def G_from_inhand(position_in_tcp_frame_m, quaternion_in_tcp_frame_wxyz):
    q = quaternion_in_tcp_frame_wxyz; return _T(TCP + np.asarray(position_in_tcp_frame_m, float), R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix())


def align_target(strip, inhand, L_now, hover_m=0.015, sigma_loc_mm=0.5, max_pos_mm=2.0, max_axis_deg=2.0):
    """strip: {'x','y','top_z','yaw_rad'} from locate_mm (or None/abstained); inhand: value dict of inhand_pose
    {'position_in_tcp_frame_m','quaternion_in_tcp_frame_wxyz'} + 'uncertainty' {member_spread_mm, member_spread_deg, position_sigma_mm,
    rotation_sigma_deg}; L_now: current link6 4x4. Returns the L1 contract."""
    if not strip: return out(note='no outlet positions (locate_mm abstained or was not called): no target computed')
    if not inhand or not inhand.get('quaternion_in_tcp_frame_wxyz'):
        return out(note='no in-hand orientation (inhand_pose abstained or was not called): no target computed')
    u = inhand.get('uncertainty') or {}
    s_ip = float(u.get('position_sigma_mm') or u.get('member_spread_mm') or 0.6); s_rot = float(u.get('rotation_sigma_deg') or u.get('member_spread_deg') or 0.5)
    s_pos = math.sqrt(sigma_loc_mm ** 2 + s_ip ** 2 + (R_PRONG * 1000 * math.radians(s_rot)) ** 2)
    G = G_from_inhand(inhand['position_in_tcp_frame_m'], inhand['quaternion_in_tcp_frame_wxyz']); Gi = np.linalg.inv(G)
    Ts = strip_frame(strip['x'], strip['y'], strip['top_z'], strip['yaw_rad']); L_now = np.asarray(L_now, float); rows = []
    for k, seat, hov in seated_object_poses(Ts, hover_m):
        Lh = hov @ Gi; Ls = seat @ Gi
        rows.append(dict(outlet=k, hover_link6=dict(position=[round(float(v), 5) for v in Lh[:3, 3]], quaternion_wxyz=_q(Lh)),
                         seated_link6=dict(position=[round(float(v), 5) for v in Ls[:3, 3]], quaternion_wxyz=_q(Ls)),
                         charger_at_hover_world=dict(position=[round(float(v), 5) for v in hov[:3, 3]], quaternion_wxyz=_q(hov)),
                         wrist_turn_from_now_deg=round(_rot_deg(Lh, L_now), 1), hand_travel_m=round(float(np.linalg.norm(Lh[:3, 3] - L_now[:3, 3])), 3)))
    unc = dict(predicted_sigma_pos_mm=round(s_pos, 2), predicted_sigma_axis_deg=round(s_rot, 2), inputs=dict(sigma_loc_mm=sigma_loc_mm, sigma_inhand_pos_mm=s_ip, sigma_inhand_rot_deg=s_rot),
               model='outlet seat poses from 6 native successes; strip CAD; TCP offset; hover %.0f mm along world z' % (hover_m * 1000))
    ab = s_pos > max_pos_mm or s_rot > max_axis_deg
    note = ('predicted 1-sigma error at the charger %.1f mm / %.1f deg exceeds %.1f mm / %.1f deg: targets withheld' % (s_pos, s_rot, max_pos_mm, max_axis_deg) if ab else
            'gripper poses that put the charger on each outlet axis (hover %.0f mm above the seated depth, and seated); predicted 1-sigma %.1f mm / %.1f deg at the charger; 0 control steps' % (hover_m * 1000, s_pos, s_rot))
    return out(value=None if ab else dict(targets=rows, frame='world (environment_origin), metres; link6 = the end-effector frame of your actions'),
               uncertainty=unc, abstain=ab, note=note)
