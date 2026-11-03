"""grasp_compatible v1.1 = v1 + (rung-2 iteration 2, 17:4xZ): gripper state from proprioception AND the last commanded opening (action history):
"busy" = commanded closed but stopped open (holds an object) -> that arm is not used; two path options per candidate (direct, or rise
to a safe height first), the executor follows exactly the evaluated path; collision only evaluated for controller-reachable paths;
executor returns held yes/no/unknown + source (finger-joint check only when the closing extent >= 5 mm; else unknown). v1 docstring:
grasp_compatible v1 (harness v2 L1; capability line, 2026-10-28 16:xxZ). General two-finger grasp candidates from the object CAD model.

    out = grasp_compatible(obj_vertices, T_obj, q_now, arm_state, support_top_z=None, next_stage=None, mode='info', ...)
    out == {"value", "uncertainty", "abstain", "steps_used", "note"}

Deploy inputs: the object's CAD bounding box (8 local vertices; class name + CAD are deploy-time priors, user rule 10-28), the object
pose (from a locate tool, e.g. locate_mm('mahjong_tile'); simulator truth only in rung-2 tests), measured joints and gripper joints,
robot constants (URDF, arm roots, table plane = arm mount height). No other object poses.

Candidate generation (no task rules):
  * OBB axes and extents from the CAD vertices (object frame). A closing axis k is allowed if extent_k + 2*FINGER_MARGIN <= MAX_OPEN.
  * Approach directions = +-OBB axes perpendicular to k and their 45-deg blends, in the world; never from below (approach z > 0.3 rejected).
  * Depth: the TCP goes as deep as the object allows, but the fingertips stay CLEARANCE (1.5 mm, rsiws-ee thin-standing G1 finding)
    above the support top (= the object's lowest point, or the given support_top_z) and above the table + 3 mm.
  * Offsets along the remaining axis: 0, +-25 % of its half extent. Pre-grasp opening = extent_k + 2*FINGER_MARGIN (rung-2 lesson mku/mkv:
    a fully open gripper hit the neighbours, 0/26 lifted; opened to the object width: 23/24 lifted; margin 2 mm per side).
  * Family label: 'thin_standing' when k is the thinnest axis, that axis is horizontal and the approach is top-down (rsiws-ee family);
    'top_pinch' for other top-down approaches; 'side_pinch' for horizontal approaches; 'oblique' otherwise.
Filters (facts, 0 control steps):
  * reachable: native-controller model from the measured joints through pre-grasp (6 cm back along the approach) -> grasp -> lift (+6 cm);
  * collision_free + margin_mm: arm/gripper collision spheres (cuRobo sphere model, fingers at the pre-grasp width) vs the table and a
    support column box (object footprint + SUPPORT_PAD, from the table up to the support top), along the controller trace to the pre-grasp
    and the lift; the object itself is ignored for the last 6 cm (the fingers straddle it);
  * next_stage (optional): list of object target poses (4x4); reachable from the lift joints with the grasp held (controller model).
Ranking: reachable & collision_free, then next_stage reachable, then margin, then distance travelled.
Executor (mode='execute', only on an explicit call; step cap): open to the pre-grasp width, rise to a safe height if below it, line to the
pre-grasp, line in, close, lift; returns status 'held' | 'not_held' | 'unreachable' | 'step_cap' from deploy signals only (measured gripper
width after closing vs the commanded empty width).
"""
import itertools, math
import numpy as np
from scipy.spatial.transform import Rotation as R

TCP = np.array([0.14, 0.0, 0.0]); TIP = 0.0176           # link6 -> TCP; TCP -> fingertip along the approach (X5 gripper)
MAX_OPEN = 0.088; FINGER_MARGIN = 0.002; CLEARANCE = 0.0015; TABLE_CLEAR = 0.003; SUPPORT_PAD = 0.03; PRE_BACK = 0.06; LIFT = 0.06


def _T(p, Rm):
    M = np.eye(4); M[:3, :3] = Rm; M[:3, 3] = p; return M


def _q(M):
    x, y, z, w = R.from_matrix(M[:3, :3]).as_quat(); return [float(w), float(x), float(y), float(z)]


def obb(vertices):
    V = np.asarray(vertices, float); lo, hi = V.min(0), V.max(0); return (lo + hi) / 2, hi - lo


def candidates(vertices, T_obj, support_top_z=None, table_z=0.765):
    """All CAD-derived grasp candidates: list of dicts with link6 grasp pose L (4x4 world), approach, closing axis, pre-open width, family."""
    c_loc, ext = obb(vertices); Ro = T_obj[:3, :3]; c_w = T_obj[:3, :3] @ c_loc + T_obj[:3, 3]
    Vw = (Ro @ np.asarray(vertices, float).T).T + T_obj[:3, 3]; obj_bottom = float(Vw[:, 2].min())
    sup = obj_bottom if support_top_z is None else float(support_top_z); floor = max(sup + CLEARANCE, table_z + TABLE_CLEAR)
    thin = int(np.argmin(ext)); out = []
    for k in range(3):
        if ext[k] + 2 * FINGER_MARGIN > MAX_OPEN: continue
        others = [j for j in range(3) if j != k]
        for sgn in (1.0, -1.0):
          cw = sgn * Ro[:, k]
          dirs = []
          for j in others:
              for s in (1, -1): dirs.append(s * Ro[:, j])
          for s1, s2 in itertools.product((1, -1), (1, -1)):
              dirs.append((s1 * Ro[:, others[0]] + s2 * Ro[:, others[1]]) / math.sqrt(2))
          for a in dirs:
              if a[2] > 0.3: continue                                       # never from below
              m = [j for j in others if abs(Ro[:, j] @ a) < 0.5]            # remaining (offset) axis: the other axis not along a
              m_ax = Ro[:, m[0]] if m else np.zeros(3); half_m = ext[m[0]] / 2 if m else 0.0
              for off in (0.0, 0.25, -0.25):
                  tcp = c_w + m_ax * off * half_m
                  # depth: move the TCP from the object centre along a until the fingertips reach the floor (support top + 1.5 mm), at most to
                  # the object's far face; horizontal approaches keep the TCP at the centre (pads straddle the middle)
                  half_a = 0.5 * float(np.abs(ext) @ np.abs(Ro.T @ a))
                  s_deep = min(half_a, (tcp[2] - floor) / -a[2] - TIP) if a[2] < -1e-6 else 0.0
                  for depth_mode, s_ in (('deep', s_deep), ('centred', 0.0)):   # centred: pinch line through the CAD centre (no gravity torque about the pinch axis)
                    if depth_mode == 'centred' and abs(s_deep) < 1e-6: continue
                    tcp_g = tcp + a * s_
                    if a[2] >= -1e-6 and tcp_g[2] - 0.010 < floor: continue   # horizontal fingers (about 2 x 10 mm thick) would touch the support
                    y = cw - a * (a @ cw); y /= np.linalg.norm(y); Rl = np.column_stack([a, y, np.cross(a, y)])
                    L = _T(tcp_g - Rl @ TCP, Rl)
                    fam = ('thin_standing' if (k == thin and abs(cw[2]) < 0.3 and a[2] < -0.9) else 'top_pinch' if a[2] < -0.9 else
                           'side_pinch' if abs(a[2]) < 0.3 else 'oblique')
                    out.append(dict(L=L, approach=a.round(4).tolist(), closing_axis=cw.round(4).tolist(), closing_extent_m=float(ext[k]),
                                    pre_open_width_m=float(min(MAX_OPEN, ext[k] + 2 * FINGER_MARGIN)), offset_frac=off, family=fam, depth=depth_mode))
    ded = {}
    for c in out: ded.setdefault((tuple(np.round(c['L'][:3, 3], 4)), tuple(np.round(c['L'][:3, :3].ravel(), 3))), c)
    return list(ded.values()), dict(center=c_w, ext=ext, bottom=obj_bottom, support_top=sup, footprint=Vw)


def _support_box(geo, table_z):
    V = geo['footprint']; lo = V[:, :2].min(0) - SUPPORT_PAD; hi = V[:, :2].max(0) + SUPPORT_PAD; ztop = geo['support_top']
    if ztop - table_z < 0.002: return None
    T = np.eye(4); T[:3, 3] = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (table_z + ztop) / 2]
    return ('support', (T, np.array([(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, (ztop - table_z) / 2])))


def ticks_to(rm, arm, q0, wps, cap=400, tol=(0.003, math.radians(3))):
    """Controller-model ticks (= control steps at one target per step) to converge through the waypoints; None if a waypoint is not reached."""
    q = np.asarray(q0, float); total = 0
    for M in wps:
        goal = dict(position=M[:3, 3], quaternion_wxyz=_q(M)); n = None
        for t in range(cap):
            q = np.asarray(rm.fk[arm].bounded_step(q, rm.lim[arm], rm.root[arm], goal), float)
            P = rm.pose(arm, q)
            if np.linalg.norm(P[:3, 3] - M[:3, 3]) < tol[0] and (R.from_matrix(P[:3, :3]) * R.from_matrix(M[:3, :3]).inv()).magnitude() < tol[1]:
                n = t + 1; break
        if n is None: return None
        total += n
    return total


def _paths(rm, arm, q_now, L, a):
    pre = L.copy(); pre[:3, 3] -= a * PRE_BACK; lift = L.copy(); lift[2, 3] += LIFT
    cur = rm.pose(arm, q_now); rise = cur.copy(); rise[2, 3] = max(cur[2, 3], pre[2, 3] + 0.05)
    out = [('direct', [pre, L, lift])]
    if rise[2, 3] - cur[2, 3] > 0.01: out.append(('rise', [rise, pre, L, lift]))
    return out


def evaluate(cr, rm, arm, q_now, cands, geo, table_z, next_stage=None, top=5, other=None):
    """Facts per candidate; for each, the first path option (direct, then rise-first) that is controller-reachable and collision-free."""
    box = _support_box(geo, table_z); boxes = [box] if box else []; rows = []
    OS = None
    if cr is not None and other is not None:   # (arm_name, q_other): the other arm's collision spheres at its measured joints (proprioception)
        S = cr.spheres_world(other[0], np.asarray(other[1], float)[None])[0]; OS = S[S[:, 3] > 0]
    def arm_gap(tr):
        if OS is None: return np.inf
        S = cr.spheres_world(arm, tr); v = S[..., 3] > 0
        d = np.linalg.norm(S[:, :, None, :3] - OS[None, None, :, :3], axis=-1) - S[:, :, None, 3] - OS[None, None, :, 3]
        d = np.where(v[:, :, None], d, np.inf); return float(d.min())
    for c in cands:
        L = c['L']; a = np.asarray(c['approach']); best = None
        for name, wps in _paths(rm, arm, q_now, L, a):
            p, q_end = rm.path(arm, q_now, wps); reach = all(x[2] for x in p)
            row = dict(c, path=name, waypoints=wps, reachable=bool(reach), residual_mm=[x[0] for x in p], collision_free=None, margin_mm=None, q_end=q_end)
            if reach: row['model_ticks'] = ticks_to(rm, arm, q_now, wps[:-1])
            if reach and cr is not None:
                q = np.asarray(q_now, float); dmin = np.inf; who = None
                for k, W in enumerate(wps):
                    tr, _, _, _ = cr.controller_trace(arm, q, W); q = tr[-1]
                    ga = arm_gap(tr[::5])
                    if ga < dmin: dmin = ga; who = 'other_arm'
                    if W is L or k == len(wps) - 1: continue        # the last 6 cm in (fingers straddle the object) and the lift are not checked against the support
                    d, w = cr.clearance(arm, tr, boxes)
                    if d.min() < dmin: dmin = float(d.min()); who = str(w[int(np.argmin(d))])
                trl, _, _, _ = cr.controller_trace(arm, q, wps[-1]); dl, _ = cr.clearance(arm, trl[len(trl) // 2:], [])
                m = min(dmin, float(dl.min())); row.update(collision_free=bool(m >= 0), margin_mm=round(m * 1000, 1), first_obstacle=who if m < 0 else None)
            if best is None or (row['reachable'] and row['collision_free'] is not False and not (best['reachable'] and best['collision_free'] is not False)):
                best = row
            if row['reachable'] and row['collision_free'] is not False: break
        row = best
        if next_stage is not None and row['reachable']:
            ok = []; turns = []
            for Tt in next_stage['targets']:
                Lt = Tt @ next_stage['T_obj_inv'] @ L; H = Lt.copy(); H[2, 3] += 0.06
                pp, _ = rm.path(arm, row['q_end'], [H, Lt]); good = all(x[2] for x in pp); ok.append(good)
                if good: turns.append(float(np.degrees((R.from_matrix(Lt[:3, :3]) * R.from_matrix(L[:3, :3]).inv()).magnitude())))
            row['next_stage_reachable'] = bool(any(ok)); row['next_stage_ok_targets'] = int(sum(ok))
            row['next_stage_turn_deg'] = round(min(turns), 1) if turns else None   # A1: wrist turn grasp -> seat (a pinched object swings in large turns)
        rows.append(row)
    key = lambda r: (not r['reachable'], r.get('collision_free') is not True, not r.get('next_stage_reachable', True), r.get('depth') != 'centred', r.get('model_ticks') or 9999, -(r.get('margin_mm') or 0.0))
    rows.sort(key=key); return rows


def facts(rows, top=5):
    out = []
    for r in rows[:top]:
        L = r['L']; out.append(dict(family=r['family'], grasp_link6=dict(position=L[:3, 3].round(4).tolist(), quaternion_wxyz=[round(v, 5) for v in _q(L)]),
                                    approach=r['approach'], closing_axis=r['closing_axis'], pre_open_width_mm=round(r['pre_open_width_m'] * 1000, 1),
                                    reachable=r['reachable'], collision_free=r.get('collision_free'), margin_mm=r.get('margin_mm'),
                                    next_stage_reachable=r.get('next_stage_reachable'), model_steps_to_grasp=r.get('model_ticks')))
    return out


def gripper_state(gripper_joint, last_command):
    """'busy' (holds an object): commanded (nearly) closed but the fingers stopped open; else 'empty'. last_command from the action history."""
    if last_command is None:
        return 'busy' if 0.05 < gripper_joint < 0.9 else 'empty'
    return 'busy' if (last_command < 0.5 and gripper_joint - last_command > 0.05) else 'empty'


def grasp_compatible(arm, q_now, gripper_joint, vertices, T_obj, rm, cr_factory=None, support_top_z=None, table_z=0.765, next_stage=None, top=5, last_command=None, other=None):
    """mode='info': L1 return with ranked candidate facts. cr_factory(width) -> reach_cu.CollisionReach (or None: controller-only)."""
    try:
        if gripper_state(gripper_joint, last_command) == 'busy':
            return dict(value=None, uncertainty=None, abstain=True, steps_used=0, busy=True,
                        note='the %s gripper holds an object (commanded %.2f, stopped at %.2f); it is not used' % (arm, last_command if last_command is not None else -1, gripper_joint))
        cands, geo = candidates(vertices, T_obj, support_top_z, table_z)
        if not cands:
            return dict(value=None, uncertainty=None, abstain=True, steps_used=0, note='no closing axis of the object fits the gripper (max opening 88 mm)')
        by_w = {}
        for c in cands: by_w.setdefault(round(c['pre_open_width_m'], 3), []).append(c)
        rows = []
        for w, cs in by_w.items():
            rows += evaluate(cr_factory(w) if cr_factory else None, rm, arm, np.asarray(q_now, float), cs, geo, table_z, next_stage, other=other)
        rows.sort(key=lambda r: (not r['reachable'], r.get('collision_free') is not True, not r.get('next_stage_reachable', True), r.get('model_ticks') or 9999, -(r.get('margin_mm') or 0.0)))
        n_ok = sum(r['reachable'] and r.get('collision_free') is not False for r in rows) if cr_factory is None else sum(r['reachable'] and r.get('collision_free') is True for r in rows)
        return dict(value=dict(candidates=facts(rows, top), n_candidates=len(rows), n_reachable_collision_free=n_ok, _rows=rows),
                    uncertainty=dict(model='native-controller trace + sphere collision (table, support column); object pose as given'),
                    abstain=n_ok == 0, steps_used=0,
                    note=('%d of %d CAD grasp candidates are reachable and collision-free for the %s arm' % (n_ok, len(rows), arm)))
    except Exception as e:  # noqa: BLE001
        return dict(value=None, uncertainty=None, abstain=True, steps_used=0, note='internal error %s: %s' % (type(e).__name__, str(e)[:200]))


def goto(tool, arm, M, cap, tol, opening, q_of=_q):
    """Command the final target directly (as the controller model does: the native EEF controller bounds the motion itself) until within
    tol / 3 deg or cap steps. Interpolated Cartesian lines stalled 11 mm / 12 deg short near the wrist singularity in v1.7b states."""
    s0 = tool.steps; pos = M[:3, 3]; quat = q_of(M); Rt = R.from_matrix(M[:3, :3])
    while tool.steps - s0 < cap and not tool.terminal():
        p, q = tool.eef(arm); pe = float(np.linalg.norm(np.asarray(p) - pos)); re = float((R.from_quat([q[1], q[2], q[3], q[0]]) * Rt.inv()).magnitude())
        if pe <= tol and re <= math.radians(3): break
        tool.act(arm, pos.tolist(), quat, opening, 'grasp_compatible goto', steps=5)
    p, q = tool.eef(arm); pe = float(np.linalg.norm(np.asarray(p) - pos)); re = float((R.from_quat([q[1], q[2], q[3], q[0]]) * Rt.inv()).magnitude())
    return dict(measured=dict(position_error_m=pe, rotation_error_rad=re, steps_used=tool.steps - s0))


def execute(tool, arm, row, ST, grip_joint, step_cap=150, q_of=_q):
    """Executor (explicit call only): follows the evaluated waypoints exactly. Deploy signals only."""
    s0 = tool.steps; wps = row['waypoints']; L = row['L']
    def left(): return step_cap - (tool.steps - s0)
    tool.set_gripper(arm, opening=ST.opening_for_width(row['pre_open_width_m']), reason='gc_open')
    for k, W in enumerate(wps[:-2]):                      # rise (optional) and pre-grasp
        if left() <= 0: return dict(status='step_cap', held='unknown', held_source='none', steps_used=tool.steps - s0)
        r = tool.line(arm, W[:3, 3], q_of(W), min(120, left()), 0.004 if k == len(wps) - 3 else 0.01, 'gc_pre%d' % k, 'grasp_compatible')
    if r['measured']['position_error_m'] > 0.015:
        return dict(status='unreachable', held='no', held_source='not grasped', steps_used=tool.steps - s0, pre_err_mm=round(r['measured']['position_error_m'] * 1000, 1))
    if left() <= 0: return dict(status='step_cap', held='unknown', held_source='none', steps_used=tool.steps - s0)
    r2 = tool.line(arm, L[:3, 3], q_of(L), min(40, left()), 0.003, 'gc_in', 'grasp_compatible')
    tool.set_gripper(arm, opening=0.0, reason='gc_close'); w = grip_joint()
    p, q = tool.eef(arm)
    if left() > 0: tool.line(arm, [p[0], p[1], p[2] + LIFT + 0.01], q, min(30, left()), 0.006, 'gc_lift', 'grasp_compatible')
    w2 = grip_joint()
    if row['closing_extent_m'] >= 0.005:
        held = 'yes' if w2 > 0.05 else 'no'; src = 'gripper_joint'      # an empty closed gripper reads ~0
    else:
        held = 'unknown'; src = 'none: the finger reading cannot tell an object thinner than 5 mm from an empty grasp; visual check not built'
    return dict(status='done', held=held, held_source=src, steps_used=tool.steps - s0, grip_joint_after_close=w, grip_joint_after_lift=w2,
                in_err_mm=round(r2['measured']['position_error_m'] * 1000, 1), path=row['path'])
