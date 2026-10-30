"""vs_exec.py (C1 line, 2026-10-29): GENERIC keypoint relative visual-servo insertion executor.

Method (finepose README-KPNet-visual-servo.md, src/vs_run.py sha256 acc313ab801cce0e, copied ideas: ray / tri; no finepose file edited):
held-object keypoints AND target keypoints are measured in the SAME two idle-arm views, triangulated, and the hand is moved by the
relative error (held insertion points -> target entry points), so camera calibration errors largely cancel and no in-hand pose is needed.

Task-independent parts (this file): triangulation, the rigid correction that brings the held points onto the targets (tilt to the insertion
axis + turn about it + translation), the residual, and the step-budgeted loop
    one-shot first move -> <= 2 corrections -> one re-check -> guarded insert -> final action.
Task-specific parts come in through `spec` and an `io` object (see vs_plug.py for plug_in_charger):
  spec = dict(held_pts=[names], held_axis=([root names], [tip names]) or None, targets={tid: [names in the order of held_pts]}, pairings=[perm, ...],
              axis=(0, 0, -1) insertion direction (world), hover_m, depth_m, tol_mm, tol_deg, max_corr=2, final='release')
  io.observe(order) -> list of views [dict(tag, K, T (FK camera->world), det={name: (u, v, p)})]; io.free_at -> current park tag
  io.hand() -> link6 4x4; io.reach(L) -> bool (controller model); io.move(L, kind) -> dict(err_mm, err_deg); io.insert(L_hover, L_seat) -> dict
  io.steps() -> native steps used since the handover.
All model-facing text is English; no truth is read here (truth enters only through the io's keypoint source in truth mode)."""
import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def ray(K, T, uv):   # USD camera (looks along -Z, +Y up), as finepose vs_run.ray / combined_tools.project
    d = np.array([(uv[0] - K[0][2]) / K[0][0], -(uv[1] - K[1][2]) / K[1][1], -1.0]); T = np.asarray(T); d = T[:3, :3] @ d; return T[:3, 3], d / np.linalg.norm(d)


def tri(rays):
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, d in rays:
        M = np.eye(3) - np.outer(d, d); A += M; b += M @ o
    return np.linalg.solve(A, b)


def triangulate(views, names, p_min=0.5, max_resid_m=0.002):
    """-> {name: xyz} for points seen (p >= p_min) in >= 2 views whose rays agree within max_resid_m; resid {name: mm}."""
    P, resid = {}, {}
    for n in names:
        rs = [ray(v['K'], v['T'], v['det'][n][:2]) for v in views if n in v['det'] and v['det'][n][2] >= p_min]
        if len(rs) < 2: continue
        X = tri(rs); r = max(float(np.linalg.norm(np.cross(d, X - o))) for o, d in rs); resid[n] = round(r * 1000, 2)
        if r <= max_resid_m: P[n] = X
    return P, resid


def _unit(v): v = np.asarray(v, float); return v / max(np.linalg.norm(v), 1e-12)


def _rot_between(a, b):
    a, b = _unit(a), _unit(b); c = np.cross(a, b); s = np.linalg.norm(c); d = float(a @ b)
    if s < 1e-9: return np.eye(3) if d > 0 else R.from_rotvec(np.pi * _unit(np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0]))).as_matrix()
    return R.from_rotvec(c / s * math.atan2(s, d)).as_matrix()


def correction(Ph, Pt, spec, perm):
    """Rigid world motion C (4x4) that puts the held insertion points (spec['held_pts'], measured Ph) at their hover points above the
    target entries (Pt in the order of held_pts permuted by perm): tilt the held axis onto the insertion axis, turn about it so the point
    pattern matches, translate. Returns (C, residual dict) or (None, reason)."""
    ax = _unit(spec['axis']); hp = spec['held_pts']
    if not all(n in Ph for n in hp): return None, 'held points not triangulated: %s' % [n for n in hp if n not in Ph]
    H = np.array([Ph[n] for n in hp]); T_ = np.array([Pt[i] for i in perm]); hov = T_ - ax * spec['hover_m']
    Rt = np.eye(3); tilt = 0.0
    ha = spec.get('held_axis')   # (root names, tip names): the held object's insertion direction = centroid(tips) - centroid(roots)
    if ha and all(n in Ph for n in list(ha[0]) + list(ha[1])):
        a_h = np.mean([Ph[n] for n in ha[1]], 0) - np.mean([Ph[n] for n in ha[0]], 0); Rt = _rot_between(a_h, ax); tilt = math.degrees(math.acos(max(-1, min(1, float(_unit(a_h) @ ax)))))
    c = H.mean(0); H1 = (Rt @ (H - c).T).T
    yaw = 0.0
    if len(hp) >= 2:   # turn about the axis: least-squares angle between the in-plane point patterns
        def flat(X): X = X - X.mean(0); return X - np.outer(X @ ax, ax)
        A, B = flat(H1), flat(hov); num = sum(float(ax @ np.cross(a, b)) for a, b in zip(A, B)); den = sum(float(a @ b) for a, b in zip(A, B)); yaw = math.degrees(math.atan2(num, den))
    Ry = R.from_rotvec(ax * math.radians(yaw)).as_matrix(); Rc = Ry @ Rt
    C = np.eye(4); C[:3, :3] = Rc; C[:3, 3] = hov.mean(0) - Rc @ c
    d = hov.mean(0) - c; along = float(d @ ax)
    res = dict(dxy_mm=round(float(np.linalg.norm(d - along * ax)) * 1000, 2), dz_mm=round(-along * 1000, 2), dyaw_deg=round(yaw, 2), tilt_deg=round(tilt, 2),
               rot_deg=round(math.degrees(R.from_matrix(Rc).magnitude()), 2))
    return C, res


def converged(res, spec): return res['dxy_mm'] <= spec['tol_mm'] and abs(res['dyaw_deg']) <= spec['tol_deg'] and res['tilt_deg'] <= spec.get('tol_tilt_deg', 3.0) and abs(res['dz_mm']) <= spec.get('tol_dz_mm', 3.0)


def run(io, spec, log):
    """The executor loop. log (dict) receives every measurement; returns the outcome string."""
    names = list(spec['held_pts']) + (list(spec['held_axis'][0]) + list(spec['held_axis'][1]) if spec.get('held_axis') else []) + sorted({n for v in spec['targets'].values() for n in v})
    names = list(dict.fromkeys(names)); mem = {}

    def measure(tag):
        if getattr(io, 'measure3d', None) is not None:   # overfit round 1: keypoints in 3-D straight from the source (no views)
            P, rs, views = io.measure3d(), {}, []
        elif tag == 'first' and hasattr(io, 'first_measure'):   # task io may overlap the transport with the target views (step budget)
            P, rs, views = io.first_measure(names, spec)
        else:
            order = io.order(tag) if hasattr(io, 'order') else (('A', 'B') if io.free_at != 'B' else ('B', 'A'))
            views = io.observe(order); P, rs = triangulate(views, names)
        for n in [x for v in spec['targets'].values() for x in v]:   # targets do not move: remember them (the held object hides them late)
            if n in P: mem[n] = P[n]
            elif n in mem: P[n] = mem[n]
        row = dict(tag=tag, step=io.steps(), n_pts=len(P), resid_mm=rs, vis={v['tag']: sorted(k for k, d in v['det'].items() if d[2] >= 0.5) for v in views})
        if not views: row['source'] = '3d'
        if hasattr(io, 'diag'): row['diag'] = io.diag(P, CH[0])   # offline only: truth residual / held-object motion between the two views
        log.setdefault('obs', []).append(row); return P
    CH = [None]

    def best(P, only=None):
        cands = []
        for tid, tn in spec['targets'].items():
            if only is not None and tid != only[0]: continue
            if not all(n in P for n in tn): continue
            Pt = [P[n] for n in tn]
            for pi, perm in enumerate(spec['pairings']):
                if only is not None and pi != only[1]: continue
                C, res = correction(P, Pt, spec, perm)
                if C is None: continue
                L = C @ io.hand(); cands.append((tid, pi, C, L, res))
        return cands

    P = measure('first')
    if getattr(io, 'probe', False): return 'probe_views'   # view half: measurement coverage / 3-D error only, no motion after it
    cands = best(P)
    if not cands: return 'no_measurement'
    rows = []
    for tid, pi, C, L, res in cands:
        Ls = L.copy(); Ls[:3, 3] += np.asarray(spec['axis'], float) * (spec['hover_m'] + spec['depth_m']); ok = io.reach(L) and io.reach(Ls)
        rows.append((not ok, res['rot_deg'] + spec.get('pairing_cost', {}).get(pi, 0.0), tid, pi, L, res, ok))   # pairing_cost: prefer the native seated orientation
    rows.sort(key=lambda r: (r[0], r[1])); log['choices'] = [dict(target=r[2], pairing=r[3], reach=r[6], rot_deg=r[5]['rot_deg']) for r in rows]
    if not rows[0][6]: return 'no_reachable_target'
    _, _, tid, pi, L, res, _ = rows[0]; CH[0] = (tid, pi); log['obs'][-1]['diag'] = io.diag(P, CH[0]) if hasattr(io, 'diag') else None; log['target'] = dict(id=tid, pairing=pi, first_residual=res)
    log['first_move'] = io.move(L, 'first'); n_corr = 0
    while True:
        P = measure('check%d' % n_corr); c = best(P, (tid, pi))
        if not c: log['last'] = 'lost measurement'; return 'lost_measurement'
        res = c[0][4]; log.setdefault('residuals', []).append(res)
        if converged(res, spec): break
        if n_corr >= spec.get('max_corr', 2): break
        log.setdefault('corrections', []).append(io.move(c[0][3], 'corr%d' % n_corr)); n_corr += 1
    log['n_corr'] = n_corr; log['pre_insert_residual'] = res
    if res['dxy_mm'] > spec.get('abort_mm', 2.0) or abs(res['dyaw_deg']) > spec.get('abort_deg', 6.0): return 'not_aligned'
    Lh = io.hand(); Ls = Lh.copy(); Ls[:3, 3] += np.asarray(spec['axis'], float) * (spec['hover_m'] + spec['depth_m'])
    log['insert'] = io.insert(Lh, Ls)
    return 'inserted' if log['insert'].get('inserted') else 'insert_failed'
