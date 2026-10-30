"""finepose step 3 (user 10-29): relative visual servoing with KPNet on insert_key (closed loop, sim). Deploy inputs only for decisions:
RGB (left-wrist side views), camera intrinsics/extrinsics (rpc camera), link6 poses (proprio), KPNet (kp8-lw.pt) and the CAD keypoint layout.
Truth is only logged (analysis) and used for the dev start state (official grasp pinned at the hover + C-style perturbation, exp.perturb).
Loop: observe (left arm parks at A az180 / B az235, 0.15 m, aim = right TCP + 3 cm along the gripper) -> KPNet -> triangulate key tip, teeth,
blade root, keyhole centre, slot end from the SAME two images -> relative error (dxy tip->hole, dz, dyaw key teeth dir -> slot dir) ->
move the right hand (yaw about the vertical through the tip, then translate) -> repeat until |dxy| < tol and |dyaw| < 1 deg -> guarded straight
descent (Env.ref_descend) -> depth check from the side views (blade root) -> clockwise turn (Env.ref_turn).
  v3 (10-29, step compression): full first correction (no 8 mm clamp), alternate park order, proprio seating check before any extra view; per-phase step log.
  python vs_run.py --seed N --gpu G --out DIR [--ckpt kp/models/kp8-lw.pt]"""
import argparse, json, math, sys, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'kp' / 'h200'))
import fp_lib as F  # noqa: E402
from fp_lib import CT, R  # noqa: E402
from exp import perturb  # noqa: E402
import kpnet3  # noqa: E402
CAM_IN_L6 = np.array([0.085, 0.0, 0.051]); CAM_VIEW_L6 = np.array([0.87, 0.0, -0.5]); TIP_TO_ROOT = 0.0428; HOVER = 0.010  # tip 10 mm above the lock top before descending
TIP, T1, T2, ROOT, BOW, HOLE, SLOT, RIM = range(8)


def park(env, az, dist=0.15):
    L6r = env.T_link6(); aim = L6r[:3, :3] @ (F.TCP + np.array([0.03, 0, 0])) + L6r[:3, 3]
    h = np.array([math.cos(math.radians(az)), math.sin(math.radians(az)), 0.0]); xv = -h; zv = np.array([0, 0, 1.0]); yv = np.cross(zv, xv)
    L = np.eye(4); L[:3, :3] = np.column_stack([xv, yv, zv]); view = L[:3, :3] @ (CAM_VIEW_L6 / np.linalg.norm(CAM_VIEW_L6)); L[:3, 3] = aim - dist * view - L[:3, :3] @ CAM_IN_L6
    env.tool.line('left', L[:3, 3], CT.q_of(L), 200, 0.003, 'vs_park', 'vs'); A = CT.T_of(*env.tool.eef('left')); return float(np.linalg.norm(A[:3, 3] - L[:3, 3]))


def ray(Kin, T, uv):
    d = np.array([(uv[0] - Kin[0][2]) / Kin[0][0], -(uv[1] - Kin[1][2]) / Kin[1][1], -1.0]); T = np.asarray(T); d = T[:3, :3] @ d; return T[:3, 3], d / np.linalg.norm(d)


def tri(rays):
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, d in rays: M = np.eye(3) - np.outer(d, d); A += M; b += M @ o
    return np.linalg.solve(A, b)


def observe(env, det, tag, order=(180.0, 235.0)):
    views = []; Lhold = env.T_link6()
    for az in order:  # v3: caller alternates A,B / B,A (no return trip)
        e = park(env, az); env.tool.act('right', Lhold[:3, 3], CT.q_of(Lhold), env.close, 'vs_hold', steps=3)
        c = env.rpc('camera')['cam_left_wrist']; im = next(i for i in env.ad.packet['images'] if i['camera'] == 'cam_left_wrist')
        img = np.asarray(Image.open(im['path']).convert('RGB')); Image.fromarray(img).save(env.out / 'img' / ('%s-az%d.png' % (tag, az)))
        views.append(dict(K=c['intrinsics'], T=c['extrinsics_cam_to_world'], det=det(img), park_err_mm=round(e * 1000, 1)))
    P = {}; resid = {}
    for k in (TIP, T1, T2, ROOT, HOLE, SLOT):
        rs = [ray(v['K'], v['T'], v['det'][k][:2]) for v in views if v['det'][k][2] >= 0.5]
        if len(rs) == 2:
            X = tri(rs); r = max(float(np.linalg.norm(np.cross(d, X - o))) for o, d in rs); resid[k] = round(r * 1000, 2)
            if r < 0.002: P[k] = X  # v4: drop points whose two rays miss each other by > 2 mm (inconsistent detections)
    return P, [dict(park_err_mm=v['park_err_mm'], vis=[round(d[2], 2) for d in v['det']]) for v in views] + [dict(ray_residual_mm=resid)]


LOCK_MEM = {}  # v2: keyhole / slot points remembered from the last view in which they were visible (the lock does not move; the key hides them once inserted)


def relative(P):
    """relative error from the triangulated points (all from the same two images; keyhole from memory when hidden by the inserted key)."""
    for k in (HOLE, SLOT):
        if k in P: LOCK_MEM[k] = P[k]
        elif k in LOCK_MEM: P[k] = LOCK_MEM[k]
    if not all(k in P for k in (HOLE, SLOT)) or not (TIP in P or ROOT in P): return None
    tip = P[TIP] if TIP in P else None; out = dict(hole=P[HOLE].round(4).tolist())
    if tip is not None: out.update(dx_mm=(P[HOLE][0] - tip[0]) * 1000, dy_mm=(P[HOLE][1] - tip[1]) * 1000, dz_mm=(tip[2] - P[HOLE][2]) * 1000)
    if ROOT in P and P[ROOT][2] < P[HOLE][2] + 0.06: out['depth_from_root_mm'] = (P[HOLE][2] - (P[ROOT][2] - TIP_TO_ROOT)) * 1000  # tip hidden inside the lock
    sd = P[SLOT] - P[HOLE]; slot_yaw = math.degrees(math.atan2(sd[1], sd[0]))
    if (T1 in P or T2 in P) and (TIP in P and ROOT in P):
        tm = np.mean([P[k] for k in (T1, T2) if k in P], 0); ax = P[ROOT] - P[TIP]; ax /= np.linalg.norm(ax); w = tm - P[TIP]; w -= ax * (w @ ax)
        out['dyaw_deg'] = (slot_yaw - math.degrees(math.atan2(w[1], w[0])) + 180) % 360 - 180; out['tilt_deg'] = math.degrees(math.acos(min(1.0, abs(ax[2]))))
    return out


def servo_move(env, rel, gain=1.0, max_mm=25.0, max_deg=30.0, max_dz=60.0):
    """yaw about the vertical through the key tip (estimated), then translate the hand; link6 target via move_to."""
    L = env.T_link6(); dyaw = max(-max_deg, min(max_deg, gain * rel.get('dyaw_deg', 0.0))); tip = np.array(rel['tip']) if 'tip' in rel else L[:3, 3]
    Rz = R.from_euler('z', dyaw, degrees=True).as_matrix(); L2 = np.eye(4); L2[:3, :3] = Rz @ L[:3, :3]; L2[:3, 3] = tip + Rz @ (L[:3, 3] - tip)
    d = np.array([rel['dx_mm'], rel['dy_mm']]) * gain; n = np.linalg.norm(d); d = d * min(1.0, max_mm / max(n, 1e-9)); L2[:2, 3] += d / 1000
    if 'dz_mm' in rel: L2[2, 3] += max(-max_dz, min(max_dz, HOVER * 1000 - rel['dz_mm'])) / 1000  # v3: full height correction in one move
    return env.move_to(L2, 'vs_move', 150, 0.0003)  # v4: fine tolerance so sub-mm corrections are executed


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--gpu', type=int, required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', default=str(Path(__file__).resolve().parent.parent / 'kp' / 'models' / 'kp8-lw.pt')); ap.add_argument('--iters', type=int, default=6); ap.add_argument('--tol-mm', type=float, default=0.6)  # v4: v3's 0.3 mm was below the move_to tolerance (1 mm) -> idle iterations
    a = ap.parse_args(); res = dict(args=vars(a), iters=[]); env = None; t0 = time.time()
    try:
        det = kpnet3.Detector(a.ckpt, 0); env = F.Env(a.out, a.gpu)
        if not env.setup_held(): res['failure'] = 'setup'; raise StopIteration
        res['setup_steps'] = int(env.ad.packet['step_id']); res['perturbation'] = perturb(env, a.seed); env.log('perturbed'); s0 = int(env.ad.packet['step_id']); res['perturb_steps'] = s0 - res['setup_steps']
        for it in range(a.iters):
            st0 = int(env.ad.packet['step_id']); P, meta = observe(env, det, 'it%02d' % it, (180.0, 235.0) if it % 2 == 0 else (235.0, 180.0)); rel = relative(P); tr = env.truth(); res.setdefault('phase_steps', []).append(('observe%d' % it, int(env.ad.packet['step_id']) - st0))
            if TIP in P: (rel or {}).update(tip=P[TIP].tolist())
            rec = dict(it=it, step=int(env.ad.packet['step_id']), est=None if rel is None else {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rel.items()}, views=meta,
                       truth=dict(key_vs_slot_xy_mm=tr['key_vs_slot_xy_mm'], yaw360=tr.get('key_yaw_vs_keyway_deg360'), tilt=tr['key_tilt_deg'], depth=tr['insertion_depth_mm']))
            res['iters'].append(rec); (Path(a.out) / 'progress.json').write_text(json.dumps(res, indent=1, default=str))
            if rel is None or 'dx_mm' not in rel: res['failure'] = 'perception (keypoints not seen in both views)'; raise StopIteration
            if math.hypot(rel['dx_mm'], rel['dy_mm']) < a.tol_mm and abs(rel.get('dyaw_deg', 0)) < 1.0 and abs(rel['dz_mm'] - HOVER * 1000) < 3: break
            st0 = int(env.ad.packet['step_id']); servo_move(env, rel); res['phase_steps'].append(('move%d' % it, int(env.ad.packet['step_id']) - st0))
        res['aligned_truth'] = env.log('vs_aligned'); env.settle()
        for t in range(3):  # guarded descent + depth check from the side views (tip hidden: blade root)
            z0 = env.T_link6()[2, 3]; st0 = int(env.ad.packet['step_id']); env.ref_descend(); res['phase_steps'].append(('descend%d' % t, int(env.ad.packet['step_id']) - st0)); drop = (z0 - env.T_link6()[2, 3]) * 1000
            if drop >= 38: res.setdefault('depth_checks', []).append(dict(try_=t, est_depth_mm=None, how='proprio drop %.1f mm' % drop, truth_depth_mm=env.truth()['insertion_depth_mm'])); break  # v3: proprio first, no extra observation
            P, meta = observe(env, det, 'depth%d' % t); rel = relative(P) or {}
            dep = rel.get('depth_from_root_mm'); res.setdefault('depth_checks', []).append(dict(try_=t, est_depth_mm=None if dep is None else round(dep, 2), truth_depth_mm=env.truth()['insertion_depth_mm'],
                                                                                                 drop_mm=round((z0 - env.T_link6()[2, 3]) * 1000, 1)))
            if (dep is not None and dep >= 27) or (dep is None and (z0 - env.T_link6()[2, 3]) * 1000 >= 38): break  # v2: proprio fallback = ref_insert seated rule
            for _ in range(4): env.move(dz_mm=5, tag='vs_retry_lift', src='vs')
            P, meta = observe(env, det, 'retry%d' % t); rel = relative(P)
            if rel and 'dx_mm' in rel: servo_move(env, rel)
        st0 = int(env.ad.packet['step_id']); env.ref_turn(); res.setdefault('phase_steps', []).append(('turn', int(env.ad.packet['step_id']) - st0)); res['action_steps'] = int(env.ad.packet['step_id']) - s0
    except StopIteration:
        pass
    except Exception as e:  # noqa: BLE001
        res['error'] = '%s: %s' % (type(e).__name__, str(e)[:300]); res['trace'] = traceback.format_exc()[-1500:]
    finally:
        if env is not None:
            try:
                c = env.checks(); res['checks'] = c; res['native_success'] = bool((env.ad.native_end or {}).get('success')) or c.get('all', 0) >= 1.0
            except Exception as e:  # noqa: BLE001
                res['check_error'] = str(e)[:200]; res['native_success'] = bool((env.ad.native_end or {}).get('success'))
            res['final_truth'] = env.events[-1] if env.events else None; res['steps_total'] = int(env.ad.packet['step_id'])
            res['max_depth_mm'] = max((e.get('insertion_depth_mm', -99) for e in env.events), default=None); env.close_sim()
        res['seconds'] = round(time.time() - t0); Path(a.out, 'result.json').write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps(dict(out=Path(a.out).name, success=res.get('native_success'), iters=len(res['iters']), failure=res.get('failure'), error=res.get('error'),
                              max_depth=res.get('max_depth_mm'), steps=res.get('steps_total')), default=str), flush=True)


if __name__ == '__main__':
    main()
