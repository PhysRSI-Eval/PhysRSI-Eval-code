"""finepose T5 (SAM3 + KPNet, use.md recipe) training/test view generator for the HELD key (10-29). One simulator episode per process:
setup_handover(seed) -> smooth reorient tip-down above the lock (target_hand, truth: TRAINING data only) -> for each of --npert in-hand perturbations
(re-pin the key in the closed fingers: axis shift, rotation about the pinch axis, tilt; some states lowered into the lock 5-30 mm with truth
alignment) -> --nview LEFT-wrist side views (free arm, camera ~0.12-0.18 m from the nominal key centre, azimuth 140-260 deg, aim noise +-1.5 cm)
+ 1 RIGHT-wrist view. Labels: 5 keypoints of the key (object frame below) projected with the sim camera RPC (USD convention).
Record format = use.md 4.1 (images/<id>-<cam>.jpg + L<layout>-s<seed>.jsonl), layout = state id.   python kp_gen.py --seed N --gpu G --out kp/data-v1"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fp_lib as F  # noqa: E402
from fp_lib import CT, R  # noqa: E402
KP = np.array([[-0.0021, 0, -0.0314], [0.0106, 0, -0.0258], [0.0106, 0, -0.0150], [-0.0021, 0, 0.0114], [-0.0021, 0, 0.0214]])  # bbox-centre frame (visual mesh)
KP_NAMES = ['tip', 'tooth1_outer', 'tooth2_outer', 'blade_root', 'bow_centre']
CAM_IN_L6 = np.array([0.085, 0.0, 0.051]); CAM_VIEW_L6 = np.array([0.87, 0.0, -0.5])


def project(K, T, P):
    pc = (np.linalg.inv(np.asarray(T)) @ np.c_[P, np.ones(len(P))].T).T[:, :3]; K = np.asarray(K)
    return [None if z >= 0 else [float(K[0, 0] * x / -z + K[0, 2]), float(-K[1, 1] * y / -z + K[1, 2])] for x, y, z in pc]


def pin(env, Gp):
    """hold the key at the in-hand pose Gp (key in link6) with the fingers closed (re-pin; same pose every time -> no ghost)."""
    K2 = env.T_link6() @ Gp; Ro = K2[:3, :3]; po = K2[:3, 3] - Ro @ env.cl; q = R.from_matrix(Ro).as_quat(); qw = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]; p_, q_ = env.tool.eef('right')
    for _ in range(8): env.rpc('place_obj', inst=env.kn, position=[float(x) for x in po], quaternion_wxyz=qw); env.tool.act('right', p_, q_, 0.0, 'repin', steps=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--npert', type=int, default=3); ap.add_argument('--nview', type=int, default=6); ap.add_argument('--fixed', action='store_true')  # TEST mode (PREREG-T5): views A=az180/0.15m, B=az235/0.15m, no aim noise
    ap.add_argument('--gpu', type=int, required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
    out = Path(a.out); (out / 'images').mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(5000 + a.seed + (77777 if a.fixed else 0)); stat = dict(seed=a.seed, ok=0, err=None)
    env = F.Env(out / 'sim' / ('s%d' % a.seed), a.gpu, extend=2500); recs = []
    try:
        assert env.setup_handover(a.seed)
        up = env.T_link6(); up[2, 3] += 0.06; env.move_to(up, 'up', 120, 0.002)
        for p in range(a.npert):
            sid = a.seed * 10 + p; hov = rng.uniform(0.0, 0.06); eyaw = rng.uniform(-30, 30); ex, ey = rng.uniform(-0.006, 0.006, 2); inserted = bool(p == a.npert - 1 and rng.random() < 0.5)  # v2: insertion only as the LAST state of an episode
            T = env.slot_goal(0.045 + hov if not inserted else 0.045, eyaw=F.KEY['keyway_yaw_deg'] + (0 if inserted else eyaw)); T[:3, 3] += [0, 0, 0] if inserted else [ex, ey, 0]
            Gnom = np.linalg.inv(F.grasp_in_obj(F.KEY['grasp']['beta'], env.handover_roll, F.KEY['grasp']['d']))
            if p > 0: env.tool.set_gripper('right', opening=0.3, reason='loosen'); pin(env, Gnom)  # v5: restore the nominal (stable) grasp before moving to the next state
            env.move_to(F.target_hand(T, env.G()), 'pose%d' % p, 250, 0.002)
            if not inserted:  # in-hand perturbation by re-pinning (TRAINING data only)
                dz, rp, rt = rng.uniform(-0.012, 0.012), rng.uniform(-20, 20), rng.uniform(-8, 8)
                L6 = env.T_link6(); K = L6 @ np.linalg.inv(F.grasp_in_obj(F.KEY['grasp']['beta'], env.handover_roll, F.KEY['grasp']['d'])); E = np.eye(4);  # v2: perturb around the NOMINAL grasp (no cascade after a drop)
                E[:3, :3] = R.from_euler('yx', [rp, rt], degrees=True).as_matrix(); E[:3, 3] = [0, 0, dz]; K2 = K @ E; Gint = np.linalg.inv(L6) @ K2
                Ro = K2[:3, :3]; po = K2[:3, 3] - Ro @ env.cl; q = R.from_matrix(Ro).as_quat(); qw = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]; p_, q_ = env.tool.eef('right')
                env.tool.set_gripper('right', opening=0.3, reason='loosen')
                for _ in range(12): env.rpc('place_obj', inst=env.kn, position=[float(x) for x in po], quaternion_wxyz=qw); env.tool.act('right', p_, q_, 0.0, 'repin', steps=1)
            else:  # lower into the lock 5-30 mm with the (truth) realigned key
                env.realign(); d = rng.uniform(0.005, 0.030); env.ref_descend(depth_mm=(0.045 - 0.0307 + d) * 1000 + 10.7 - 15.0)  # stops on stall anyway
            env.tool.act('right', *env.tool.eef('right'), 0.0, 'settle', steps=5); Gp = env.G() if inserted else Gint; Lhold = env.T_link6(); pin(env, Gp) if not inserted else None  # v4: keep the INTENDED perturbed pose (large perturbations slip at once); v3: the state's in-hand pose; re-pinned before every view (slow slips under gravity)
            L6r = env.T_link6(); aim0 = L6r[:3, :3] @ (F.TCP + np.array([0.03, 0, 0])) + L6r[:3, 3]; views = 0
            for v in range(a.nview + 1):
                if not inserted and np.linalg.norm((env.T_link6() @ Gp)[:3, 3] - env.T_obj()[:3, 3]) > 0.001: pin(env, Gp)
                tr0 = env.truth()
                if tr0.get('inhand_shift_mm', 0) > 30 or tr0['key_tilt_deg'] > 30: stat['err'] = 'key lost at state %d view %d' % (p, v); break
                if v < a.nview:  # left side view
                    az = rng.uniform(140, 260); dist = rng.uniform(0.12, 0.18); aim = aim0 + np.r_[rng.uniform(-0.015, 0.015, 2), rng.uniform(-0.01, 0.01)]
                    if a.fixed: az, dist, aim = (180.0, 235.0)[v], 0.15, aim0
                    h = np.array([math.cos(math.radians(az)), math.sin(math.radians(az)), 0.0]); xv = -h; zv = np.array([0, 0, 1.0]); yv = np.cross(zv, xv)
                    L = np.eye(4); L[:3, :3] = np.column_stack([xv, yv, zv]); view = L[:3, :3] @ (CAM_VIEW_L6 / np.linalg.norm(CAM_VIEW_L6)); L[:3, 3] = aim - dist * view - L[:3, :3] @ CAM_IN_L6
                    env.tool.line('left', L[:3, 3], CT.q_of(L), 150, 0.004, 'kpview', 'gen'); A = CT.T_of(*env.tool.eef('left'))
                    if np.linalg.norm(A[:3, 3] - L[:3, 3]) > 0.01: continue
                    cam = 'cam_left_wrist'
                else: cam = 'cam_right_wrist'
                if a.fixed: env.tool.act('right', Lhold[:3, 3], CT.q_of(Lhold), 0.0, 'hold', steps=3)  # TEST: same right pose for both views
                else: env.tool.act('right', *env.tool.eef('right'), 0.0, 'hold', steps=1)
                c = env.rpc('camera')[cam]; Kt = env.T_obj(); P = (Kt[:3, :3] @ KP.T).T + Kt[:3, 3]; uv = project(c['intrinsics'], c['extrinsics_cam_to_world'], P)
                inlock = [bool(pt[2] < env.top and np.hypot(pt[0] - env.Ts[0, 3], pt[1] - env.Ts[1, 3]) < 0.03) for pt in P]  # v2: inside the lock = occluded
                arm = 'left' if 'left' in cam else 'right'; Tfk = CT.T_of(*env.tool.eef(arm)); Mfk = np.eye(4); Mfk[:3, :3] = np.column_stack([[0, -1.0, 0], [0.5, 0, 0.866], [-0.866, 0, 0.5]]); Mfk[:3, 3] = CAM_IN_L6
                fk_err = float(np.linalg.norm((Tfk @ Mfk)[:3, 3] - np.asarray(c['extrinsics_cam_to_world'])[:3, 3]))  # camera RPC vs link6 FK mount (stale-extrinsics check)
                im = next(i for i in env.ad.packet['images'] if i['camera'] == cam); iid = 'L%03d-s%d-v%02d' % (sid, a.seed, v); fn = '%s-%s.jpg' % (iid, 'lw' if arm == 'left' else 'rw')
                Image.open(im['path']).convert('RGB').save(out / 'images' / fn, quality=95)
                tr = env.truth(); recs.append(dict(sample_id=iid, image=fn, layout=sid, seed=a.seed, camera=cam, intrinsics=c['intrinsics'], extrinsics_cam_to_world=c['extrinsics_cam_to_world'],
                                                   outlets=P.tolist(), outlets_uv=uv, in_image=[u is not None and 0 <= u[0] < 640 and 0 <= u[1] < 480 and not o for u, o in zip(uv, inlock)], in_lock=inlock, keypoint_names=KP_NAMES,
                                                   key_pose=Kt.tolist(), link6_right=L6r.tolist(), link6_cam_arm=Tfk.tolist(), cam_fk_vs_rpc_m=round(fk_err, 5), inserted=inserted,
                                                   insertion_depth_mm=tr['insertion_depth_mm'], key_tilt_deg=tr['key_tilt_deg'], inhand_shift_mm=tr.get('inhand_shift_mm'), step=int(env.ad.packet['step_id'])))
                views += 1
            stat['ok'] += views
            if stat['err']: break
    except Exception as e:  # noqa: BLE001
        stat['err'] = '%s: %s' % (type(e).__name__, str(e)[:300])
    finally:
        env.close_sim()
        with open(out / ('L%03d-s%d.jsonl' % (a.seed, a.seed)), 'w') as f:
            for r in recs: f.write(json.dumps(r) + '\n')
        print(json.dumps(stat), flush=True)


if __name__ == '__main__':
    main()
