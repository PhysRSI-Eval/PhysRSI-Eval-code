"""keypoint-tool factory, instance 1 (plug_in_charger) state generator (2026-10-29 12:0xZ, post-training line). Real physics, no teleport.
Native reset of a layout -> physical top-down pinch of the charger with a random grasp (flip-head scripted2 parameter ranges) -> lift ->
carry the charger to a hover above a random outlet (TCP 5-12 cm above the socket top, xy +-1.5 cm, wrist yaw +-25 deg) -> the IDLE arm
parks at --nview side views (finepose park(): camera 0.12-0.18 m from the aim = holding TCP + 3 cm along the approach, 30 deg down,
azimuth 140-260 deg for a left observer, mirrored for a right observer, aim noise +-1.5 cm) and captures its wrist RGB + depth
(gpd_server_ext, approximate_depth mm); plus the holding wrist and the head. Cameras = FK link6 x cplabels.MOUNT (deploy proprio).
Records truth poses of the charger and the socket for the CAD-projected labels (make_labels_kp.py).
  python kp_charger_gen.py --gpu G --jobs NAME=layout:seed[,..] --out DIR"""
import argparse, json, math, shutil, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/lizekai/RSIws/experiments/ee-coin-g1-20261028/src'); sys.path.insert(0, '/lizekai/RSIws/experiments/regrasp-20261027/env/pkg'); sys.path.insert(0, '/lizekai/RSIws/experiments/ee-coinpose-20261028/src')
import coin_sim  # noqa: E402
import cplabels as CL  # noqa: E402
from rsi import combined_tools as CT, stage_tools as ST  # noqa: E402
_op = ST._Tool.opening
ST._Tool.opening = lambda self, arm: (lambda o: 0.0 if o < 0.5 else o)(_op(self, arm))
_pp = coin_sim.RA.subprocess.Popen
PYP = ':'.join(['/lizekai/RSIws/experiments/mva-20261029/src', '/lizekai/RSIws/experiments/combined-online-20261026/plug-offline-eval/v14/src', '/lizekai/RSIws/experiments/inhand-pose-20261026/v3/gen'])


def _popen(argv, **kw):  # depth-enabled server (gpd_server_ext: ip3_server_ext + private env_cfg with depth annotators)
    import os as _os
    argv = [('gpd_server_ext' if x in ('pe_server_ext', 'replay_server') else x) for x in argv]; env = dict(kw.get('env') or _os.environ); env['PYTHONPATH'] = PYP + ':' + env.get('PYTHONPATH', ''); kw['env'] = env
    return _pp(argv, **kw)


coin_sim.RA.subprocess.Popen = _popen
RA = coin_sim.RA; TCP_OFF = ST.FINGER_PAD; CAM_IN_L6 = np.array([0.085, 0.0, 0.051]); CAM_VIEW_L6 = np.array([0.87, 0.0, -0.5])


def unit(v): v = np.asarray(v, float); return v / np.linalg.norm(v)
def mk(Rm, p): M = np.eye(4); M[:3, :3] = Rm; M[:3, 3] = p; return M


def L6_for(tcp, ax, ay):
    ax = unit(ax); ay = unit(np.asarray(ay, float) - ax * (np.asarray(ay, float) @ ax)); Rm = np.column_stack([ax, ay, np.cross(ax, ay)]); return mk(Rm, np.asarray(tcp, float) - Rm @ np.array([TCP_OFF, 0, 0]))


def wait_ram(min_gb=25):
    while int(next(l for l in open('/proc/meminfo') if l.startswith('MemAvailable')).split()[1]) / 1048576 < min_gb: time.sleep(30)


def run(sim, lay, seed, out, obsdir, nview):
    rpc = sim.rpc.request; rng = np.random.default_rng(seed); rec = dict(layout=lay, seed=seed, failure=None, views=[]); out.mkdir(parents=True, exist_ok=True); (out / 'images').mkdir(exist_ok=True)
    try:
        rpc('reset', layout_id=lay); rpc('extend_limit', extra=600); ad = coin_sim.FakeAdapter(sim, obsdir); tool = ST._Tool(ad, 'kf', 'kf', None, {})
        o = rpc('objects'); ch = next(n for n in o if n.startswith('charger')); so = next(n for n in o if n.startswith('socket')); rec.update(charger=ch, socket=so)
        T = lambda inst: (lambda x: CT.T_of(x['position'], x['quaternion_wxyz']))(rpc('objects')[inst])
        g = RA.plain(rpc('rgbhint_geom', bbox=[ch, so])); V = np.asarray(g['bbox'][ch]['local_vertices'], float); lo, hi = V.min(0), V.max(0); ctr = (lo + hi) / 2; ext = hi - lo
        To = T(ch); Rw = To[:3, :3]; up = int(np.argmax(np.abs(Rw[2]))); hor = sorted([k for k in range(3) if k != up], key=lambda k: -ext[k])
        pinch = int(rng.integers(2)); kp_, ko = hor[pinch], hor[1 - pinch]; sign = int(rng.choice([1, -1])); depth = float(rng.uniform(0.3, 0.75)); lat = float(rng.uniform(-0.015, 0.015)); tilt = float(rng.uniform(-30, 30)); yaw = float(rng.uniform(-20, 20))
        cw = Rw @ ctr + To[:3, 3]; top = float(g['bbox'][ch]['z_max']); height = top - float(g['bbox'][ch]['z_min'])
        eo = unit(Rw[:, ko] - [0, 0, Rw[2, ko]]); tcp = np.array([cw[0], cw[1], top - depth * height + 0.006]) + lat * eo; tl = math.radians(tilt)
        ax = math.cos(tl) * np.array([0, 0, -1.0]) + math.sin(tl) * eo; ay = Rot.from_rotvec(math.radians(yaw) * unit(ax)).as_matrix() @ (sign * (Rw[:, kp_] - [0, 0, Rw[2, kp_]]))
        rec['grasp'] = dict(pinch=pinch, sign=sign, depth=round(depth, 3), lateral=round(lat, 4), tilt=round(tilt, 1), yaw=round(yaw, 1))
        arm = 'left' if cw[0] < 0 else 'right'; idle = 'right' if arm == 'left' else 'left'; rec.update(arm=arm, idle=idle); Lg = L6_for(tcp, ax, ay); pre = Lg.copy(); pre[2, 3] += 0.08
        tool.set_gripper(arm, opening=1.0, reason='open'); tool.line(arm, pre[:3, 3], CT.q_of(pre), 160, 0.004, 'pre', 'grasp')
        ST.STEP_M = 0.02; tool.line(arm, Lg[:3, 3], CT.q_of(Lg), 60, 0.002, 'in', 'grasp'); ST.STEP_M = 0.049; tool.set_gripper(arm, opening=0.0, reason='close')
        p, q = tool.eef(arm); ST.STEP_M = 0.03; tool.line(arm, p + [0, 0, 0.10], q, 60, 0.01, 'lift', 'grasp'); ST.STEP_M = 0.049
        Gm = np.linalg.inv(CT.T_of(*tool.eef(arm))) @ T(ch)
        if np.linalg.norm(Gm[:3, :3] @ ctr + Gm[:3, 3] - [TCP_OFF, 0, 0]) > 0.06: rec['failure'] = 'not_held'; raise StopIteration
        Ts = T(so); outlets = [Ts[:3, :3] @ np.array([x, CT.OUT_Y, 0.0]) + Ts[:3, 3] for x in CT.OFFS]; oc = outlets[int(rng.integers(3))]; oc[2] = float(g['bbox'][so]['z_max'])
        L6 = CT.T_of(*tool.eef(arm)); tgt_tcp = oc + np.r_[rng.uniform(-0.015, 0.015, 2), rng.uniform(0.05, 0.12)]
        # 多数布局插座在另一侧，持握臂够不到：目标 x 截到持握臂一侧中线外 5 cm 内（同侧布局则是真正悬停在插座上方）
        tgt_tcp[0] = min(tgt_tcp[0], 0.05) if arm == 'left' else max(tgt_tcp[0], -0.05); rec['hover_over_socket'] = bool(abs(tgt_tcp[0] - oc[0]) < 0.02)
        Rn = Rot.from_rotvec([0, 0, math.radians(rng.uniform(-25, 25))]).as_matrix() @ L6[:3, :3]; Lt = mk(Rn, tgt_tcp - Rn @ [TCP_OFF, 0, 0])
        ST.STEP_RAD = 0.2; r = tool.line(arm, Lt[:3, 3], CT.q_of(Lt), 200, 0.006, 'carry', 'hover'); ST.STEP_RAD = 0.34; rec['carry_err_mm'] = round(r['measured']['position_error_m'] * 1000, 1)
        L6 = CT.T_of(*tool.eef(arm)); aim0 = L6[:3, :3] @ (np.array([TCP_OFF + 0.03, 0, 0])) + L6[:3, 3]; home_idle = tool.eef(idle)
        azs = rng.uniform(120, 260, nview) if idle == 'left' else 180 - rng.uniform(120, 260, nview)

        def snap(tag, cams):
            ob = rpc('observe'); st = np.asarray(ob['states'], float); v = dict(tag=tag, step=int(ob['step_id']), states=st.tolist(), charger=RA.plain(rpc('objects')[ch]), socket=RA.plain(rpc('objects')[so]),
                                                                              link6={a: np.round(CT.T_of(*tool.eef(a)), 6).tolist() for a in ('left', 'right')}, images={}, depth={})
            from PIL import Image
            for cam in cams:
                f = out / 'images' / ('%s-%s.jpg' % (tag, cam)); Image.fromarray(np.asarray(ob['images'][cam], np.uint8)).save(f, quality=95); v['images'][cam] = 'images/' + f.name
                if cam != 'cam_high':
                    d = np.asarray(rpc('depth', cam=cam), np.uint16); df = out / 'images' / ('%s-%s-depth.png' % (tag, cam)); Image.fromarray(d).save(df); v['depth'][cam] = 'images/' + df.name
                    Lc = np.asarray(v['link6']['left' if 'left' in cam else 'right']); v.setdefault('cam_to_world', {})[cam] = np.round(Lc @ CL.MOUNT, 6).tolist()
            rec['views'].append(v)
        snap('hold', ('cam_high', 'cam_%s_wrist' % arm))
        aim_s = oc + np.array([0, 0, 0.01])  # 一半视角对准插座孔（空闲臂多半在插座一侧），一半对准手里的充电器
        for k, az in enumerate(azs):
            base = aim0 if k % 2 == 0 else aim_s; aim = base + rng.uniform(-0.015, 0.015, 3); dist = float(rng.uniform(0.12, 0.18)); h = np.array([math.cos(math.radians(az)), math.sin(math.radians(az)), 0.0])
            xv = -h; zv = np.array([0, 0, 1.0]); yv = np.cross(zv, xv); Lp = np.eye(4); Lp[:3, :3] = np.column_stack([xv, yv, zv]); view = Lp[:3, :3] @ unit(CAM_VIEW_L6); Lp[:3, 3] = aim - dist * view - Lp[:3, :3] @ CAM_IN_L6
            r = tool.line(idle, Lp[:3, 3], CT.q_of(Lp), 300, 0.004, 'park%d' % k, 'view'); err = r['measured']['position_error_m']
            if err > 0.02: rec.setdefault('unreached', []).append(dict(k=k, az=round(float(az), 1), err_mm=round(err * 1000, 1))); continue
            snap('v%d%s' % (k, 'c' if k % 2 == 0 else 's'), ('cam_%s_wrist' % idle,))
        tool.line(idle, np.asarray(home_idle[0]), home_idle[1], 200, 0.01, 'home', 'view')
    except StopIteration:
        pass
    except Exception as e:  # noqa: BLE001
        rec['failure'] = rec['failure'] or 'error'; rec['error'] = '%s: %s' % (type(e).__name__, str(e)[:300]); rec['tb'] = traceback.format_exc()[-1500:]
    (out / 'record.json').write_text(json.dumps(RA.plain(rec))); print(json.dumps(dict(out=out.name, failure=rec['failure'], views=len(rec['views']), unreached=len(rec.get('unreached', [])))), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--gpu', type=int, required=True); ap.add_argument('--jobs', required=True); ap.add_argument('--out', required=True); ap.add_argument('--nview', type=int, default=4); ap.add_argument('--name', default='kf')
    a = ap.parse_args()
    for spec in a.jobs.split(','):
        nm, body = spec.split('='); lay, seed = map(int, body.split(':'))
        if (Path(a.out) / nm / 'record.json').exists(): continue
        wait_ram(); simdir = Path('/root/pe-1026/sim') / ('kf-%s-%s' % (a.name, nm)); sim = None
        try:
            sim = RA.Sim('plug_in_charger', simdir, a.gpu); run(sim, lay, seed, Path(a.out) / nm, simdir / 'obs', a.nview)
        except Exception as e:  # noqa: BLE001
            (Path(a.out) / nm).mkdir(parents=True, exist_ok=True); (Path(a.out) / nm / 'record.json').write_text(json.dumps(dict(failure='sim_start_error', error=str(e)[:300])))
        finally:
            if sim is not None:
                try: sim.close()
                except Exception: pass  # noqa: BLE001
            time.sleep(3); subprocess.run(['pkill', '-f', 'output %s ' % simdir]); shutil.rmtree(simdir, ignore_errors=True)
