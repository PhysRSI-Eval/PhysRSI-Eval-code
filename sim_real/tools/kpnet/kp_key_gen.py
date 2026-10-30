"""keypoint factory, insert_key (C1 instance 2; 2026-10-30 07:0xZ, post-training line). Real physics, no teleport.
Native reset -> top-down pinch of the key lying on the table (closing across the key width, at the bow end; jitter) -> lift -> carry to
the finepose goal frame above the lock (slot_goal: long axis up, tip down, x along the keyway; hover 45 mm + U(0, 40) mm, xy +-6 mm,
keyway yaw +-30 deg; 15 %: hover 45 - U(5, 25) mm = tip in the keyhole) using the truth in-hand pose (data generation only) ->
captures: holding wrist + head, then the FREE arm at the finepose parks (A azimuth 180, B 235 deg for a left observer, mirrored for a
right one; +-20 deg, 0.12-0.18 m, aim = held TCP + 3 cm, rising 12 cm above the scene before every park so it does not bump), RGB + sim
depth (gpd2_server_ext), FK cameras, truth key / lock poses per view (static check is done at labelling).
  python kp_key_gen.py --gpu G --out DIR --jobs NAME=layout:seed[,..] --name W"""
import argparse, json, math, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
_P0 = subprocess.Popen


def _popen0(cmd, **kw):
    if isinstance(cmd, list) and 'pe_server_ext' in cmd:
        cmd = [('gpd2_server_ext' if x == 'pe_server_ext' else x) for x in cmd]; env = dict(kw.get('env') or os.environ)
        env['PYTHONPATH'] = '/lizekai/RSIws/experiments/kpfactory-20261029/src:/lizekai/RSIws/experiments/combined-online-20261026/plug-offline-eval/v14/src:' + env.get('PYTHONPATH', ''); kw['env'] = env
    return _P0(cmd, **kw)


subprocess.Popen = _popen0
sys.path.insert(0, '/lizekai/RSIws/experiments/ee-coin-g1-20261028/src'); sys.path.insert(0, '/lizekai/RSIws/experiments/regrasp-20261027/env/pkg'); sys.path.insert(0, '/lizekai/RSIws/experiments/ee-coinpose-20261028/src')
import coin_sim  # noqa: E402
import cplabels as CLB  # noqa: E402
from rsi import combined_tools as CT, stage_tools as ST  # noqa: E402
_op = ST._Tool.opening
ST._Tool.opening = lambda self, arm: (lambda o: 0.0 if o < 0.5 else o)(_op(self, arm))
RA = coin_sim.RA; TCP_OFF = ST.FINGER_PAD; CAM_IN_L6 = np.array([0.085, 0.0, 0.051]); CAM_VIEW_L6 = np.array([0.87, 0.0, -0.5])
EVAL_AXIS = np.array([0.5, -math.sqrt(3) / 2, 0.0]); KEYWAY_YAW = 60.0; SEAT_DEPTH = 0.035


def unit(v): v = np.asarray(v, float); return v / np.linalg.norm(v)
def mk(Rm, p): M = np.eye(4); M[:3, :3] = Rm; M[:3, 3] = p; return M


def L6_for(tcp, ax, ay):
    ax = unit(ax); ay = unit(np.asarray(ay, float) - ax * (np.asarray(ay, float) @ ax)); Rm = np.column_stack([ax, ay, np.cross(ax, ay)]); return mk(Rm, np.asarray(tcp, float) - Rm @ np.array([TCP_OFF, 0, 0]))


def run(sim, lay, seed, out, obsdir):
    rpc = sim.rpc.request; rng = np.random.default_rng(seed); rec = dict(layout=lay, seed=seed, failure=None, states=[]); out.mkdir(parents=True, exist_ok=True); (out / 'images').mkdir(exist_ok=True)
    try:
        rpc('reset', layout_id=lay); rpc('extend_limit', extra=800); ad = coin_sim.FakeAdapter(sim, obsdir); tool = ST._Tool(ad, 'kk', 'kk', None, {})
        o = rpc('objects'); kn = next(n for n in o if n.startswith('key') and 'slot' not in n); sn = next(n for n in o if 'slot' in n); rec.update(key=kn, lock=sn)
        T = lambda inst: (lambda x: CT.T_of(x['position'], x['quaternion_wxyz']))(rpc('objects')[inst])
        g = RA.plain(rpc('rgbhint_geom', bbox=[kn, sn])); LV = np.asarray(g['bbox'][kn]['local_vertices'], float); cl = (LV.max(0) + LV.min(0)) / 2; half = float((LV.max(0) - LV.min(0))[2]) / 2
        top = float(g['bbox'][sn]['z_max']); rec.update(key_centre_local=cl.tolist(), key_half=half, lock_top=top)
        Tk = T(kn); Rw = Tk[:3, :3]; bow = Rw @ (cl + np.array([0, 0, rng.uniform(0.0, 0.012)])) + Tk[:3, 3]  # pinch near the centre of mass (bow-end pinches dropped the key on the 90 deg turn)
        arm = 'left' if bow[0] < 0 else 'right'; free = 'right' if arm == 'left' else 'left'; rec.update(arm=arm, free=free)
        ay = Rw[:, 0] - np.array([0, 0, Rw[2, 0]]); ay = Rot.from_rotvec([0, 0, math.radians(rng.uniform(-10, 10))]).as_matrix() @ ay
        tcp = bow.copy(); tcp[2] = float(g['bbox'][kn]['z_max']) - 0.006; Lg = L6_for(tcp, [0, 0, -1.0], ay); pre = Lg.copy(); pre[2, 3] += 0.08
        tool.set_gripper(arm, opening=1.0, reason='open'); tool.line(arm, pre[:3, 3], CT.q_of(pre), 160, 0.004, 'pre', 'grasp')
        ST.STEP_M = 0.015; tool.line(arm, Lg[:3, 3], CT.q_of(Lg), 60, 0.002, 'in', 'grasp'); ST.STEP_M = 0.049; tool.set_gripper(arm, opening=0.0, reason='close')
        p, q = tool.eef(arm); ST.STEP_M = 0.03; tool.line(arm, p + [0, 0, 0.10], q, 60, 0.01, 'lift', 'grasp'); ST.STEP_M = 0.049
        Kc = lambda: (lambda M: mk(M[:3, :3], M[:3, :3] @ cl + M[:3, 3]))(T(kn)); Gm = np.linalg.inv(CT.T_of(*tool.eef(arm))) @ Kc()
        if np.linalg.norm(Gm[:3, 3] - [TCP_OFF, 0, 0]) > 0.06: rec['failure'] = 'not_held'; raise StopIteration
        Ts = T(sn); axw = Ts[:3, :3] @ EVAL_AXIS; yaw0 = math.atan2(axw[1], axw[0]) + math.radians(KEYWAY_YAW)
        home_free = tool.eef(free)
        # goal yaw with the least wrist rotation (any keyway yaw is fine for keypoint data; the labels follow the truth)
        L0 = CT.T_of(*tool.eef(arm)); best = None
        for yy in range(0, 360, 15):
            Tg = np.eye(4); Tg[:3, :3] = Rot.from_euler('z', math.radians(yy)).as_matrix(); Lt = Tg @ np.linalg.inv(Gm); dd = Rot.from_matrix(L0[:3, :3].T @ Lt[:3, :3]).magnitude()
            if best is None or dd < best[0]: best = (dd, math.radians(yy))
        yaw0 = best[1]; rec['goal_yaw_deg'] = round(math.degrees(yaw0), 1)
        for s in range(6):
            partial = rng.random() < 0.15; hov = 0.045 - rng.uniform(0.005, 0.025) if partial else 0.045 + rng.uniform(0, 0.04)
            Tg = np.eye(4); Tg[:3, :3] = Rot.from_euler('z', yaw0 + math.radians(rng.uniform(-20, 20))).as_matrix()
            Tg[:3, 3] = [Ts[0, 3] + (0 if partial else rng.uniform(-0.006, 0.006)), Ts[1, 3] + (0 if partial else rng.uniform(-0.006, 0.006)), top - SEAT_DEPTH + half + hov]
            Gm = np.linalg.inv(CT.T_of(*tool.eef(arm))) @ Kc(); Lt = Tg @ np.linalg.inv(Gm)
            ST.STEP_M, ST.STEP_RAD = 0.015, 0.05; r = tool.line(arm, Lt[:3, 3], CT.q_of(Lt), 400, 0.002, 'state%d' % s, 'kp'); ST.STEP_M, ST.STEP_RAD = 0.049, 0.34
            st = dict(k=s, kind='partial' if partial else 'hover', reach_mm=round(r['measured']['position_error_m'] * 1000, 2), views=[])
            if np.linalg.norm((np.linalg.inv(CT.T_of(*tool.eef(arm))) @ Kc())[:3, 3] - [TCP_OFF, 0, 0]) > 0.06: rec['failure'] = 'dropped'; break

            def snap(tag, cams):
                from PIL import Image
                ob = rpc('observe'); v = dict(tag=tag, step=int(ob['step_id']), joints=list(map(float, ob['states'])), key=RA.plain(rpc('objects')[kn]), lock=RA.plain(rpc('objects')[sn]),
                                              link6={a: np.round(CT.T_of(*tool.eef(a)), 6).tolist() for a in ('left', 'right')}, images={}, depth={}, cam_to_world={})
                for cam in cams:
                    f = out / 'images' / ('%s-%s.jpg' % (tag, cam)); Image.fromarray(np.asarray(ob['images'][cam], np.uint8)).save(f, quality=95); v['images'][cam] = 'images/' + f.name
                    d = np.asarray(rpc('depth', cam=cam), np.uint16); df = out / 'images' / ('%s-%s-depth.png' % (tag, cam)); Image.fromarray(d).save(df); v['depth'][cam] = 'images/' + df.name
                    if cam != 'cam_high': v['cam_to_world'][cam] = np.round(np.asarray(v['link6']['left' if 'left' in cam else 'right']) @ CLB.MOUNT, 6).tolist()
                return v
            st['views'].append(snap('s%d-hold' % s, ('cam_%s_wrist' % arm, 'cam_high')))
            L6 = CT.T_of(*tool.eef(arm)); aim = L6[:3, :3] @ np.array([TCP_OFF + 0.03, 0, 0]) + L6[:3, 3]
            for tag, az0 in (('A', 180.0), ('B', 235.0)):
                az = (az0 if free == 'left' else 180 - az0) + rng.uniform(-20, 20); dist = rng.uniform(0.12, 0.18); h = np.array([math.cos(math.radians(az)), math.sin(math.radians(az)), 0.0])
                xv = -h; zv = np.array([0, 0, 1.0]); yv = np.cross(zv, xv); Lp = np.eye(4); Lp[:3, :3] = np.column_stack([xv, yv, zv]); Lp[:3, 3] = aim - dist * (Lp[:3, :3] @ unit(CAM_VIEW_L6)) - Lp[:3, :3] @ CAM_IN_L6
                pf, qf = tool.eef(free); tool.line(free, [pf[0], pf[1], max(pf[2], aim[2] + 0.12)], qf, 60, 0.01, 'rise', 'view')  # rise first: do not bump
                tool.line(free, [Lp[0, 3], Lp[1, 3], max(Lp[2, 3], aim[2] + 0.12)], CT.q_of(Lp), 150, 0.01, 'over', 'view')
                rr = tool.line(free, Lp[:3, 3], CT.q_of(Lp), 120, 0.004, 'park', 'view')
                if rr['measured']['position_error_m'] > 0.02: st.setdefault('unreached', []).append(tag); continue
                st['views'].append(snap('s%d-%s' % (s, tag), ('cam_%s_wrist' % free,)))
            pf, qf = tool.eef(free); tool.line(free, [pf[0], pf[1], pf[2] + 0.1], qf, 40, 0.01, 'rise', 'view'); tool.line(free, list(home_free[0]), home_free[1], 150, 0.02, 'home', 'view')
            rec['states'].append(st); (out / 'record.json').write_text(json.dumps(RA.plain(rec)))
    except StopIteration:
        pass
    except Exception as e:  # noqa: BLE001
        rec['failure'] = rec['failure'] or 'error'; rec['error'] = '%s: %s' % (type(e).__name__, str(e)[:300]); rec['tb'] = traceback.format_exc()[-1500:]
    (out / 'record.json').write_text(json.dumps(RA.plain(rec))); print(json.dumps(dict(out=out.name, failure=rec['failure'], states=len(rec['states']), views=sum(len(s['views']) for s in rec['states']))), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--gpu', type=int, required=True); ap.add_argument('--jobs', required=True); ap.add_argument('--out', required=True); ap.add_argument('--name', default='kk')
    a = ap.parse_args()
    for spec in a.jobs.split(','):
        nm, body = spec.split('='); lay, seed = map(int, body.split(':'))
        if (Path(a.out) / nm / 'record.json').exists(): continue
        while int(next(l for l in open('/proc/meminfo') if l.startswith('MemAvailable')).split()[1]) / 1048576 < 25: time.sleep(30)
        simdir = Path('/root/pe-1026/sim') / ('ip4-kk-%s-%s' % (a.name, nm)); sim = None
        try:
            sim = RA.Sim('insert_key', simdir, a.gpu); run(sim, lay, seed, Path(a.out) / nm, simdir / 'obs')
        except Exception as e:  # noqa: BLE001
            (Path(a.out) / nm).mkdir(parents=True, exist_ok=True); (Path(a.out) / nm / 'record.json').write_text(json.dumps(dict(failure='sim_start_error', error=str(e)[:300])))
        finally:
            if sim is not None:
                try: sim.close()
                except Exception: pass  # noqa: BLE001
            time.sleep(3); subprocess.run(['pkill', '-f', 'output %s ' % simdir]); shutil.rmtree(simdir, ignore_errors=True)
