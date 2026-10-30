"""keypoint factory, plug_in_charger CO-VISIBLE states (2026-10-29 12:3xZ, post-training line). Runs the capability line's rg3f-v2
executor (macro-tools src/rg3fast2.py -> loc1cov.py; their code unchanged, imported read-only) on Astra start states, in TRUTH mode, and
hooks loc1cov_lib.insert_recipe: when the charger has been carried to the hover above the target outlet, instead of inserting it
captures --n-states states per job:
  holding link6 = hover + dz U(0, 40 mm) (15 %: seated + U(0, 10 mm) = partly inserted), xy +-10 mm, yaw +-20 deg about the TCP;
  per state: the holding wrist + the FREE arm at the LOC1 parks A (0.13 m, -15 deg) / B (0.17 m, +20 deg) via CT.aim(mount, gaze, h, yaw)
  with h +-0.02 m, yaw +-10 deg, gaze (strip top centre) +-1.5 cm; RGB + sim depth (gpd2_server_ext) + FK link6 + truth charger / socket.
Spec: reports/progress/mailbox/to-ip4.md 12:0xZ (capability line). Real physics, no teleport. The job ends after the captures.
  (cwd = macro-tools-20261027)  python kp_hover_capture.py --kp-out DIR --n-states N src/loc1cov.py <loc1cov args> --gpu G --out O"""
import json, math, os, subprocess, sys
from pathlib import Path
import numpy as np
K = Path('/lizekai/RSIws/experiments/kpfactory-20261029'); MT = Path('/lizekai/RSIws/experiments/macro-tools-20261027')
argv = sys.argv[1:]; KP_OUT = Path(argv[argv.index('--kp-out') + 1]); NST = int(argv[argv.index('--n-states') + 1])
for f in ('--kp-out', '--n-states'):
    i = argv.index(f); del argv[i:i + 2]
_P = subprocess.Popen


def _popen(cmd, **kw):  # simulator server -> depth-enabled wrapper of the same pe_server_ext
    if isinstance(cmd, list) and 'pe_server_ext' in cmd:
        cmd = [('gpd2_server_ext' if x == 'pe_server_ext' else x) for x in cmd]; env = dict(kw.get('env') or os.environ); env['PYTHONPATH'] = str(K / 'src') + ':' + env.get('PYTHONPATH', ''); kw['env'] = env
    return _P(cmd, **kw)


subprocess.Popen = _popen
sys.path.insert(0, '/root/pe-1026/pkg'); sys.path.insert(0, str(MT / 'src'))
from rsi import combined_tools as CT, stage_tools as ST  # noqa: E402
import loc1cov_lib as LC  # noqa: E402
_orig_insert = LC.insert_recipe


def capture(tool, co, ad, rpc, objs, arm, Lh, Ls):
    sys.path.insert(0, '/lizekai/RSIws/experiments/ee-coinpose-20261028/src'); import cplabels as CLB
    from scipy.spatial.transform import Rotation as Rot
    from PIL import Image
    free = 'right' if arm == 'left' else 'left'; rng = np.random.default_rng(abs(hash(str(KP_OUT))) % (2 ** 32)); KP_OUT.mkdir(parents=True, exist_ok=True); (KP_OUT / 'images').mkdir(exist_ok=True)
    o = objs(); ch = next(n for n in o if n.startswith('charger')); so = next(n for n in o if n.startswith('socket')); gb = rpc('rgbhint_geom', bbox=[so])['bbox'][so]
    gaze0 = np.array([gb['pose'][0], gb['pose'][1], gb['z_max']]); mount = CT.CALIB['mount'] if hasattr(CT, 'CALIB') else None
    rec = dict(arm=arm, free=free, charger=ch, socket=so, Lh=np.asarray(Lh).tolist(), Ls=np.asarray(Ls).tolist(), states=[])

    def snap(tag, cams):
        ob = rpc('observe'); v = dict(tag=tag, step=int(ob['step_id']), joints=list(map(float, ob['states'])), charger=objs()[ch], socket=objs()[so],
                                      link6={a: np.round(np.asarray(co.l6(a)), 6).tolist() for a in ('left', 'right')}, images={}, depth={}, cam_to_world={})
        for cam in cams:
            f = KP_OUT / 'images' / ('%s-%s.jpg' % (tag, cam)); Image.fromarray(np.asarray(ob['images'][cam], np.uint8)).save(f, quality=95); v['images'][cam] = 'images/' + f.name
            d = np.asarray(rpc('depth', cam=cam), np.uint16); df = KP_OUT / 'images' / ('%s-%s-depth.png' % (tag, cam)); Image.fromarray(d).save(df); v['depth'][cam] = 'images/' + df.name
            if cam != 'cam_high': v['cam_to_world'][cam] = np.round(np.asarray(co.l6('left' if 'left' in cam else 'right')) @ CLB.MOUNT, 6).tolist()
        return v
    for s in range(NST):
        if rng.random() < 0.15: base = np.asarray(Ls).copy(); base[2, 3] += rng.uniform(0, 0.010); kind = 'partial'
        else: base = np.asarray(Lh).copy(); base[2, 3] += rng.uniform(0, 0.040); kind = 'hover'
        base[:2, 3] += rng.uniform(-0.010, 0.010, 2); yaw = rng.uniform(-20, 20); tcp = base[:3, :3] @ np.array([0.14, 0, 0]) + base[:3, 3]
        Rz = Rot.from_rotvec([0, 0, math.radians(yaw)]).as_matrix(); T = base.copy(); T[:3, :3] = Rz @ base[:3, :3]; T[:3, 3] = tcp - T[:3, :3] @ np.array([0.14, 0, 0])
        m0, r0 = ST.STEP_M, ST.STEP_RAD; ST.STEP_M, ST.STEP_RAD = 0.015, 0.08
        try: r = tool.line(arm, T[:3, 3].tolist(), LC._q_of(T), 120, 0.002, 'kp_state', 'kpcap')
        finally: ST.STEP_M, ST.STEP_RAD = m0, r0
        st = dict(k=s, kind=kind, dz_yaw=[float(base[2, 3] - np.asarray(Lh)[2, 3]), float(yaw)], reach_mm=round(r['measured']['position_error_m'] * 1000, 2), views=[])
        st['views'].append(snap('s%d-hold' % s, ('cam_%s_wrist' % arm, 'cam_high')))
        for tag, h, y in (('A', 0.13, -15.0), ('B', 0.17, 20.0)):
            gz = gaze0 + np.r_[rng.uniform(-0.015, 0.015, 2), 0.0]; hh = h + rng.uniform(-0.02, 0.02); yy = y + rng.uniform(-10, 10)
            try:
                Tlc = np.asarray(mount[free]) if mount is not None else CLB.MOUNT
                p, q = CT.aim(Tlc, gz, hh, yy); rr = tool.line(free, list(map(float, p)), list(map(float, q)), 120, 0.004, 'kp_park', 'kpcap')
                if rr['measured']['position_error_m'] > 0.02: st.setdefault('unreached', []).append(tag); continue
                v = snap('s%d-%s' % (s, tag), ('cam_%s_wrist' % free,)); v['park'] = dict(tag=tag, h=round(hh, 3), yaw=round(yy, 1), gaze=np.round(gz, 4).tolist()); st['views'].append(v)
            except Exception as e:  # noqa: BLE001
                st.setdefault('errors', []).append('%s: %s' % (tag, str(e)[:200]))
        rec['states'].append(st)
        (KP_OUT / 'record.json').write_text(json.dumps(rec, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))
    return rec


def insert_hook(name, tool, co, ad, rpc, objs, arm, Lh, Ls, seat_true, refresh=None, n_refresh=0):
    try:
        rec = capture(tool, co, ad, rpc, objs, arm, Lh, Ls); return dict(inserted=False, kp_capture=len(rec['states']))
    except Exception as e:  # noqa: BLE001
        import traceback; (KP_OUT / 'capture_error.txt').write_text(traceback.format_exc()); return dict(inserted=False, kp_capture_error=str(e)[:200])


LC.insert_recipe = insert_hook
import runpy  # noqa: E402
sys.argv = [str(MT / 'src' / 'rg3fast2.py')] + argv
runpy.run_path(str(MT / 'src' / 'rg3fast2.py'), run_name='__main__')
