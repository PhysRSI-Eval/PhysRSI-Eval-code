"""rg3census2.py (C1 line, 2026-10-29): census v2 of the 'hover unreachable' rows. Same frozen rg3f TRUTH chain as src/rg3census.py
(rg3fast.py / rg3fast_lib, run via loc1cov.py), plus
  1. contact diagnosis at EVERY stalled line of the chain (transport rise / travel / lower, hover, alternatives): PhysX overlap of the
     charger and both grippers' colliders (server op c1_contacts, c1/src/c1_server_ext.py), charger offset from the TCP (held / lost),
     charger box lowest corner vs table and strip top, strip displacement, controller-model progress from the stalled joints, cuRobo
     sphere clearance per link at the stalled joints (table plane + strip box);
  2. alternative approaches IN SIM after a hover stall (a census never inserts with the original target):
     ALT order: same outlet vertical approach (rise to 6 cm above the hover, then straight down), then other outlets, then the charger's
     180 deg symmetry, controller-model ok first; at most 3 tries; the first that reaches the hover runs the frozen R1 insertion
     (loc1cov_lib.insert_recipe, no refresh) so the row also says whether that approach inserts.
Diagnosis uses simulator truth (offline only).  python rg3census2.py [rg3fast options] src/loc1cov.py <loc1cov args...>"""
import importlib.util, json, os, runpy, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

C1 = '/lizekai/RSIws/experiments/macro-tools-20261027/c1/src'
_po = subprocess.Popen


def _popen_c1(argv, **kw):   # server module pe_server_ext -> c1_server_ext (adds op c1_contacts); c1/src on the server's PYTHONPATH
    argv = [('c1_server_ext' if x == 'pe_server_ext' else x) for x in argv]
    if kw.get('env') is not None: kw['env'] = dict(kw['env'], PYTHONPATH=C1 + os.pathsep + kw['env'].get('PYTHONPATH', ''))
    return _po(argv, **kw)


subprocess.Popen = _popen_c1
KEYS = {'--rg3f-tspeed': 'RG3F_TSPEED', '--rg3f-pre': 'RG3F_PRE_M', '--rg3f-refresh': 'RG3F_REFRESH_N', '--rg3f-v4': 'RG3F_V4_URL'}
argv = sys.argv[1:]
while argv and argv[0] in KEYS:
    os.environ[KEYS[argv[0]]] = argv[1]; argv = argv[2:]
sys.path.insert(0, '/root/pe-1026/pkg'); sys.path.insert(0, '/lizekai/RSIws/experiments/combined-online-20261026/plug-offline-eval/v14/src')
from rsi import stage_tools as ST, combined_tools as CT  # noqa: E402
sp = importlib.util.spec_from_file_location('grip_cmd', '/lizekai/RSIws/experiments/macro-tools-20261027/pkg-v2/rsi/grip_cmd.py'); G = importlib.util.module_from_spec(sp); sp.loader.exec_module(G)
G.install(ST._Tool)
sys.path.insert(0, '/lizekai/RSIws/experiments/macro-tools-20261027/src')
import loc1cov_lib as LC  # noqa: E402
import rg3fast_lib as RF  # noqa: E402
V = '/lizekai/RSIws/experiments/macro-tools-20261027'
_ins = LC.insert_recipe; CTX = {}; DIAG = []; TABLE_Z = 0.765
MATCH = ['charger', 'link6', 'link7', 'link8', 'finger', 'gripper', 'hand']


def _ld(n, f):
    s = importlib.util.spec_from_file_location(n, f); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def _cr():
    if 'cr' not in CTX:
        try:
            RC = _ld('reach_cu_v2', V + '/pkg-v2/rsi/reach_cu.py'); CTX['RC'] = RC; CTX['cr'] = RC.CollisionReach(width=0.0)
            k = RC.robot_cfg(0.0)['robot_cfg']['kinematics']; CTX['sph_names'] = [ln for ln in k['collision_link_names'] for _ in k['collision_spheres'][ln]]
        except Exception as e:  # noqa: BLE001
            CTX['cr'] = None; CTX['cu_err'] = '%s: %s' % (type(e).__name__, str(e)[:160])
    return CTX['cr']


def _rm():
    if 'rm' not in CTX: CTX['rm'] = _ld('reach_model_v15', V + '/pkg-v15/rsi/reach_model.py').Reach()
    return CTX['rm']


def diagnose(label, arm, target=None):
    """Truth-side snapshot at a stall (offline diagnosis only)."""
    co, ad, rpc, objs = CTX['co'], CTX['ad'], CTX['rpc'], CTX['objs']; d = dict(label=label, arm=arm, step=int(ad.packet['step_id']))
    try:
        i_ = CT.ARMS.index(arm); q = np.asarray(ad.packet['current_state'][7 * i_:7 * i_ + 6], float); L = co.l6(arm); tcp = co.tcp(arm)
        d.update(q=q.round(4).tolist(), l6=L[:3, 3].round(4).tolist(), tcp=tcp.round(4).tolist(), grip=round(float(co.grip(arm)), 4))
        o = objs()[LC.CH]; Tc = LC._T(o['position'], o['quaternion_wxyz']); d['charger_from_tcp_mm'] = round(float(np.linalg.norm(Tc[:3, 3] - tcp)) * 1000, 1)
        d['charger_lost'] = bool(d['charger_from_tcp_mm'] > 80.0)
        gbc = rpc('rgbhint_geom', bbox=[LC.CH, 'socket_0_2'])['bbox']; gs = gbc['socket_0_2']
        d['charger_zmin_above_table_mm'] = round((gbc[LC.CH]['z_min'] - TABLE_Z) * 1000, 1); d['charger_zmin_above_strip_top_mm'] = round((gbc[LC.CH]['z_min'] - gs['z_max']) * 1000, 1)
        d['tcp_above_table_mm'] = round((tcp[2] - TABLE_Z) * 1000, 1); d['strip_top_z'] = round(gs['z_max'], 4)
        if 'strip0' in CTX: d['strip_moved_mm'] = round(float(np.linalg.norm(np.asarray(gs['pose'][:2]) - CTX['strip0'])) * 1000, 1)
        if target is not None:
            _, pe, re, ok = _rm().go(arm, q, target); d['model_from_stall'] = dict(ok=bool(ok), pe_mm=round(pe * 1000, 1), re_deg=round(re, 1))
            d['target_l6'] = np.asarray(target)[:3, 3].round(4).tolist(); d['err_mm'] = round(float(np.linalg.norm(np.asarray(target)[:3, 3] - L[:3, 3])) * 1000, 1)
        cr = _cr()
        if cr is not None:
            box = CTX['RC'].box_from_bbox(gs); S = cr.spheres_world(arm, q)[0]; nm = CTX['sph_names']; per = {}
            loc = (S[:, :3] - box[0][:3, 3]) @ box[0][:3, :3]; qq = np.abs(loc) - box[1]; dbox = np.linalg.norm(np.maximum(qq, 0), axis=-1) + np.minimum(qq.max(-1), 0) - S[:, 3]
            dtab = S[:, 2] - S[:, 3] - cr.table_z(arm)
            for j in range(len(S)):
                if S[j, 3] <= 0: continue
                n = nm[j] if j < len(nm) else '?'; v = per.setdefault(n, [9.9, 9.9]); v[0] = min(v[0], dtab[j]); v[1] = min(v[1], dbox[j])
            d['cu_clear_mm'] = {n: dict(table=round(v[0] * 1000, 1), strip=round(v[1] * 1000, 1)) for n, v in per.items() if min(v) < 0.03}
        else: d['cu_error'] = CTX.get('cu_err')
        other = [x for x in CT.ARMS if x != arm][0]; L2 = co.l6(other)
        sph = dict(tcp_r30=list(tcp) + [0.03], charger_r30=list(map(float, Tc[:3, 3])) + [0.03], l6_r60=list(map(float, L[:3, 3])) + [0.06],
                   other_l6_r60=list(map(float, L2[:3, 3])) + [0.06])   # tcp / charger spheres must hit the charger (positive control)
        c = rpc('c1_contacts', match=MATCH, list=('colliders' not in CTX), spheres=sph); CTX.setdefault('colliders', c.get('colliders'))
        d['spheres'] = c.get('spheres')
        d['contacts'] = {k: v for k, v in (c.get('hits') or {}).items() if v}; d['contact_errors'] = dict(list((c.get('errors') or {}).items())[:3])
    except Exception as e:  # noqa: BLE001
        d['diag_error'] = '%s: %s' % (type(e).__name__, str(e)[:300])
    DIAG.append(d); return d


_line0 = ST._Tool.line


def line_diag(self, arm, pos, quat, max_steps, tol, phase, reason):
    r = _line0(self, arm, pos, quat, max_steps, tol, phase, reason)
    if 'co' in CTX and phase in ('rise', 'travel', 'lower', 'census', 'alt_up', 'alt_down') and not r['derived']['reached']:
        diagnose('%s:%s' % (phase, 'stalled' if r['derived'].get('stalled') else 'budget'), arm, LC._T(pos, quat))
    return r


ST._Tool.line = line_diag
_run0 = RF.run_fast


def run_capture(a, rec, co, tool, ad, rpc, objs, arm, rm, tru, SOCK, end):
    CTX.update(co=co, ad=ad, rpc=rpc, objs=objs, tool=tool); CTX['strip0'] = np.asarray(rpc('rgbhint_geom', bbox=[SOCK])['bbox'][SOCK]['pose'][:2])
    try: return _run0(a, rec, co, tool, ad, rpc, objs, arm, rm, tru, SOCK, end)
    finally:
        rec['census2'] = dict(diag=DIAG, colliders_n=None if CTX.get('colliders') is None else len(CTX['colliders']), out=CTX.get('census'))
        if CTX.get('colliders'): rec['census2']['colliders'] = CTX['colliders']


LC.run = run_capture


def census_recipe(name, tool, co, ad, rpc, objs, arm, Lh, Ls, seat_true, refresh=None, n_refresh=0):
    m0, r0 = ST.STEP_M, ST.STEP_RAD; ST.STEP_M, ST.STEP_RAD = 0.015, 0.08
    try: r1 = tool.line(arm, Lh[:3, 3].tolist(), LC._q_of(Lh), 200, 0.001, 'census', 'census')
    finally: ST.STEP_M, ST.STEP_RAD = m0, r0
    reached = bool(r1['measured']['position_error_m'] <= 0.003 and np.degrees(r1['measured']['rotation_error_rad']) <= 3.0)
    ex = dict(recipe='census2', hover_reached=reached, hover_err_mm=round(r1['measured']['position_error_m'] * 1000, 2), stalled=r1['derived'].get('stalled'), inserted=False)
    out = CTX['census'] = dict(reached=reached, hover_err_mm=ex['hover_err_mm'], arm=arm)
    if reached: return ex
    AT = LC._at(); rm = _rm(); i_ = CT.ARMS.index(arm); q = np.asarray(ad.packet['current_state'][7 * i_:7 * i_ + 6], float)
    o = objs()[LC.CH]; L = co.l6(arm); Gt = np.linalg.inv(L) @ LC._T(o['position'], o['quaternion_wxyz'])
    if np.linalg.norm(Gt[:3, 3] - CT.TCP) > 0.08:
        out['class_hint'] = 'charger_lost'; ex['census'] = out; return ex
    gb = rpc('rgbhint_geom', bbox=['socket_0_2'])['bbox']['socket_0_2']; Ts = LC._T(gb['pose'][:3], gb['pose'][3:]); yaw = float(R.from_matrix(Ts[:3, :3]).as_euler('xyz')[2])
    Fl = np.eye(4); Fl[:3, :3] = R.from_euler('y', 180, degrees=True).as_matrix()
    alts = []; k0 = None
    for k, seat, hov in AT.seated_object_poses(AT.strip_frame(float(Ts[0, 3]), float(Ts[1, 3]), float(gb['z_max']), yaw), 0.015):
        for flip, Gx in (('as_held', Gt), ('flip180', Gt @ Fl)):
            Lhx = hov @ np.linalg.inv(Gx); Lsx = seat @ np.linalg.inv(Gx); _, pe, re, ok = rm.go(arm, q, Lhx); _, pe2, re2, ok2 = rm.go(arm, q, Lsx)
            same = bool(np.linalg.norm(Lhx[:3, 3] - Lh[:3, 3]) < 0.01 and LC._rot_deg(Lhx, Lh) < 5)
            if same: k0 = k
            alts.append(dict(outlet=k, sym=flip, same_as_failed=same, hover_ctrl=dict(ok=bool(ok), pe_mm=round(pe * 1000, 1)), seated_ctrl=dict(ok=bool(ok2), pe_mm=round(pe2 * 1000, 1)),
                             wrist_turn_deg=round(LC._rot_deg(Lhx, L), 1), _M=(Lhx, Lsx, seat)))
    order = sorted(alts, key=lambda x: (not x['same_as_failed'], not (x['hover_ctrl']['ok'] and x['seated_ctrl']['ok']), x['sym'] != 'as_held', x['wrist_turn_deg']))
    tries = []
    for al in order[:3]:
        if tool.terminal(): break
        Lhx, Lsx, seat = al['_M']; up = Lhx.copy(); up[2, 3] += 0.06; t0 = tool.steps
        m0, r0 = ST.STEP_M, ST.STEP_RAD; ST.STEP_M, ST.STEP_RAD = 0.015, 0.08
        try:
            p_ = co.l6(arm)[:3, 3]; tool.line(arm, [float(p_[0]), float(p_[1]), float(max(p_[2], up[2, 3]))], LC._q_of(up), 60, 0.01, 'alt_up', 'census2')
            ru = tool.line(arm, up[:3, 3].tolist(), LC._q_of(up), 60, 0.005, 'alt_up', 'census2')
            rd = tool.line(arm, Lhx[:3, 3].tolist(), LC._q_of(Lhx), 80, 0.001, 'alt_down', 'census2')
        finally: ST.STEP_M, ST.STEP_RAD = m0, r0
        ok = bool(rd['measured']['position_error_m'] <= 0.003 and np.degrees(rd['measured']['rotation_error_rad']) <= 3.0)
        t = dict(outlet=al['outlet'], sym=al['sym'], same_as_failed=al['same_as_failed'], up_err_mm=round(ru['measured']['position_error_m'] * 1000, 1),
                 hover_err_mm=round(rd['measured']['position_error_m'] * 1000, 1), hover_reached=ok, steps=tool.steps - t0)
        if ok and not tool.terminal():
            ei = _ins('R1', tool, co, ad, rpc, objs, arm, Lhx, Lsx, seat, refresh=None, n_refresh=0)
            t.update(inserted=bool(ei.get('inserted')), descent=ei.get('descent'), checks=ei.get('checks'))
            tries.append(t); break
        tries.append(t)
    out.update(alternatives=[{k: v for k, v in x.items() if k != '_M'} for x in alts], tries=tries, failed_outlet=k0)
    ex['census'] = out; ex['inserted'] = any(t.get('inserted') for t in tries); return ex


LC.insert_recipe = census_recipe
print('rg3census2: frozen rg3f chain + contact diagnosis + in-sim alternatives; cfg', RF.CFG, flush=True)
script = argv[0]; sys.argv = [script] + argv[1:]; sys.path.insert(0, str(Path(script).resolve().parent))
runpy.run_path(script, run_name='__main__')
