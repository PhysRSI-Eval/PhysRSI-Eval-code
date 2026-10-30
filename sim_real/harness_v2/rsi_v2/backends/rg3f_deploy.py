"""Truth-free deploy backend of the insert_charger executor (bottleneck line / harness layer, 10-29 12:xxZ).

Source (copied, capability line owns it): macro-tools-20261027/src/rg3fast_lib.py sha256 85cb5958be27a05cb615c1f5975ddc8ff1468d992632dca30ecd2c58311c4659
(md5 0457dff785cdba90988529ae261fa044, frozen per rg3f/CODE-MD5.txt) and loc1cov_lib.insert_recipe (loc1cov_lib.py sha256 b2d2f7e4...).
The control path is unchanged (same acts, same order, same estimators, same thresholds). What changed:
  T1 no simulator-truth read anywhere: rpc('rgbhint_geom'), objs(), truth_outlets, strip trace, truth in-hand errors and the final
     rgbhint_check are removed. The charger CAD box (deploy prior) comes from rsi_v2/cad/charger_bbox.json. Diagnostics that need truth are
     emitted through an optional diag(tag, **info) callback that the OFFLINE driver may implement; online it is None. Nothing returned by
     diag is used for control.
  T2 'inserted' is not decided here (the native evaluator decides online; the offline driver scores with rgbhint_check after the chain).
  T3 step cap: every act first checks steps since the call; reaching the cap aborts (abort-to-Astra: the robot stays where it is, the
     gripper command is not changed, control returns to Astra with the reason). Any refusal or exception also aborts. Background-arm hooks
     and global step sizes are always restored.
  T4 optional lift (online): if the held charger has not just been lifted, lift 4.5 cm first (the same line as the offline driver).
Drives the robot ONLY through tool.act / tool.line of rsi.stage_tools._Tool, i.e. robodojo_act sub-calls of the native adapter (the same
channel, validation and guard as Astra's own turns).
  run(co, tool, ad, arm, rm, cfg, diag=None, lift=False, step_cap=None) -> dict(chain=..., inserted=None, steps_used, chain_end_step, failure)"""
import base64, json, math, urllib.request
import importlib.util as U
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from rsi import combined_tools as CT, stage_tools as ST
import loc1cov_lib as LC   # helpers only: _coarse (SAM3 service), _joints, _q_of, _at (align_target), FIX_VIEWS, OPEN_MIN/CLOSED_MAX/CLEAR_Z, constants

V = '/lizekai/RSIws/experiments/macro-tools-20261027'
CAD = Path(__file__).resolve().parent.parent / 'cad' / 'charger_bbox.json'
DEFAULT = dict(v4_url='http://127.0.0.1:24821', tspeed=(0.049, 0.34), pre_m=0.03, clear_m=0.05, refresh_n=2, insert='R1')


class Abort(Exception):
    pass


def _ld(n, f):
    sp = U.spec_from_file_location(n, f); m = U.module_from_spec(sp); sp.loader.exec_module(m); return m


def _T(p, q): return CT.T_of(p, q)
def _rot_deg(A, B): return float(np.degrees((R.from_matrix(np.asarray(A)[:3, :3]) * R.from_matrix(np.asarray(B)[:3, :3]).inv()).magnitude()))


def _b64(p):
    with open(p, 'rb') as f: return base64.b64encode(f.read()).decode()


def inhand_v4(ad, arm, url):
    pk = ad.packet; im = {i['camera']: i['path'] for i in pk['images']}
    req = dict(images_b64={'cam_high': _b64(im['cam_high']), 'cam_%s_wrist' % arm: _b64(im['cam_%s_wrist' % arm])}, states=pk['current_state'],
               eef_positions=[pk['current_eef'][a]['position'] for a in CT.ARMS], eef_quaternions_wxyz=[pk['current_eef'][a]['quaternion_wxyz'] for a in CT.ARMS],
               arm=arm, category='charger')
    r = json.loads(urllib.request.urlopen(urllib.request.Request(url.rstrip('/') + '/inhand', data=json.dumps(req).encode(),
                                                                 headers={'Content-Type': 'application/json'}), timeout=120).read())
    raw = dict(abstain=r.get('abstain'), flip_flag=r.get('flip_flag'), uncertainty=r.get('uncertainty'), step=pk['step_id'])
    if r.get('abstain', True) or 'position_m' not in r: return None, None, raw
    q = r['quaternion_wxyz']; G = np.eye(4); G[:3, :3] = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); G[:3, 3] = CT.TCP + np.asarray(r['position_m'])
    inh = dict(position_in_tcp_frame_m=list(r['position_m']), quaternion_in_tcp_frame_wxyz=list(q), uncertainty=dict(r.get('uncertainty') or {}))
    return inh, G, raw


class BgArm2:
    """verbatim from rg3fast_lib.BgArm2 (+ restore())."""
    def __init__(self, ad, tool, arm):
        self.ad, self.tool, self.arm, self.segs, self.log, self.snap = ad, tool, arm, [], [], {}
        e = ad.packet['current_eef'][arm]; self.hold_p, self.hold_q = list(map(float, e['position'])), list(map(float, e['quaternion_wxyz']))
        self.open = tool.opening(arm); self._orig = ad._invoke
        def inv(name, args, call_id, on_ack):
            if self.segs and name == 'robodojo_act': args['target'][self.arm] = self.target()
            return self._orig(name, args, call_id, on_ack)
        ad._invoke = inv

    def restore(self): self.ad._invoke = self._orig

    def line(self, pos, quat, tol=0.005, max_acts=24, tag=''):
        self.segs.append(dict(k='line', pos=np.asarray(pos, float), quat=list(quat), tol=tol, max=max_acts, n=0, best=None, stall=0, tag=tag)); return self

    def grip(self, opening, acts=4, tag=''):
        self.segs.append(dict(k='grip', opening=float(opening), max=acts, n=0, tag=tag)); return self

    def target(self):
        e = self.ad.packet['current_eef'][self.arm]; p = np.asarray(e['position'], float); q = list(e['quaternion_wxyz'])
        while self.segs:
            sg = self.segs[0]
            if sg['k'] == 'grip':
                if sg['n'] == 0: self.snap[sg['tag']] = dict(position=p.tolist(), quaternion_wxyz=q, step=int(self.ad.packet['step_id']))
                if sg['n'] >= sg['max']:
                    self.segs.pop(0); self.log.append(dict(tag=sg['tag'], acts=sg['n'])); continue
                sg['n'] += 1; self.open = sg['opening']; self.ad.__dict__.setdefault('_grip_cmd', {})[self.arm] = sg['opening']; break
            d = sg['pos'] - p; dist = float(np.linalg.norm(d)); ang = float((ST.rot(sg['quat']) * ST.rot(q).inv()).magnitude()); er = (dist, ang)
            if sg['best'] is not None and er[0] > sg['best'][0] - 0.001 and er[1] > sg['best'][1] - 0.01: sg['stall'] += 1
            else: sg['stall'] = 0
            sg['best'] = er if sg['best'] is None or er[0] < sg['best'][0] else sg['best']
            if (dist <= sg['tol'] and ang <= np.radians(3)) or sg['n'] >= sg['max'] or sg['stall'] >= 3:
                self.segs.pop(0); self.log.append(dict(tag=sg['tag'], acts=sg['n'], err_mm=round(er[0] * 1000, 1)))
                if er[0] <= sg['tol']: self.hold_p, self.hold_q = list(map(float, sg['pos'])), list(sg['quat'])
                else: self.hold_p = list(map(float, p))
                continue
            sg['n'] += 1; sp = p + (d if dist <= ST.STEP_M else d / dist * ST.STEP_M); fr = 1.0 if ang <= ST.STEP_RAD else ST.STEP_RAD / ang
            qs = ST.wxyz(ST.Slerp([0, 1], R.concatenate([ST.rot(q), ST.rot(sg['quat'])]))([fr])[0]); self.hold_p, self.hold_q = list(map(float, sp)), list(map(float, qs)); break
        if not self.segs: self.ad._invoke = self._orig
        return dict(position=[float(x) for x in self.hold_p], quaternion_wxyz=[float(x) for x in self.hold_q], gripper_closed=bool(self.open < 0.5), gripper_opening=float(self.open))

    def drain(self, tool, fg_arm, reason='bg'):
        while self.segs and not tool.terminal():
            p, q = tool.eef(fg_arm); tool.act(fg_arm, p, q, tool.opening(fg_arm), reason)
        self.ad._invoke = self._orig


def locate_fast(co, tool, ad, rm, hold_arm, budget=120):
    """verbatim rg3fast_lib.locate_fast (no truth read there)."""
    pk = ad.packet; start = tool.steps; free = [x for x in CT.ARMS if x != hold_arm][0]
    loc, src = LC._coarse(co, pk)
    meas = dict(variant='fix2w-fast', coarse=dict(source=src, center_xyz=loc.get('center_xyz'), top_z=loc.get('top_z'), long_axis_deg=loc.get('long_axis_deg')))
    if not loc.get('found') or loc.get('center_xyz') is None:
        co.outlets = None
        return dict(status='failed', tier=None, measured=dict(meas, steps_used=0), derived=dict(found=False, note='strip not found in the head camera')), None
    c = np.asarray(loc['center_xyz'], float); gaze = np.array([c[0], c[1], loc.get('top_z') if loc.get('top_z') is not None else c[2]])
    g = co.grip(free); tcp = co.tcp(free)
    elig = 'empty, open' if g >= LC.OPEN_MIN else ('empty, closed, clear of the strip' if (g <= LC.CLOSED_MAX and tcp[2] - gaze[2] >= LC.CLEAR_Z) else None)
    meas['eligibility'] = elig or 'not used: gripper partly closed (%.2f) or resting %.0f mm above the strip top' % (g, (tcp[2] - gaze[2]) * 1000)
    views_ok = []; pre = {}; qj = LC._joints(ad, free)
    if elig:
        for tag, h, y in LC.FIX_VIEWS:
            p, q = CT.aim(np.asarray(CT.CALIB['mount'][free]), gaze, h, y); _, pe, re, conv = rm.go(free, qj, _T(p, q))
            pre[tag] = round(pe * 1000, 1)
            if conv: views_ok.append((tag, p, q))
    meas['view_precheck_residual_mm'] = pre
    f = None; views = []; moves = []; home = None
    if views_ok:
        home = (free, co.l6(free), np.asarray(qj, float))
        for i, (tag, p, q) in enumerate(views_ok):
            if i > 0 and (tool.steps - start) + 25 + 60 > budget: break
            r = tool.line(free, p, q, 60, 0.01, 'observe', 'locate_outlets'); moves.append(dict(view=tag, **r['measured']))
            if r['measured']['position_error_m'] > 0.02: continue
            img = CT._imgs(ad.packet)['cam_%s_wrist' % free]
            try:
                d = CT._post(co.cfg['infer_url'].rstrip('/') + '/outlets', dict(image_b64=CT._b64(img)), timeout=60)['detections']
            except Exception as e:  # noqa: BLE001
                moves[-1]['detector_error'] = type(e).__name__; continue
            Tc = co.l6(free) @ np.asarray(CT.CALIB['mount'][free]); views.append(dict(tag=tag, K=CT.CALIB['K_wrist'], T=Tc.tolist(), det=d))
            f = CT.fit(views, gaze.tolist(), math.radians(loc.get('long_axis_deg') or 0.0))
            if f is not None and f['rms_px'] <= CT.RMS_UNCERTAIN_PX: break
    meas.update(observer_arm=free if views_ok else None, observation_moves=moves, steps_used=tool.steps - start,
                detections={v['tag']: [[round(x, 1) for x in dd[:2]] + [round(dd[2], 2)] for dd in v['det']] for v in views})
    if f is not None and f['rms_px'] <= CT.RMS_UNCERTAIN_PX:
        co.outlets = f
        return dict(status='ok', tier='wrist_mm', measured=dict(meas, fit_rms_px=round(f['rms_px'], 2)), derived=dict(found=True, fit_rms_px=round(f['rms_px'], 2))), home
    co.outlets = None
    return dict(status='abstain', tier=None, measured=dict(meas, fit_rms_px=None if f is None else round(f['rms_px'], 2)), derived=dict(found=False, note='wrist layer gave no fit')), home


def insert_recipe(name, tool, co, ad, arm, Lh, Ls, refresh=None, n_refresh=0, diag=None):
    """loc1cov_lib.insert_recipe with every truth read removed (the _charger_vs_seat diagnostics go to diag; no rgbhint_check)."""
    dg = diag or (lambda *a_, **k_: None); _q_of = LC._q_of
    def line(M, n, tol): return tool.line(arm, M[:3, 3].tolist(), _q_of(M), n, tol, 'at', 'align')
    m0, r0 = ST.STEP_M, ST.STEP_RAD; ST.STEP_M, ST.STEP_RAD = 0.015, 0.08
    try: r1 = line(Lh, 200, 0.001)
    finally: ST.STEP_M, ST.STEP_RAD = m0, r0
    reached = bool(r1['measured']['position_error_m'] <= 0.003 and math.degrees(r1['measured']['rotation_error_rad']) <= 3.0)
    ex = dict(recipe=name, hover_err_mm=round(r1['measured']['position_error_m'] * 1000, 2), hover_reached=reached); dg('hover', arm=arm)
    for it in range(n_refresh):
        if not reached or tool.terminal() or refresh is None: break
        nt = refresh(arm)
        if nt is None: ex.setdefault('refresh', []).append('abstain'); continue
        Lh, Ls = nt; r1 = line(Lh, 60, 0.001); reached = bool(r1['measured']['position_error_m'] <= 0.003 and math.degrees(r1['measured']['rotation_error_rad']) <= 3.0)
        dg('hover_refresh%d' % it, arm=arm); ex['hover_reached'] = reached
    if not reached or tool.terminal(): return ex
    e0 = co.l6(arm); tcp0 = e0[:3, 3] + e0[:3, :3] @ CT.TCP; z0 = e0[2, 3]; qd = _q_of(Ls); d = np.array([0, 0, -1.0])
    extra = 0.005 if name == 'R0' else 0.004; travel = float(z0 - Ls[2, 3]) + extra; n = int(math.ceil(travel / LC.SPEED_AT)); prog = [0.0]; cpr = [0.0]; outcome = None; fired = None; stall = 0
    for i in range(1, n + LC.SETTLE_AT + 1):
        if tool.terminal(): outcome = 'budget'; break
        s_ = min(i * LC.SPEED_AT, travel); tool.act(arm, [float(Ls[0, 3]), float(Ls[1, 3]), float(z0 - s_)], qd, tool.opening(arm), 'guarded', steps=1)
        L = co.l6(arm); pr = float(((L[:3, 3] + L[:3, :3] @ CT.TCP) - tcp0) @ d); dp = pr - prog[-1]; adv = s_ - cpr[-1]; prog.append(pr); cpr.append(s_)
        if name == 'R0':
            if (i - 1) >= int(LC.DET_AT.get('warm', 2)) and adv > 1e-6 and dp < float(LC.DET_AT['th']) * adv: outcome = 'stopped_by_contact'; fired = i; break
        else:
            stall = stall + 1 if (adv > 1e-6 and dp < 0.3 * adv) else 0
            if stall >= 3: outcome = 'stalled'; fired = i; break
        if i >= n and abs(s_ - pr) <= 0.002: outcome = 'reached'; break
    outcome = outcome or 'stopped'; dg('descent_end', arm=arm)
    pm, qm = tool.eef(arm)
    if name == 'R0':
        if outcome == 'stopped_by_contact':
            for _ in range(LC.UNLOAD_AT):
                if tool.terminal(): break
                tool.act(arm, pm, qm, tool.opening(arm), 'unload', steps=1)
        if not tool.terminal(): tool.set_gripper(arm, opening=1.0, reason='release')
        dg('after_open', arm=arm); p, q = tool.eef(arm)
        if not tool.terminal(): tool.line(arm, list(np.asarray(p) + [0, 0, 0.05]), q, 25, 0.01, 'retreat', 'align')
    else:
        for _ in range(5):
            if tool.terminal(): break
            tool.act(arm, [float(Ls[0, 3]), float(Ls[1, 3]), float(Ls[2, 3] - 0.004)], qd, tool.opening(arm), 'press', steps=1)
        if not tool.terminal(): tool.set_gripper(arm, opening=0.6, reason='partial open')
        for _ in range(5):
            if tool.terminal(): break
            p, q = tool.eef(arm); tool.act(arm, p, q, tool.opening(arm), 'wait', steps=1)
        dg('after_open', arm=arm); p, q = tool.eef(arm)
        for h in range(6):
            if tool.terminal(): break
            tool.act(arm, list(np.asarray(p) + [0, 0, 0.005 * (h + 1)]), q, tool.opening(arm), 'slow rise', steps=1)
        if not tool.terminal(): tool.set_gripper(arm, opening=1.0, reason='release')
        p, q = tool.eef(arm)
        if not tool.terminal(): tool.line(arm, list(np.asarray(p) + [0, 0, 0.03]), q, 20, 0.01, 'retreat', 'align')
    dg('end', arm=arm); ex.update(descent=outcome, fired=fired)
    return ex


def run(co, tool, ad, arm, rm, cfg=None, diag=None, lift=False, step_cap=None):
    C = dict(DEFAULT, **(cfg or {})); dg = diag or (lambda *a_, **k_: None)
    AT = LC._at(); GC = _ld('grasp_compat2', V + '/pkg-v2/rsi/grasp_compat2.py'); RMm = _ld('reach_model_v15', V + '/pkg-v15/rsi/reach_model.py'); rm2 = RMm.Reach()
    LV = np.asarray(json.load(open(CAD))['local_vertices'], float)   # charger CAD bounding box (deploy prior)
    s0 = int(ad.packet['step_id']); ch = dict(mode='deploy', branch='regrasp', cfg=dict(C, tspeed=list(C['tspeed'])), step_cap=step_cap)
    other = [x for x in CT.ARMS if x != arm][0]
    PH = ['start']; book = ch.setdefault('phase_steps', {}); act0 = tool.act; bgs = []; mstep = (ST.STEP_M, ST.STEP_RAD)
    def act(arm_, pos, quat, opening, reason, steps=5):
        if step_cap is not None and int(ad.packet['step_id']) - s0 + int(steps) > step_cap: raise Abort('step_cap')
        b = int(ad.packet['step_id']); r_ = act0(arm_, pos, quat, opening, reason, steps); book[PH[0]] = book.get(PH[0], 0) + int(ad.packet['step_id']) - b; return r_
    tool.act = act
    def phase(n): PH[0] = n
    def fin(fail=None):
        if fail: ch['fail'] = fail
        ch['steps'] = int(ad.packet['step_id']) - s0; ch['end_step'] = int(ad.packet['step_id'])
        return dict(chain=ch, inserted=None, steps_used=ch['steps'], chain_end_step=ch['end_step'], failure=ch.get('fail'), phase_steps=book)
    calls = ch.setdefault('inhand_calls', [])
    def hold_info(ar, tag):
        inh, G, raw = inhand_v4(ad, ar, C['v4_url'])
        calls.append(dict(tag=tag, arm=ar, abstain=raw['abstain'], flip=raw['flip_flag'], spread_deg=(raw['uncertainty'] or {}).get('member_spread_deg'), step=raw['step']))
        dg('inhand_' + tag, arm=ar, G_est=None if G is None else G.tolist()); return inh, G
    try:
        if lift:
            phase('lift'); p, q = tool.eef(arm); tool.line(arm, p + np.array([0, 0, 0.045]), q, 40, 0.003, 'lift', 'insert_charger lift')
        # ---- 1. in-hand pose of the held charger (where it is put down), 0 steps; abstain -> no intervention
        inh0, G0 = hold_info(arm, 'lift')
        if G0 is None: return fin('inhand_abstain')
        # ---- 2. put-down + release + retreat in the background while the free arm moves to the LOC1 view
        phase('locate+putdown'); L = co.l6(arm); Lp = L.copy(); Lp[2, 3] -= 0.044; ep = ad.packet['current_eef'][arm]
        bg = BgArm2(ad, tool, arm); bgs.append(bg); bg.line(Lp[:3, 3], LC._q_of(Lp), tol=0.003, max_acts=8, tag='putdown').grip(1.0, acts=4, tag='release')
        bg.line(np.asarray(ep['position'], float) + [0, 0, 0.016], LC._q_of(Lp), tol=0.01, max_acts=5, tag='retreat')
        t_loc = int(ad.packet['step_id'])
        try:
            lr, home = locate_fast(co, tool, ad, rm, arm)
        except ST.Refused as e_:
            lr, home = dict(status='refused', note=str(e_)[:200]), None
        ch['locate_loc1'] = dict(tier=lr.get('tier'), status=lr.get('status'), steps=int(ad.packet['step_id']) - t_loc, note=(lr.get('derived') or {}).get('note') or lr.get('note'))
        dg('after_locate'); phase('putdown_drain'); bg.drain(tool, other); dg('after_putdown'); ch['bg_putdown'] = bg.log
        co.cmd_closed[arm] = False
        sn = bg.snap.get('release')
        if sn is None: return fin('putdown_not_done')
        Lrel = _T(sn['position'], sn['quaternion_wxyz']); T_ch = Lrel @ G0; dg('placed', T_est=T_ch.tolist())
        if lr.get('status') == 'ok' and co.outlets: dg('outlets', P_est=np.asarray(co.outlets['outlets']).tolist())
        strip = dict(zip(('x', 'y', 'top_z', 'yaw_rad'), map(float, co.outlets['theta']))) if (lr.get('status') == 'ok' and co.outlets) else None
        if strip is None:
            if home is not None: phase('observer_return'); tool.line(home[0], home[1][:3, 3], CT.q_of(home[1]), 60, 0.01, 'return', 'locate_outlets')
            return fin('no_outlets')
        seats = [s for _, s, _ in AT.seated_object_poses(AT.strip_frame(strip['x'], strip['y'], strip['top_z'], strip['yaw_rad']), 0.015)]
        def targets(ar, tag):
            inh, G = hold_info(ar, tag)
            if inh is None: return None, 'inhand_abstain'
            r = AT.align_target(strip, inh, co.l6(ar), hover_m=0.015)
            if r['abstain']: return None, 'align_abstain: ' + r['note'][:80]
            return r['value']['targets'], G
        # ---- 3. compatible re-grasp
        def joints(ar): i_ = CT.ARMS.index(ar); return np.asarray(ad.packet['current_state'][7 * i_:7 * i_ + 6], float)
        ns = dict(targets=seats, T_obj_inv=np.linalg.inv(T_ch)); table_z = float(rm2.root['left'][2, 3]); cands = []
        for ar in CT.ARMS:
            i_ = CT.ARMS.index(ar); gj = float(ad.packet['current_state'][7 * i_ + 6]); qa = home[2] if (home is not None and ar == home[0]) else joints(ar)
            o2 = GC.grasp_compatible(ar, qa, gj, LV, T_ch, rm2, None, None, table_z, ns, last_command=(1.0 if ar == arm else None))
            for r_ in ((o2.get('value') or {}).get('_rows') or []):
                if r_['reachable'] and r_.get('next_stage_reachable'): cands.append((ar, r_))
        ch['n_compatible'] = len(cands)
        if not cands:
            if home is not None: phase('observer_return'); tool.line(home[0], home[1][:3, 3], CT.q_of(home[1]), 60, 0.01, 'return', 'locate_outlets')
            return fin('no_compatible_grasp')
        ar2, row = min(cands, key=lambda c: (c[1].get('depth') != 'centred', round((c[1].get('next_stage_turn_deg') or 999) / 15.0), c[1].get('model_ticks') or 999))
        i2 = CT.ARMS.index(ar2); bg2 = None
        if home is not None:
            if ar2 == home[0]:
                phase('observer_return'); tool.line(home[0], home[1][:3, 3], CT.q_of(home[1]), 60, 0.01, 'return', 'locate_outlets')
            else:
                bg2 = BgArm2(ad, tool, home[0]).line(home[1][:3, 3], CT.q_of(home[1]), tol=0.01, max_acts=12, tag='observer_return'); bgs.append(bg2)
        phase('regrasp'); ex = GC.execute(tool, ar2, row, ST, lambda: float(ad.packet['current_state'][7 * i2 + 6]), step_cap=150, q_of=LC._q_of)
        if bg2 is not None: phase('observer_return_drain'); bg2.drain(tool, ar2); ch['bg_return'] = bg2.log
        dg('after_regrasp')
        ch['grasp'] = dict(arm=ar2, family=row['family'], depth=row.get('depth'), turn_to_seat_deg=row.get('next_stage_turn_deg'), status=ex.get('status'), held=ex.get('held'), steps=ex.get('steps_used'))
        if ex.get('held') != 'yes': return fin('regrasp_not_held')
        co.cmd_closed[ar2] = True; use_arm = ar2
        tg, G = targets(ar2, 'after_regrasp')
        if tg is None: return fin('after_regrasp_' + str(G))
        best = min(tg, key=lambda x: (x['wrist_turn_from_now_deg'], x['hand_travel_m'])); k = best['outlet']
        Lh = _T(best['hover_link6']['position'], best['hover_link6']['quaternion_wxyz']); Ls = _T(best['seated_link6']['position'], best['seated_link6']['quaternion_wxyz'])
        L_before = co.l6(use_arm); ch['transport_plan'] = dict(wrist_turn_deg=round(_rot_deg(Lh, L_before), 1), travel_mm=round(float(np.linalg.norm(Lh[:3, 3] - L_before[:3, 3])) * 1000, 1))
        def refresh(ar, tag='hover_refresh'):
            tg2, _ = targets(ar, tag)
            if tg2 is None: return None
            b2 = [x for x in tg2 if x['outlet'] == k][0]
            return _T(b2['hover_link6']['position'], b2['hover_link6']['quaternion_wxyz']), _T(b2['seated_link6']['position'], b2['seated_link6']['quaternion_wxyz'])
        # ---- 4. transport through a safe height to the pre-hover, re-measure + re-aim, hover / refresh / R1
        phase('transport'); Lpre = Lh.copy(); Lpre[2, 3] += C['pre_m']; ST.STEP_M, ST.STEP_RAD = C['tspeed']
        z_tr = max(float(co.l6(use_arm)[2, 3]), float(Lpre[2, 3]) + C['clear_m']); p_ = co.l6(use_arm)[:3, 3]; qd = LC._q_of(Lpre)
        try:
            tool.line(use_arm, [float(p_[0]), float(p_[1]), z_tr], qd, 60, 0.01, 'rise', 'rg3f')
            tool.line(use_arm, [float(Lpre[0, 3]), float(Lpre[1, 3]), z_tr], qd, 100, 0.01, 'travel', 'rg3f')
            rt = tool.line(use_arm, Lpre[:3, 3].tolist(), qd, 60, 0.003, 'lower', 'rg3f')
        finally: ST.STEP_M, ST.STEP_RAD = mstep
        ch['transport'] = dict(err_mm=round(rt['measured']['position_error_m'] * 1000, 2)); dg('prehover')
        nt = refresh(use_arm, 'prehover'); ch['prehover_reaim'] = nt is not None
        if nt is not None: Lh, Ls = nt
        phase('insert'); exi = insert_recipe(C['insert'], tool, co, ad, use_arm, Lh, Ls, refresh=lambda ar: refresh(ar, 'hover_refresh'), n_refresh=C['refresh_n'], diag=dg)
        ch.update(outlet=k, arm=use_arm, insert=exi)
        return fin()
    except Abort as e:
        return fin('abort_to_astra: %s' % e)
    except ST.Refused as e:
        return fin('abort_to_astra: refused %s' % str(e)[:120])
    except Exception as e:  # noqa: BLE001  (online: never raise into the episode; the reason is reported to Astra)
        return fin('abort_to_astra: %s %s' % (type(e).__name__, str(e)[:120]))
    finally:
        for b in bgs: b.restore()
        tool.act = act0; ST.STEP_M, ST.STEP_RAD = mstep
