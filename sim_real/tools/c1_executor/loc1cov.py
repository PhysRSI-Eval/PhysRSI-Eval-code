"""loc1cov.py (capability line, 2026-10-28 12:xxZ): reach_m3.py with the LOC1 block replaced by loc1cov_lib.run (diagnosis + view variant); stops after LOC1."""
"""reach_m3.py (capability line, PREREG-reach-M3, 2026-10-28) = m3_run.py + --outlets loc1 (deploy LOC1 locate_outlets after the lift) + collision-aware
check_reach v2 prediction for the chosen hover pose, then the hover line as the label; stops after the hover (no descent).
"""
"""m3_run.py = bottleneck bn_run.py (read-only source) + metadata for non-v14a Astra plug episodes (capability line M3, 2026-10-28).
"""
"""bottleneck track B (2026-10-27, zero quota, offline): HANDOVER FROM ASTRA'S REAL STATES (v1.4 online episodes v14a).
Reset the episode's layout, replay Astra's recorded native actions up to the handover step (default: auto = first step at which an arm
holds the charger: grip in (0.02, 0.9), charger within 6 cm of that TCP, charger still within 1 cm of its reset height), then the scripted
executor runs the plug pipeline of plug-retest rt_run.py from that state (lift 4.5 cm -> post-lift report -> targets -> hover -> guarded
descent -> release -> retreat -> home). Astra's own grasp is kept; the executor does not regrasp.
  --outlets online : the outlet estimate Astra's locate_outlets returned online in this episode (deploy-valid; TREAT episodes only)
  --outlets oracle : simulator truth outlets (upper bound for a zero-step outlet localiser, e.g. head camera before the grasp)
  --cond est_lift / oracle_lift : in-hand pose measured once at the lift (= v1.4 online advice, executed precisely)        [variant i]
  --cond est / oracle           : + re-estimate at the hover and re-aim (closed-loop macro of hover+insert)              [variant ii]
  --mode descend --t T : no lift/hover; from step T continue Astra's own descent toward the ONLINE seated target (guarded) then release
Truth is read only for scoring (as in rt_run). Original rt_run docstring follows.
plug-retest (2026-10-27, dev, zero quota): copy of plug-offline-eval/v14/src/v14_run.py with
  * --cond oracle_lift: truth outlets + TRUTH in-hand pose measured ONCE right after the lift; the hover "refresh" keeps that pose
    (same hover2 settle move to the unchanged target, so step structure matches 'oracle').
  * --cond oracle (= hover re-measure): truth in-hand pose after the lift AND re-measured (truth) at the hover, --refresh times.
  * --cond est: service in-hand estimate after the lift, re-estimated at the hover --refresh times (service via --infer-url).
  * truth in-hand pose trace at grasp / lift / hover arrival / after refresh / descent end (drift vs lift).
  * paths: v14/src imported read-only; STARTS / kin.json read from plug-offline-eval.
Original docstring follows.
plug-offline-eval v1.4 dev feasibility driver (zero quota; NOT a formal evaluation). Native reset of a layout, then:
  locate_outlets FIRST (both hands free -> observer = the arm on the strip side, as the frozen tool picks)
  -> grasp rule (truth-aimed scripted grasp, dev): 'direct' = charger-side arm grasps with the canonical Astra pinch rotated about the
     charger width axis by theta and about the vertical by psi (pivot = grasp point; v14/src/scan.py cand_H tcp);
     'reloc' = charger-side arm picks it with the canonical grasp, puts it down at x = x_mid (toward the strip), opens, retreats; then
     the strip-side arm grasps it with the theta/psi rule
  -> lift 4.5 cm (automatic post-lift report with targets) -> insertion reach pre-check prototype (reach.precheck: estimated in-hand pose,
     estimated outlets, measured joints, robot constants) -> the same hover / 2 refreshes / guarded descent / hold 3 / release / home
     as plug-offline-v1 (pe_run.py).
--cond est | oracle as in pe_run.py. --extend N raises the native step limit by N (budget-free feasibility only; success is then also
reported against the 400-step limit).
  python v14_run.py --layout L --rule direct|reloc --rep k --cond est|oracle --gpu G --out DIR"""
import argparse, json, math, sys, time, traceback
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
HERE = Path('/lizekai/RSIws/experiments/combined-online-20261026/bottleneck/src')
DEC = json.JSONDecoder()
V14 = Path('/lizekai/RSIws/experiments/combined-online-20261026/plug-offline-eval/v14/src')
sys.path.insert(0, str(V14)); sys.path.insert(0, '/root/pe-1026/pkg')
import v14_sim as pe_sim  # noqa: E402
import reach as RM, scan as SC  # noqa: E402
from rsi import combined_tools as CT, stage_tools as ST  # noqa: E402
RA = pe_sim.RA
import os as _os
_po2 = RA.subprocess.Popen
def _popen_v14(argv, **kw):  # host1: /root/pe-1026/code2 is the shared /lizekai copy without pe_server_ext.py -> add v14/src for the server only
    if kw.get('env') is not None: kw['env'] = dict(kw['env'], PYTHONPATH=str(V14) + _os.pathsep + kw['env'].get('PYTHONPATH', ''))
    return _po2(argv, **kw)
RA.subprocess.Popen = _popen_v14
CFG = dict(infer_url='http://127.0.0.1:24792', locate_url='http://10.255.244.66:26480/locate')
sys.path.insert(0, '/lizekai/RSIws/experiments/macro-tools-20261027/pkg-v2'); sys.path.insert(0, '/lizekai/RSIws/experiments/macro-tools-20261027/src')
import importlib.util as _ilu

CHECKS = [['is_A_depth_in_B', dict(label_A='charger', label_B='socket', z_threshold=0.015)], ['is_A_in_B', dict(label_A='charger', label_B='socket')],
          ['is_axis_up', dict(label='charger', axis=[0, 1, 0], threshold=10)]]
DET = json.load(open('/lizekai/RSIws/experiments/innerloop-20261026/track1-contact/src/detector_frozen.json'))
SPEED = 0.002; SETTLE = 6; UNLOAD = 3; REFRESH = 2; HOVER_OK = (0.003, 3.0)
CH, SOCK = 'charger_0_1', 'socket_0_2'
STARTS = json.load(open(V14.parent.parent / 'starts.json')); KIN = json.load(open(V14.parent / 'out' / 'kin.json'))


def Tq(p, q): return CT.T_of(p, q)


def rot_err(A, B): return float(np.degrees((R.from_matrix(A[:3, :3]) * R.from_matrix(B[:3, :3]).inv()).magnitude()))


def tcp_of(L): return L[:3, 3] + L[:3, :3] @ CT.TCP


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--episode', required=True); ap.add_argument('--t', type=int, default=-1); ap.add_argument('--outlets', choices=['online', 'oracle', 'loc1'], default='online'); ap.add_argument('--mode', choices=['tools', 'descend'], default='tools'); ap.add_argument('--rule', choices=['direct', 'reloc'], default='direct')
    ap.add_argument('--theta', type=float, default=-30.0); ap.add_argument('--psi', type=float, default=0.0); ap.add_argument('--x-mid', type=float, default=0.05)
    ap.add_argument('--rep', type=int, default=0); ap.add_argument('--cond', choices=['est', 'est_lift', 'oracle', 'oracle_lift'], required=True); ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--extend', type=int, default=0)
    ap.add_argument('--refresh', type=int, default=1); ap.add_argument('--infer-url', default='http://127.0.0.1:24796'); ap.add_argument('--tag', default='rt'); ap.add_argument('--variant', default='base'); ap.add_argument('--exec-mode', dest='exec_mode', default=None); ap.add_argument('--insert', default='R0'); ap.add_argument('--chain', default=None); ap.add_argument('--force-regrasp', dest='force_regrasp', action='store_true'); ap.add_argument('--refresh-n', dest='refresh_n', type=int, default=0); ap.add_argument('--diag-transport', dest='diag_transport', action='store_true'); ap.add_argument('--gate', type=float, default=None); a = ap.parse_args()
    CFG['infer_url'] = a.infer_url; ORC = a.cond in ('oracle', 'oracle_lift')
    if a.gate is not None: CT.SPREAD_UNCERTAIN_DEG = float(a.gate)  # in-memory only: v2-charger gate tau (FROZEN-v2charger.json), frozen harness file untouched
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    M_ = json.load(open(HERE / 'episodes_v14a.json')); [M_.setdefault(k_, v_) for k_, v_ in json.load(open('/lizekai/RSIws/experiments/macro-tools-20261027/run/m3_meta_extra.json')).items()]; [M_.setdefault(k_, v_) for k_, v_ in (json.load(open('/lizekai/RSIws/experiments/macro-tools-20261027/loc1cov/meta_r4.json')).items() if Path('/lizekai/RSIws/experiments/macro-tools-20261027/loc1cov/meta_r4.json').exists() else [])]  # fix: never overwrite v14a entries (online outlets)  # m3: + non-v14a episodes (layout from A labels; no online outlets)
    meta = M_[a.episode]; layout = int(meta['layout'])
    st = dict(kind='handover', id='%s-%s-%s-%s-%s' % (a.tag, a.episode[-8:], a.mode, a.outlets, a.cond), layout=layout, episode=a.episode, branch=meta['branch'])
    rec = dict(start=st, cond=a.cond, outlets_src=a.outlets, mode=a.mode, refresh=a.refresh, gate_deg=CT.SPREAD_UNCERTAIN_DEG, infer_url=a.infer_url, trace=[],
               extend=a.extend, detector=DET, stage_steps={}, failure=None); t0 = time.time(); sim = None
    CT.FLIPS = (0.0,); SC.PIVOT = 'tcp'; kin = KIN.get(str(layout), KIN['0'])['kin']; rm = RM.Reach(kin)  # robot constants identical across layouts (kin.json 0,1,7,14,15)
    try:
        sim = RA.Sim('plug_in_charger', Path('/root/pe-1026/sim') / ('LCV-%s-%s%s%s-%s' % (a.variant, a.exec_mode or 'x', a.insert, a.chain or '', st['id'])), a.gpu)  # 10-29 fix: scratch name unique per job (same-episode jobs shared it)  # server ext resolves next to /root/pe-1026; local scratch removed in finally
        rpc = sim.rpc.request
        def objs(): return rpc('objects')
        reset = rpc('reset', layout_id=layout); o0 = rpc('observe')
        ad0 = pe_sim.FakeAdapter(sim, out / 'obs_start'); START_IMGS = {i['camera']: i['path'] for i in ad0.packet['images']}  # deploy harness keeps the episode-start images (LOC1 coarse fallback)
        if a.extend: rec['step_lim'] = rpc('extend_limit', extra=int(a.extend))
        home = {arm: (list(map(float, o0['eef_positions'][i])), list(map(float, o0['eef_quaternions_wxyz'][i]))) for i, arm in enumerate(CT.ARMS)}
        ch0 = objs()[CH]; rec['charger_reset'] = ch0; sock = objs()[SOCK]
        # ---------------- replay Astra's recorded actions to the handover step
        task, lay, actions = RA.recorded(a.episode); assert lay == layout, (lay, layout)
        worst = 0.0; T = None; k = 0; held_arm = None
        while k < len(actions):
            chunk = actions[k:k + 5]
            if a.t >= 0: chunk = [c for c in chunk if c['step'] < a.t]
            if not chunk: break
            for r_, got in zip(chunk, rpc('replay', actions=[c['action'] for c in chunk])):
                if r_['states']: worst = max(worst, max(abs(x - y) for x, y in zip(got['states'], r_['states'])))
                if got['ended']: raise RuntimeError('START_INVALID episode ended during replay')
            k += len(chunk)
            if worst > RA.TOL: raise RuntimeError('START_INVALID replay diverged (err %.4f) by step %d' % (worst, k))
            if a.t < 0:
                ob = rpc('observe'); c_ = objs()[CH]
                for i_, arm_ in enumerate(CT.ARMS):
                    L_ = Tq(ob['eef_positions'][i_], ob['eef_quaternions_wxyz'][i_]); tcp_ = L_[:3, 3] + L_[:3, :3] @ CT.TCP; g_ = float(ob['states'][7 * i_ + 6])
                    if 0.02 < g_ < 0.9 and np.linalg.norm(np.asarray(c_['position']) - tcp_) < 0.06 and c_['position'][2] - ch0['position'][2] < 0.01:
                        held_arm = arm_
                if held_arm: T = k; break
        if a.t < 0 and T is None: raise RuntimeError('START_INVALID Astra never held the charger on the table in this episode')
        rec['restore'] = dict(step=k, max_state_error=worst, auto=a.t < 0)
        ad = pe_sim.FakeAdapter(sim, out / 'obs'); co = CT.state(ad, CFG); tool = ST._Tool(ad, 'pe', 'pe', None, CFG)
        def after(tag):
            items = CT.after_act(ad, CFG, None)
            for it in items: rec.setdefault('events', []).append(dict(tag=tag, step=ad.packet['step_id'], text=it['text'][:6000]))
            return items
        def end(kind, **kw):
            rec['failure'] = rec['failure'] or kind; rec.update(kw)
        if held_arm is None:  # explicit --t: the arm whose TCP is nearest the charger
            c_ = objs()[CH]; held_arm = min(CT.ARMS, key=lambda x: float(np.linalg.norm(np.asarray(c_['position']) - co.tcp(x))))
        arm = held_arm; rec['arm'] = arm
        for ar in CT.ARMS: co.cmd_closed[ar] = ar == arm
        co.start_imgs = START_IMGS
        after('start')
        # ---------------- outlets: Astra's online locate_outlets result, or truth
        gb = rpc('rgbhint_geom', bbox=[SOCK])['bbox'][SOCK]; Ts = Tq(gb['pose'][:3], gb['pose'][3:])
        tru = [(Ts[:3, :3] @ np.array([x, CT.OUT_Y, 0.0]) + Ts[:3, 3]).tolist() for x in CT.OFFS]
        for t_ in tru: t_[2] = gb['z_max']
        yaw = R.from_quat([gb['pose'][4], gb['pose'][5], gb['pose'][6], gb['pose'][3]]).as_euler('xyz')[2]; th_true = [gb['pose'][0], gb['pose'][1], gb['z_max'], yaw]
        rec['truth_outlets'] = tru; on = meta.get('online_outlets')
        if on:
            rec['online_outlet_err_mm'] = [round(float(np.linalg.norm(np.array(p_) - np.array(t_))) * 1000, 2) for p_, t_ in zip(on['outlets_xyz_m'], tru)]
        if a.outlets == 'oracle':
            co.outlets = dict(theta=th_true, outlets=CT.outlets_of(th_true).tolist(), dev_truth=True)
        elif a.outlets == 'loc1':
            co.outlets = None
        elif on:
            co.outlets = dict(theta=on['theta'], outlets=on['outlets_xyz_m'])
        else:
            end('no_online_outlets'); raise StopIteration
        if a.outlets != 'loc1': rec['locate'] = dict(src=a.outlets, outlet_err_mm=[round(float(np.linalg.norm(np.array(p_) - np.array(t_))) * 1000, 2) for p_, t_ in zip(co.outlets['outlets'], tru)])
        def truth_G():
            L = co.l6(arm); o = objs()[CH]; return np.linalg.inv(L) @ Tq(o['position'], o['quaternion_wxyz'])
        def tr(tag, Gref=None):
            Gt = truth_G(); d_ = dict(tag=tag, step=ad.packet['step_id'], G=Gt.round(5).tolist())
            if Gref is not None: d_.update(vs_lift_mm=round(float(np.linalg.norm(Gt[:3, 3] - Gref[:3, 3])) * 1000, 2), vs_lift_deg=round(rot_err(Gt, Gref), 2))
            rec['trace'].append(d_); return Gt
        Gs = truth_G(); tr('handover'); dist = float(np.linalg.norm(Gs[:3, 3] - CT.TCP)); rise = objs()[CH]['position'][2] - ch0['position'][2]
        rec['start_check'] = dict(step=ad.packet['step_id'], charger_to_tcp_m=round(dist, 4), charger_rise_m=round(rise, 4), grip=co.grip(arm))
        if not (dist < 0.06 and 0.02 < co.grip(arm) < 0.9):
            end('not_holding_at_handover'); raise StopIteration
        if a.mode == 'descend':
            sd = meta['online_seated_link6']; e0 = co.l6(arm); l6_0 = e0[:3, 3].copy(); tcp0 = tcp_of(e0); qd = sd['quaternion_wxyz']; d = np.array([0, 0, -1.0])
            max_travel = float(l6_0[2] - sd['position'][2]) + 0.005; n_move = int(math.ceil(max(max_travel, 0) / SPEED)); prog = [0.0]; cpr = [0.0]; fired = None; outcome = None
            for k in range(1, n_move + SETTLE + 1):
                if tool.terminal(): outcome = 'budget'; break
                s = min(k * SPEED, max_travel); tgt = np.array([sd['position'][0], sd['position'][1], l6_0[2] - s])
                tool.act(arm, tgt, qd, tool.opening(arm), 'guarded', steps=1)
                pr = float((tcp_of(co.l6(arm)) - tcp0) @ d); dprog = pr - prog[-1]; adv = s - cpr[-1]; prog.append(pr); cpr.append(s)
                if (k - 1) >= int(DET.get('warm', 2)) and adv > 1e-6 and dprog < float(DET['th']) * adv: fired = k; outcome = 'stopped_by_contact'; break
                if k >= n_move and abs(s - pr) <= 0.002: outcome = 'reached'; break
            if outcome is None: outcome = 'stopped_by_contact'; fired = k
            o = objs()[CH]; To = Tq(o['position'], o['quaternion_wxyz'])
            Tst = np.eye(4); Tst[:3, :3] = R.from_euler('z', th_true[3]).as_matrix(); Tst[:3, 3] = [th_true[0], th_true[1], th_true[2] - CT.CALIB['socket_top_minus_origin_m']]
            seats = []
            for ox in CT.OFFS:
                S_ = np.eye(4); S_[:3, :3] = CT.SEAT_R; S_[:3, 3] = [ox, CT.OUT_Y, CT.SEAT_P]; seats.append(Tst @ S_)
            Ttrue_seat = min(seats, key=lambda S2: float(np.linalg.norm(S2[:2, 3] - To[:2, 3])))
            rec['descent'] = dict(outcome=outcome, ticks=len(prog) - 1, fired_tick=fired, link6_above_seated_mm=round((co.l6(arm)[2, 3] - sd['position'][2]) * 1000, 1),
                                  charger_above_true_seat_mm=round(float(To[2, 3] - Ttrue_seat[2, 3]) * 1000, 1), charger_vs_true_seat_xy_mm=((To[:2, 3] - Ttrue_seat[:2, 3]) * 1000).round(2).tolist(),
                                  charger_vs_true_seat_rot_deg=round(rot_err(To, Ttrue_seat), 2))
            if outcome == 'stopped_by_contact':
                pm, qm = tool.eef(arm)
                for _ in range(UNLOAD):
                    if tool.terminal(): break
                    tool.act(arm, pm, qm, tool.opening(arm), 'unload', steps=1)
            if tool.terminal(): end('budget_in_descent'); raise StopIteration
            tool.set_gripper(arm, opening=1.0, reason='release'); p, q = tool.eef(arm)
            if not tool.terminal(): tool.line(arm, p + np.array([0, 0, 0.05]), q, 25, 0.01, 'retreat', 'pe')
            rec['insertion_checks'] = RA.plain(rpc('rgbhint_check', checks=CHECKS))
            for a_ in ([arm, [x for x in CT.ARMS if x != arm][0]]):
                if tool.terminal(): break
                tool.line(a_, home[a_][0], home[a_][1], 80, 0.005, 'home', 'pe')
            raise StopIteration
        txt_ok = True
        after('grasped')  # registers the hold before the lift (as an Astra close + lift would)
        if rm is not None:
            qn0 = np.asarray(ad.packet['current_state'][7 * CT.ARMS.index(arm): 7 * CT.ARMS.index(arm) + 6], float); Pm = rm.pose(arm, qn0)
            rec['fk_check_mm'] = round(float(np.linalg.norm(Pm[:3, 3] - co.l6(arm)[:3, 3])) * 1000, 2)
        def G_err(est, tag):
            Gt = truth_G()
            if not est or est.get('status') != 'ok' or '_G' not in est:
                return dict(tag=tag, step=ad.packet['step_id'], status=(est or {}).get('status'), member_spread_deg=(est or {}).get('member_spread_deg'))
            Ge = np.asarray(est['_G']); return dict(tag=tag, step=ad.packet['step_id'], status='ok', pos_err_mm=round(float(np.linalg.norm(Ge[:3, 3] - Gt[:3, 3])) * 1000, 2),
                                                    rot_err_deg=round(rot_err(Ge, Gt), 2), member_spread_deg=est.get('member_spread_deg'))
        def oracle_est():
            return dict(status='ok', _G=truth_G().tolist(), oracle=True)
        p, q = tool.eef(arm); tool.line(arm, p + np.array([0, 0, 0.045]), q, 40, 0.003, 'lift', 'pe lift'); after('lift')
        h = co.hold[arm]; est = (h or {}).get('est'); rec['inhand_lift'] = G_err(est, 'lift'); rec['stage_steps']['lift'] = ad.packet['step_id']; G_lift = tr('lift')
        if a.outlets == 'loc1':  # loc1cov: diagnose LOC1 under Astra holds (truth only for diagnosis/scoring) + variant
            import loc1cov_lib as LC
            LC.run(a, rec, co, tool, ad, rpc, objs, arm, rm, tru, SOCK, end)
            end('probe_done'); raise StopIteration
        # ---------------- plug targets (current in-hand estimate; one re-estimate if none)
        h = co.hold[arm]
        if h is None:
            end('not_holding_after_locate'); raise StopIteration
        if ORC: h['est'] = dict(status='ok', _G=G_lift.tolist(), oracle=True)
        tg = co.plug_targets(arm, h['est']) if h.get('est') else None
        if not tg and a.cond in ('est', 'est_lift'):
            rr = co.refresh(arm); rec['inhand_retry'] = G_err(co.hold[arm].get('est'), 'retry'); tg = rr['derived']['plug_targets'] if isinstance(rr['derived']['plug_targets'], list) else None
        if not tg:
            end('no_targets_inhand_' + str((h.get('est') or {}).get('status'))); raise StopIteration
        best = min(tg, key=lambda x: (x['wrist_turn_from_now_deg'], x['hand_travel_m'])); rec['chosen'] = {k: best[k] for k in ('outlet', 'wrist_turn_from_now_deg', 'hand_travel_m')}
        rec['chosen']['outlet_err_mm'] = rec['locate']['outlet_err_mm'][best['outlet']]
        hv, sd = best['hover_link6'], best['seated_link6']
        try:  # insertion reach pre-check prototype (deploy inputs only: estimated G, estimated (or oracle) outlets, measured joints, robot constants)
            x_, y_, zt_, yw_ = co.outlets['theta']; Ts_ = np.eye(4); Ts_[:3, :3] = R.from_euler('z', yw_).as_matrix(); Ts_[:3, 3] = [x_, y_, zt_ - CT.CALIB['socket_top_minus_origin_m']]
            seats_ = []
            for ox in CT.OFFS:
                S2 = np.eye(4); S2[:3, :3] = CT.SEAT_R; S2[:3, 3] = [ox, CT.OUT_Y, CT.SEAT_P]; seats_.append(Ts_ @ S2)
            qn = np.asarray(ad.packet['current_state'][7 * CT.ARMS.index(arm): 7 * CT.ARMS.index(arm) + 6], float)
            rec['precheck'] = RM.precheck(rm, arm, qn, co.l6(arm), np.asarray(h['est']['_G']), seats_) if rm is not None else dict(error='no reach model for this layout')
        except Exception as e_:  # noqa: BLE001
            rec['precheck'] = dict(error='%s: %s' % (type(e_).__name__, str(e_)[:200]))
        # PREREG-reach-M3: record the deploy inputs of the prediction (computed by reach_m3_predict.py with check_reach v2; the prediction never
        # feeds back into this run, so computing it after the hover line is equivalent)
        qn = np.asarray(ad.packet['current_state'][7 * CT.ARMS.index(arm): 7 * CT.ARMS.index(arm) + 6], float); g_ = float(ad.packet['current_state'][7 * CT.ARMS.index(arm) + 6])
        rec['reach_inputs'] = dict(arm=arm, q_now=qn.tolist(), gripper_joint=g_, chosen_outlet=best['outlet'], hover_link6={str(x['outlet']): x['hover_link6'] for x in tg},
                                   seated_link6={str(x['outlet']): x['seated_link6'] for x in tg}, step=ad.packet['step_id'])
        r1 = tool.line(arm, hv['position'], hv['quaternion_wxyz'], 120, 0.001, 'hover', 'pe'); rec['hover_first'] = r1['measured']; after('hover'); tr('hover_arrival', G_lift)
        rec['label_hover_reached'] = bool(r1['measured']['position_error_m'] <= HOVER_OK[0] and math.degrees(r1['measured']['rotation_error_rad']) <= HOVER_OK[1])
        end('probe_done'); raise StopIteration
        rec['stage_steps']['hover'] = ad.packet['step_id']
        for it in range(a.refresh):
            if tool.terminal(): break
            if a.cond == 'est_lift':  # lift estimate kept (no new estimate used); diagnostic-only estimate at the hover (0 steps, not used)
                rec.setdefault('inhand_hover_diag', []).append(G_err(co.inhand(arm), 'hoverdiag%d' % it))
                tg2 = co.plug_targets(arm, co.hold[arm]['est']) if co.hold[arm].get('est') else None
                rec.setdefault('inhand_hover', []).append(G_err(co.hold[arm].get('est'), 'hover%d' % it))
            elif a.cond == 'oracle_lift':  # stale lift pose kept: same settle move to the unchanged target
                co.hold[arm]['est'] = dict(status='ok', _G=G_lift.tolist(), oracle=True); tg2 = co.plug_targets(arm, co.hold[arm]['est'])
                rec.setdefault('inhand_hover', []).append(G_err(co.hold[arm]['est'], 'hover%d' % it))
            elif a.cond == 'oracle':
                co.hold[arm]['est'] = oracle_est(); tg2 = co.plug_targets(arm, co.hold[arm]['est']); rec.setdefault('inhand_hover', []).append(G_err(co.hold[arm]['est'], 'hover%d' % it))
            else:
                rr = co.refresh(arm); rec.setdefault('inhand_hover', []).append(G_err(co.hold[arm].get('est') if rr['status'] == 'ok' else None, 'hover%d' % it))
                tg2 = rr['derived']['plug_targets'] if isinstance(rr['derived']['plug_targets'], list) else None
                if rr['status'] != 'ok': rec['inhand_hover'][-1]['status'] = rr['status']
            if not tg2: continue  # keep the previous target (an 'uncertain' re-estimate gives no targets)
            b2 = [x for x in tg2 if x['outlet'] == best['outlet']][0]; hv, sd = b2['hover_link6'], b2['seated_link6']
            tool.line(arm, hv['position'], hv['quaternion_wxyz'], 30, 0.001, 'hover2', 'pe')
        rec['stage_steps']['refresh'] = ad.packet['step_id']; tr('after_refresh', G_lift)
        Gcur = np.asarray(co.hold[arm]['est']['_G']); L = co.l6(arm); Th = Tq(hv['position'], hv['quaternion_wxyz'])
        o = objs()[CH]; To = Tq(o['position'], o['quaternion_wxyz']); Tw_h = Th @ Gcur
        # truth target: the true seated charger pose in the chosen outlet (strip truth) lifted by the hover height
        Tst = np.eye(4); Tst[:3, :3] = R.from_euler('z', th_true[3]).as_matrix(); Tst[:3, 3] = [th_true[0], th_true[1], th_true[2] - CT.CALIB['socket_top_minus_origin_m']]
        S_ = np.eye(4); S_[:3, :3] = CT.SEAT_R; S_[:3, 3] = [CT.OFFS[best['outlet']], CT.OUT_Y, CT.SEAT_P]; Ttrue_seat = Tst @ S_; Ttrue_hover = Ttrue_seat.copy(); Ttrue_hover[2, 3] += CT.HOVER_M
        rec['at_hover'] = dict(link6_pos_err_mm=round(float(np.linalg.norm(L[:3, 3] - Th[:3, 3])) * 1000, 2), link6_rot_err_deg=round(rot_err(L, Th), 2),
                               charger_vs_true_hover_xy_mm=((To[:2, 3] - Ttrue_hover[:2, 3]) * 1000).round(2).tolist(), charger_vs_true_hover_z_mm=round(float(To[2, 3] - Ttrue_hover[2, 3]) * 1000, 2),
                               charger_vs_true_hover_rot_deg=round(rot_err(To, Ttrue_hover), 2),
                               charger_vs_commanded_xy_mm=((To[:2, 3] - Tw_h[:2, 3]) * 1000).round(2).tolist(), charger_vs_commanded_rot_deg=round(rot_err(To, Tw_h), 2),
                               inhand_truth_change_since_start=dict(mm=round(float(np.linalg.norm(truth_G()[:3, 3] - Gs[:3, 3])) * 1000, 2), deg=round(rot_err(truth_G(), Gs), 2)))
        rec['at_hover']['reached'] = bool(rec['at_hover']['link6_pos_err_mm'] <= HOVER_OK[0] * 1000 and rec['at_hover']['link6_rot_err_deg'] <= HOVER_OK[1])
        if tool.terminal():
            end('budget_before_descent'); raise StopIteration
        if not rec['at_hover']['reached']:  # amendment v1.1: the hover target was not reached (reach limit / stall): no descent
            end('hover_unreachable'); raise StopIteration
        if np.linalg.norm(truth_G()[:3, 3] - CT.TCP) > 0.06:
            end('dropped_before_descent'); raise StopIteration
        # ---------------- guarded descent (vdrop) + unload
        e0 = co.l6(arm); tcp0 = tcp_of(e0); l6_0 = e0[:3, 3].copy(); qd = sd['quaternion_wxyz']; d = np.array([0, 0, -1.0])
        max_travel = float(hv['position'][2] - sd['position'][2]) + 0.005; n_move = int(math.ceil(max_travel / SPEED)); prog = [0.0]; cpr = [0.0]; fired = None; outcome = None
        for k in range(1, n_move + SETTLE + 1):
            if tool.terminal(): outcome = 'budget'; break
            s = min(k * SPEED, max_travel); tgt = np.array([sd['position'][0], sd['position'][1], l6_0[2] - s])
            tool.act(arm, tgt, qd, tool.opening(arm), 'guarded', steps=1)
            pr = float((tcp_of(co.l6(arm)) - tcp0) @ d); dprog = pr - prog[-1]; adv = s - cpr[-1]; prog.append(pr); cpr.append(s)
            if (k - 1) >= int(DET.get('warm', 2)) and adv > 1e-6 and dprog < float(DET['th']) * adv: fired = k; outcome = 'stopped_by_contact'; break
            if k >= n_move and abs(s - pr) <= 0.002: outcome = 'reached'; break
        if outcome is None: outcome = 'stopped_by_contact'; fired = k
        if outcome == 'stopped_by_contact':
            pm, qm = tool.eef(arm)
            for _ in range(UNLOAD):
                if tool.terminal(): break
                tool.act(arm, pm, qm, tool.opening(arm), 'unload', steps=1)
        tr('descent_end', G_lift); o = objs()[CH]; To = Tq(o['position'], o['quaternion_wxyz'])
        rec['descent'] = dict(outcome=outcome, ticks=len(prog) - 1, fired_tick=fired, link6_above_seated_mm=round((co.l6(arm)[2, 3] - sd['position'][2]) * 1000, 1),
                              charger_above_true_seat_mm=round(float(To[2, 3] - Ttrue_seat[2, 3]) * 1000, 1), charger_vs_true_seat_xy_mm=((To[:2, 3] - Ttrue_seat[:2, 3]) * 1000).round(2).tolist(),
                              charger_vs_true_seat_rot_deg=round(rot_err(To, Ttrue_seat), 2))
        rec['stage_steps']['descent'] = ad.packet['step_id']
        if outcome == 'budget' or tool.terminal():
            end('budget_in_descent'); raise StopIteration
        # ---------------- release, retreat, task checks
        tool.set_gripper(arm, opening=1.0, reason='release'); p, q = tool.eef(arm)
        if not tool.terminal(): tool.line(arm, p + np.array([0, 0, 0.05]), q, 25, 0.01, 'retreat', 'pe')
        rec['stage_steps']['release'] = ad.packet['step_id']
        rec['insertion_checks'] = RA.plain(rpc('rgbhint_check', checks=CHECKS))
        o = objs()[CH]; To = Tq(o['position'], o['quaternion_wxyz'])
        rec['final_charger'] = dict(vs_true_seat_xy_mm=((To[:2, 3] - Ttrue_seat[:2, 3]) * 1000).round(2).tolist(), vs_true_seat_z_mm=round(float(To[2, 3] - Ttrue_seat[2, 3]) * 1000, 2),
                                    vs_true_seat_rot_deg=round(rot_err(To, Ttrue_seat), 2), y_axis_tilt_deg=round(float(np.degrees(np.arccos(np.clip(To[:3, 1] @ [0, 0, 1], -1, 1)))), 1))
        # ---------------- both arms back to the reset pose (native completion also needs the arms home)
        for a_ in ([arm, [x for x in CT.ARMS if x != arm][0]]):
            if tool.terminal(): break
            tool.line(a_, home[a_][0], home[a_][1], 80, 0.005, 'home', 'pe')
        for _ in range(2):
            if tool.terminal(): break
            p, q = tool.eef(arm); tool.act(arm, p, q, tool.opening(arm), 'idle', steps=5)
        rec['stage_steps']['home'] = ad.packet['step_id']
    except StopIteration:
        pass
    except Exception as e:  # noqa: BLE001
        rec['error'] = '%s: %s' % (type(e).__name__, str(e)[:400]); rec['trace'] = traceback.format_exc()[-2000:]
        if 'START_INVALID' in str(e): rec['failure'] = 'start_invalid'
        elif rec['failure'] is None: rec['failure'] = 'engineering_error'
    finally:
        try:
            if sim is not None:
                ad_ = locals().get('ad'); rec['native_end_ack'] = getattr(ad_, 'native_end', None) if ad_ is not None else None
                rec['native'] = RA.plain(sim.rpc.request('finish', reason='pe_end'))
        except Exception as e:  # noqa: BLE001
            rec['native_finish_error'] = '%s: %s' % (type(e).__name__, str(e)[:200])
        if sim is not None: sim.close(); __import__('shutil').rmtree(str(sim.out), ignore_errors=True)  # scratch only (Kit cache copy + logs); results are under /lizekai
        nat = rec.get('native') or {}; chk = rec.get('insertion_checks')
        rec['native_success'] = bool(nat.get('native_success')) if nat else None
        try:
            rec['insertion_ok'] = bool(chk is not None and all(float(x) >= 1.0 for x in chk['items']))
        except Exception:  # noqa: BLE001
            rec['insertion_ok'] = None
        if rec['failure'] is None and not rec['native_success']:
            ic = (chk or {'items': [0, 0, 0]})['items']
            rec['failure'] = ('inserted_no_native_success' if rec['insertion_ok'] else 'toppled' if float(ic[2]) < 1.0 else 'not_inserted_upright')
        rec['steps_end'] = (rec.get('native') or {}).get('native_control_steps'); rec['seconds'] = round(time.time() - t0)
        (out / 'result.json').write_text(json.dumps(RA.plain(rec), indent=1, default=str))
        print(json.dumps(dict(start=st['id'], cond=a.cond, outlets=a.outlets, mode=a.mode, precheck=(rec.get('precheck') or {}).get('verdict'), hover_reached=(rec.get('at_hover') or {}).get('reached'), native_success=rec['native_success'], insertion_ok=rec['insertion_ok'], failure=rec['failure'],
                              steps=rec['steps_end'], error=rec.get('error'), seconds=rec['seconds'])), flush=True)


if __name__ == '__main__':
    main()
