"""harness-v2 plug batch (hv2, bottleneck line 10-28): alias config combined.hv2=true (with info_only=true) adds, on top of v1.8 info_only:
  - check_hold never surfaced (no automatic CHECK_HOLD RESULT, no 'pending' line in the post-lift report);
  - locate_outlets early abstain at 0 steps: coarse strip only from the CURRENT overhead image (no fallback to episode-start images);
    observer arm eligible only if its gripper is open (>= 0.9) with the TCP >= 5 cm from the strip centre, or fully closed (command closed)
    with the TCP >= 3 cm above the strip top (LOC1-cov fix2 + rung-4 lesson); 0-step reach pre-check of observation view A with the native
    controller model (rsi/reach_model.py); hard step cap 120 (observe <= 60 + return <= 60, asserted);
  - tool check_reach (0 steps): controller-convergence fact for a caller-given link6 pose (the check_reach v2 controller term; on fresh Astra
    plug states its "not reachable" answers were 26/26 correct, "reachable" was really reached about 60 % of the time).
v1.8 docstring follows.
v1.8 info_only mode added (online-plug-info-20261028/src/make_info_only.py): alias config combined.info_only=true -> perception information only (no plug targets, reach verdicts, guidance or advice; check_insertion_reach refused). Original docstring follows.
combined-online-20261026 harness module v1.4 (copy of the frozen v1.3 hook/combined_tools.py with the v1.4 treatment changes; copied into the
v1.4 release as rsi/combined_tools.py together with rsi/reach_model.py + rsi/robot_kin.json). v1.4 changes (treatment only):
  - in-hand pose: service runs v2-charger; gate member_spread_deg > 21.664 deg (FROZEN-v2charger.json) -> uncertain;
  - the in-hand pose is measured ONCE at the lift (reference); estimate_inhand_pose after that is a slip check (> 5 mm or > 5 deg ->
    advise put down and regrasp; targets are never re-aimed with a new pose); check_hold no longer re-aims either;
  - insertion reach pre-check (reach_model.precheck) in the post-lift report and as the tool check_insertion_reach;
  - guidance text v1.4 (compatible grasp, locate first, slip check); v1.2 step wording and the v1.1 stop-on-block rule kept.
Original docstring follows.
combined-online-20261026 harness module (copied into the release as rsi/combined_tools.py). Enabled only for aliases whose
control/ablation.json entry has `combined: {infer_url, locate_url}` (plus `stage_tools` for the v2 motion tools it reuses).

Everything the model sees from this module is English. Deploy-time inputs only: the three RGB images of the native packets (and the
native per-step observation PNGs of the same episode when present), the 14-D state, link6 poses, fixed camera calibration constants
(CALIB, read once from the simulator's camera model like PyRUA-Lean's calibration), public object geometry (outlet layout of the
strip, the seated charger pose relative to the strip). No simulator truth.

1. Tool `locate_outlets` (physical: moves the free arm): LOC1 pipeline (experiments/loc1-20261026, PASS 15:39Z): coarse SAM3
   locate of the power strip from the overhead camera -> free arm to observation pose A aimed at the coarse centre (camera 0.13 m
   above, yaw -15 deg; LOC1 single-view result = two-view result) -> frozen keypoint net on the observer-wrist image -> fit of the flat strip
   pose (x, y, z, yaw) to the 3 outlet detections -> outlet positions; then the observer arm goes back to where it was.
2. Automatic post-lift report (no tool call; appended to the reply of the action that ended with the lift): one in-hand pose estimate
   of the held object (in-hand pose network v1 through the inference service; 'unavailable' if it cannot run), and - if the outlets
   are known - link6 targets that put the charger's prongs over / into each outlet.
3. Automatic check_hold (frozen v1 rule, innerloop-20261026/checkhold): wrist image of the holding arm at the post-lift reference vs
   10 and 20 control steps later (roi_flow_px > 2.0 or roi_mad > 0.12 or |grip_change| > 0.03 -> unstable). Information only; no
   scripted regrasp.
"""
import base64, io, json, math, urllib.request
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

NAMES = ('locate_outlets', 'estimate_inhand_pose', 'check_insertion_reach', 'check_reach', 'insert_charger')  # hv2e: insert_charger (opt-in)
PRIVILEGED = {'locate_outlets': False, 'estimate_inhand_pose': False, 'check_insertion_reach': False, 'insert_charger': False}
ARMS = ('left', 'right')
TCP = np.array([0.14, 0.0, 0.0])
CALIB = json.loads(Path(__file__).with_name('combined_calib.json').read_text())  # {'mount': {arm: 4x4 link6->camera}, 'K_wrist', 'socket_top_minus_origin_m'}
OFFS = (-0.0295, 0.0070, 0.0435); OUT_Y = 0.0064; VIS_MIN = 0.5
SEAT_P = 0.0202  # charger origin above the strip origin when seated (align/plug-seated-poses.json, 6 native successes)
SEAT_R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # charger axes in the strip frame when seated (same source)
FLIPS = (0.0,)  # seated orientation as in all 6 native successes (align/plug-seated-poses.json); the 180-deg variant is untested
POSES = (('A', 0.13, -15.0),)  # LOC1 single view A: median 0.22 mm, all 3 outlets < 2 mm in 76 % (same as two views), half the steps
HOVER_M = 0.04; SPREAD_UNCERTAIN_DEG = 21.664; RMS_UNCERTAIN_PX = 5.0  # v1.4: v2-charger gate tau (FROZEN-v2charger.json); v1.3 had 13.0 for v1
SLIP_MM = 5.0; SLIP_DEG = 5.0  # v1.4 slip check vs the post-lift estimate (plug-retest p1: slips of 12-22 mm were not rescued by re-aiming)
GRASP_HINT = ('grasp the charger again across its width (fingers on the two narrow sides), wrist tilted about 30 degrees about the '
              'charger\'s width axis with the fingertips toward the pin end')
_RM = None  # reach model (lazy)  # LOC1 CONCLUSION reading 2 (gross errors have large fit residuals); frozen here  # frozen v1 gate (dispatcher 2026-10-26: seed spread vs error>10 deg AUROC 0.98-0.99 on dev)
LIFT_M = 0.03; CHECK_DT = (10, 20); GW, GH = 160, 120
HV2_STEP_CAP = 120  # hv2: locate_outlets hard cap (observe <= 60 + return <= 60)
RULE = ('Plugging guidance v1.4 (from offline tests with these estimates in this simulator; advisory): the prongs must end within about 1 mm of '
        'the outlet centre - 2 mm already misses. The in-hand pose is measured once, right after the lift; use the targets of that post-lift report. '
        'If its insertion reach pre-check says "turn needed", first turn the wrist in place to the given pre-turn pose; if it says "suggest regrasp", '
        'put the charger down on a free flat spot and ' + GRASP_HINT + ' before carrying it. Go to the hover target (prongs 4 cm above the outlet) first. '
        'There you may call estimate_inhand_pose as a slip check: if it reports that the charger moved in the fingers (more than 5 mm or 5 degrees '
        'since the lift), do not re-aim with the new pose (in tests that did not get it in) - put the charger down and grasp it again. Otherwise keep '
        'the targets and descend straight down in 2 mm steps (robodojo_act or move_linear, about 2 mm per control step, wrist orientation kept). As '
        'soon as the hand stops advancing (blocked, typically 10-17 mm above the seated pose: the prongs rest on the outlet), do NOT press further: '
        'hold the current measured pose for 3 control steps, then open the gripper and lift the hand straight up - the charger then drops into the '
        'outlet by itself. In those tests pressing after the block made the charger slide in the fingers and fall over, and sideways searching made '
        'things worse. All numbers are estimates from RGB; check them against the camera images.')
CAVEAT = 'Returned values may be inaccurate; check them against the camera images yourself.'
INFO_POSE_NOTE = ('in-hand pose network v3 (offline: scripted charger holds median 0.53 mm / 0.48 deg; on Astra-like charger holds about 0.5 mm, '
                  'but about 1 frame in 8 can be wrong by a whole 180 deg turn of the charger without this report noticing it); check it against the images')


def T_of(p, q):
    M = np.eye(4); M[:3, :3] = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); M[:3, 3] = p; return M


def q_of(M):
    q = R.from_matrix(M[:3, :3]).as_quat(); return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def _post(url, obj, timeout=60):
    r = urllib.request.Request(url, data=json.dumps(obj).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())


def _b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def _imgs(pk):
    return {i['camera']: i['path'] for i in pk['images']}


# ---------------- LOC1 geometry (eval_loc1.py, frozen; copied) ----------------
def project(K, Tc2w, P):
    K = np.asarray(K); Tw2c = np.linalg.inv(np.asarray(Tc2w)); q = (Tw2c[:3, :3] @ np.asarray(P).T).T + Tw2c[:3, 3]
    z = -q[:, 2]; return np.c_[K[0, 0] * q[:, 0] / z + K[0, 2], -K[1, 1] * q[:, 1] / z + K[1, 2]], z


def outlets_of(th):
    x, y, z, yaw = th; c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[x + c * ox - s * OUT_Y, y + s * ox + c * OUT_Y, z] for ox in OFFS])


def fit(views, init_xyz, init_yaw):
    obs = [(v['K'], v['T'], k, np.array(v['det'][k][:2])) for v in views for k in range(3) if v['det'][k][2] >= VIS_MIN]
    if len(obs) < 3:
        return None
    def res(th):
        P = outlets_of(th); r = []
        for K, T, k, uv in obs:
            p, z = project(K, T, P[k:k + 1]); r.extend((p[0] - uv) if z[0] > 0 else [1e3, 1e3])
        return np.array(r)
    best = None
    for dy in (0, math.pi / 2, math.pi, -math.pi / 2):
        sol = least_squares(res, np.r_[init_xyz, init_yaw + dy], loss='huber', f_scale=3.0, x_scale=[0.01, 0.01, 0.01, 0.1])
        if best is None or sol.cost < best.cost:
            best = sol
    r = res(best.x).reshape(-1, 2); return dict(theta=best.x.tolist(), outlets=outlets_of(best.x).tolist(), rms_px=float(np.sqrt((r ** 2).sum(1).mean())), n_obs=len(obs))


def down_quat(yaw_deg):
    X = np.array([0, 0, -1.0]); Y = np.array([0, 1.0, 0]); M = np.eye(4); M[:3, :3] = np.column_stack([X, Y, np.cross(X, Y)])
    M[:3, :3] = R.from_euler('z', yaw_deg, degrees=True).as_matrix() @ M[:3, :3]; return q_of(M)


def aim(Tlc, gaze, h_cam, yaw_deg):
    q = down_quat(yaw_deg); Rl = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); Rc = Rl @ Tlc[:3, :3]; d = Rc @ np.array([0, 0, -1.0])
    p_cam = np.asarray(gaze) - (h_cam / -d[2]) * d; return p_cam - Rl @ Tlc[:3, 3], q


# ---------------- check_hold v1 features (track3-slip feats_classic.pair_feats scalars, frozen; copied) ----------------
def _small(path):
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if im is None else cv2.resize(im, (GW, GH), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def roi_feats(a, b):
    import cv2
    d = np.abs(a - b).mean(-1)
    ga, gb = (cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) for x in (a, b))
    fl = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0); mag = np.linalg.norm(fl, axis=-1)
    roi = (slice(GH // 3, GH), slice(GW // 4, 3 * GW // 4))
    return float(d[roi].mean()), float(np.median(mag[roi]))


# ---------------- per-episode state ----------------
class Co:
    def __init__(self, ad, cfg):
        self.ad, self.cfg = ad, cfg; self.hold = {a: None for a in ARMS}; self.outlets = None; self.start_imgs = None; self.log = []
        self.cmd_closed = {a: None for a in ARMS}  # last top-level command (None = unknown, e.g. after a helper tool)
        self.info_only = bool((cfg or {}).get('info_only'))  # v1.8
        self.hv2 = bool((cfg or {}).get('hv2'))  # harness-v2 plug batch

    # proprio helpers
    def l6(self, arm):
        e = self.ad.packet['current_eef'][arm]; return T_of(e['position'], e['quaternion_wxyz'])

    def tcp(self, arm):
        L = self.l6(arm); return L[:3, 3] + L[:3, :3] @ TCP

    def grip(self, arm):
        return float(self.ad.packet['current_state'][ARMS.index(arm) * 7 + 6])

    def holding(self, arm):
        g = self.grip(arm); return bool(self.cmd_closed[arm] is not False and 0.02 < g < 0.9)

    def step_png(self, step, cam):
        """Native per-step observation PNG of this episode (written by the native host), if present."""
        try:
            for d in sorted(Path(self.ad.output).glob('*/observations')):
                p = d / ('%06d-%s.png' % (step, cam))
                if p.is_file(): return p
        except Exception:  # noqa: BLE001
            pass
        return None

    def note_command(self, args):
        for arm in ARMS:
            t = ((args or {}).get('target') or {}).get(arm) or {}
            if 'gripper_closed' in t:
                self.cmd_closed[arm] = bool(t['gripper_closed'])
            elif args is not None and 'target' not in args:
                self.cmd_closed[arm] = None

    # in-hand pose (service) -> dict
    def inhand(self, arm, category='charger'):
        pk = self.ad.packet; im = _imgs(pk)
        req = dict(images_b64={'cam_high': _b64(im['cam_high']), 'cam_%s_wrist' % arm: _b64(im['cam_%s_wrist' % arm])}, states=pk['current_state'],
                   eef_positions=[pk['current_eef'][a]['position'] for a in ARMS], eef_quaternions_wxyz=[pk['current_eef'][a]['quaternion_wxyz'] for a in ARMS],
                   arm=arm, category=category, gripper_cmd=1.0 if self.cmd_closed[arm] is False else 0.0)
        try:
            r = _post(self.cfg['infer_url'].rstrip('/') + '/inhand', req, timeout=60)
        except Exception as e:  # noqa: BLE001
            return dict(status='unavailable', note='in-hand pose estimator not reachable (%s)' % type(e).__name__)
        if 'position_m' not in r:
            return dict(status='unavailable', note=str(r.get('error', 'no estimate'))[:200])
        G = np.eye(4); G[:3, :3] = R.from_quat([r['quaternion_wxyz'][1], r['quaternion_wxyz'][2], r['quaternion_wxyz'][3], r['quaternion_wxyz'][0]]).as_matrix()
        G[:3, 3] = TCP + np.asarray(r['position_m'])                       # object in the link6 frame
        L = self.l6(arm); W = L @ G
        _tau = float(self.cfg.get('spread_tau_deg', SPREAD_UNCERTAIN_DEG))  # v1.8: gate tau of the served model (v3: 22.795)
        if isinstance(r.get('member_spread_deg'), (int, float)) and r['member_spread_deg'] > _tau:  # orientation withheld, position only
            if self.info_only:
                return dict(status='uncertain', step=pk['step_id'], object=category, position_in_tcp_frame_m=[round(float(x), 4) for x in r['position_m']],
                            object_position_world_m=[round(float(x), 4) for x in W[:3, 3]],
                            note='the estimator members disagree about the orientation (spread %.0f deg > %.0f): orientation not reported' % (float(r['member_spread_deg']), _tau))
            return dict(status='uncertain', step=pk['step_id'], object=category, position_in_tcp_frame_m=[round(float(x), 4) for x in r['position_m']],
                        object_position_world_m=[round(float(x), 4) for x in W[:3, 3]],
                        note='the estimator members disagree about the orientation (spread %.0f deg > %.0f): orientation not reported, no plug targets; '
                             'look again from a clearer view (e.g. raise the hand) and call estimate_inhand_pose' % (float(r['member_spread_deg']), SPREAD_UNCERTAIN_DEG))
        return dict(status='ok', step=pk['step_id'], object=category, position_in_tcp_frame_m=[round(float(x), 4) for x in r['position_m']],
                    quaternion_in_tcp_frame_wxyz=[round(float(x), 4) for x in r['quaternion_wxyz']],
                    position_sigma_mm=None if r.get('position_sigma_m') is None else round(1000 * float(r['position_sigma_m']), 1),
                    rotation_sigma_deg=None if r.get('rotation_sigma_deg') is None else round(float(r['rotation_sigma_deg']), 1),
                    member_spread_mm=r.get('member_spread_mm'), member_spread_deg=r.get('member_spread_deg'),
                    reliability='members agree',
                    accuracy_note=INFO_POSE_NOTE if self.info_only else ('in-hand pose v2-charger; held-out charger holds: median 0.8 mm / 0.9 deg, but 10 % of cases off by more than 27 mm / 110 deg; '
                                  'scripted width grasps tilted 30 deg: median 0.6 mm / 0.3 deg'),
                    object_position_world_m=[round(float(x), 4) for x in W[:3, 3]], offset_link6_to_object_world_m=[round(float(x), 4) for x in (W[:3, 3] - L[:3, 3])],
                    _G=G.tolist())

    def plug_targets(self, arm, est):
        """link6 targets that bring the held charger (estimated in-hand pose) to the seated pose in each estimated outlet."""
        if not self.outlets or est.get('status') != 'ok': return None
        x, y, ztop, yaw = self.outlets['theta']; G = np.asarray(est['_G']); out = []
        Ts = np.eye(4); Ts[:3, :3] = R.from_euler('z', yaw).as_matrix(); Ts[:3, 3] = [x, y, ztop - CALIB['socket_top_minus_origin_m']]
        L = self.l6(arm); qcur = R.from_matrix(L[:3, :3])
        for k, ox in enumerate(OFFS):
            best = None
            for flip in FLIPS:
                S = np.eye(4); S[:3, :3] = R.from_euler('z', flip).as_matrix() @ SEAT_R; S[:3, 3] = [ox, OUT_Y, SEAT_P]
                Tw = Ts @ S; Tl = Tw @ np.linalg.inv(G); turn = float(np.degrees((R.from_matrix(Tl[:3, :3]) * qcur.inv()).magnitude()))
                if best is None or turn < best[0]: best = (turn, Tl, flip)
            turn, Tl, flip = best; H = Tl.copy(); H[2, 3] += HOVER_M
            out.append(dict(outlet=k, outlet_position_m=[round(float(v), 4) for v in self.outlets['outlets'][k]],
                            hover_link6=dict(position=[round(float(v), 4) for v in H[:3, 3]], quaternion_wxyz=[round(v, 5) for v in q_of(H)]),
                            seated_link6=dict(position=[round(float(v), 4) for v in Tl[:3, 3]], quaternion_wxyz=[round(v, 5) for v in q_of(Tl)]),
                            wrist_turn_from_now_deg=round(turn, 1), seated_variant='canonical' if flip == 0.0 else 'turned 180 deg', hand_travel_m=round(float(np.linalg.norm(H[:3, 3] - L[:3, 3])), 3)))
        return out

    # ---------------- automatic post-act processing -> list of content items ----------------
    def after(self, args=None):
        items = []
        if self.ad.packet is None: return items
        if self.start_imgs is None: self.start_imgs = _imgs(self.ad.packet)
        self.note_command(args)
        for arm in ARMS:
            h = self.hold[arm]; holding = self.holding(arm); step = int(self.ad.packet['step_id'])
            if h is not None and not holding:
                self.hold[arm] = None; continue
            if h is None and holding:
                self.hold[arm] = dict(close_step=step, z0=float(self.tcp(arm)[2]), ref=None); continue
            if h is None: continue
            if h['ref'] is None and self.tcp(arm)[2] >= h['z0'] + LIFT_M:
                p = _imgs(self.ad.packet)['cam_%s_wrist' % arm]
                h.update(ref=step, ref_img=str(p), g_ref=self.grip(arm), checks=[], reported=False)
                est = self.inhand(arm); h['est'] = est
                rep = dict(event='post_lift_report', arm=arm, step=step, inhand_pose={k: v for k, v in est.items() if not k.startswith('_')})
                tg = None if self.info_only else self.plug_targets(arm, est)
                if self.info_only: pass  # v1.8: no targets / reach verdicts
                elif tg:
                    rep['plug_targets'] = tg; rep['insertion_reach_precheck'] = self.reach_report(arm, est)
                elif est.get('status') != 'ok': rep['plug_targets'] = 'not computed: no reliable in-hand pose (see inhand_pose)'
                elif self.outlets is None: rep['plug_targets'] = 'not computed: the outlets have not been located yet (locate_outlets); call it, or ask for the targets again after it'
                if not self.hv2: rep['check_hold'] = dict(status='pending', note='the hold is checked automatically 10 and 20 control steps after this lift')
                items.append(self._text('AUTOMATIC POST-LIFT REPORT', rep, rule=not self.info_only)); self.log.append(rep); continue
            if h['ref'] is not None and not h.get('reported') and not self.hv2:  # hv2: check_hold never surfaced
                for dt in CHECK_DT:
                    if any(c['dt'] == dt for c in h['checks']) or step < h['ref'] + dt: continue
                    p = self.step_png(h['ref'] + dt, 'cam_%s_wrist' % arm); exact = p is not None
                    if p is None: p = _imgs(self.ad.packet)['cam_%s_wrist' % arm]
                    a, b = _small(h['ref_img']), _small(p)
                    if a is None or b is None: continue
                    mad, flow = roi_feats(a, b); g_now = self.grip(arm)
                    h['checks'].append(dict(dt=dt, frame_step=(h['ref'] + dt) if exact else step, roi_flow_px=round(flow, 2), roi_mad=round(mad, 3),
                                            grip_change=round(g_now - h['g_ref'], 3), alarm=bool(flow > 2.0 or mad > 0.12 or abs(g_now - h['g_ref']) > 0.03)))
                if len(h['checks']) == len(CHECK_DT) or (h['checks'] and step >= h['ref'] + CHECK_DT[-1]):
                    h['reported'] = True; items.append(self._text('AUTOMATIC CHECK_HOLD RESULT', self._hold_report(arm, h), rule=False))
        return items

    def _hold_report(self, arm, h):
        al = [c for c in h['checks'] if c['alarm']]; unstable = bool(al)
        est2 = self.inhand(arm) if unstable else None; rot = shift = 'unknown'
        if est2 and est2.get('status') == 'ok' and h['est'].get('status') == 'ok':
            G0, G1 = np.asarray(h['est']['_G']), np.asarray(est2['_G'])
            rot = round(float(np.degrees((R.from_matrix(G1[:3, :3]) * R.from_matrix(G0[:3, :3]).inv()).magnitude())), 1)
            shift = round(float(np.linalg.norm(G1[:3, 3] - G0[:3, 3]) * 1000), 1)
        rep = dict(event='check_hold', arm=arm, status='unstable' if unstable else 'stable', reference_step=h['ref'],
                   first_alarm_step=al[0]['frame_step'] if al else None,
                   roi_flow_px=[c['roi_flow_px'] for c in h['checks']], roi_mad=[c['roi_mad'] for c in h['checks']],
                   checked_steps=[c['frame_step'] for c in h['checks']], gripper_opening_ref=round(h['g_ref'], 3), gripper_opening_now=round(self.grip(arm), 3),
                   object_still_in_hand=self.holding(arm), estimated_rotation_deg=rot, estimated_shift_mm=shift,
                   basis='holding-arm wrist image vs the post-lift reference and the gripper opening; thresholds from simulation (detection 0.84, false alarm 0.19 on 131 held-out Astra holds)')
        if unstable and self.info_only:  # v1.8: factual line only
            rep['note'] = 'The held object seems to be turning or sliding in the fingers%s.' % ('' if rot == 'unknown' else ' (about %s deg since the lift)' % rot)
        elif unstable:
            rep['note'] = ('The held object seems to be turning or sliding in the fingers%s. Carrying it further usually ends with it dropped or turned, '
                           'and the plug targets assume the hold measured at the lift. Put it down on a free flat spot of the table and %s.') % (
                           '' if rot == 'unknown' else ' (about %s deg since the lift)' % rot, GRASP_HINT)
        else:
            rep['note'] = 'No turning or sliding detected in the first 20 control steps after the lift.'
        return rep  # v1.4: no re-aimed targets after an alarm (put down and regrasp instead)

    def _text(self, title, obj, rule):
        body = '=== %s (automatic, from the harness; RGB-only estimates, privileged=false) ===\n' % title + json.dumps(obj)
        if rule: body += '\n' + RULE
        return dict(type='inputText', text=body + '\n=== END ===')

    # ---------------- tool: estimate_inhand_pose (no motion) ----------------
    def refresh(self, arm=None):
        """v1.4: before a reliable post-lift estimate exists -> a fresh estimate (becomes the reference); after it -> slip check against the
        reference (the targets stay those of the reference; never re-aimed with a new pose)."""
        from .stage_tools import Refused
        busy = [a for a in ARMS if self.holding(a)]
        if arm is None:
            if len(busy) != 1: raise Refused('give arm: %s' % ('no gripper holds anything' if not busy else 'both grippers hold something'))
            arm = busy[0]
        if arm not in busy: raise Refused('the %s gripper does not hold anything' % arm)
        est = self.inhand(arm); h = self.hold[arm]; ref = (h or {}).get('est')
        der = dict(inhand_pose={k: v for k, v in est.items() if not k.startswith('_')})
        if self.info_only:  # v1.8: pose + measured change since the post-lift estimate; no advice, no targets
            if ref is not None and ref.get('status') == 'ok' and est.get('status') == 'ok':
                G0, G1 = np.asarray(ref['_G']), np.asarray(est['_G'])
                der['change_since_post_lift'] = dict(reference_step=ref.get('step'), shift_mm=round(float(np.linalg.norm(G1[:3, 3] - G0[:3, 3]) * 1000), 1),
                                                     rotation_deg=round(float(np.degrees((R.from_matrix(G1[:3, :3]) * R.from_matrix(G0[:3, :3]).inv()).magnitude())), 1))
            elif h is not None and ref is None and est.get('status') == 'ok': h['est'] = est
            return dict(status={'ok': 'ok', 'uncertain': 'uncertain'}.get(est.get('status'), 'failed'), measured=dict(arm=arm, step=self.ad.packet['step_id'], steps_used=0), derived=der)
        if ref is not None and ref.get('status') == 'ok':
            sc = dict(reference_step=ref.get('step'), threshold='%.0f mm / %.0f deg' % (SLIP_MM, SLIP_DEG))
            if est.get('status') == 'ok':
                G0, G1 = np.asarray(ref['_G']), np.asarray(est['_G'])
                sc['shift_mm'] = round(float(np.linalg.norm(G1[:3, 3] - G0[:3, 3]) * 1000), 1)
                sc['rotation_deg'] = round(float(np.degrees((R.from_matrix(G1[:3, :3]) * R.from_matrix(G0[:3, :3]).inv()).magnitude())), 1)
                sc['status'] = 'slipped' if (sc['shift_mm'] > SLIP_MM or sc['rotation_deg'] > SLIP_DEG) else 'no slip'
            else:
                sc['status'] = 'unknown'
            if sc['status'] == 'slipped':
                sc['advice'] = ('The charger has moved in the fingers since the lift. Do not re-aim with the new pose (in tests that did not get the plug in): '
                                'put the charger down on a free flat spot and %s; the post-lift report of the new lift gives new targets.' % GRASP_HINT)
                der['plug_targets'] = 'withheld: the charger moved in the fingers since the lift (see slip_check)'
            else:
                sc['advice'] = ('No slip detected: keep the targets of the post-lift estimate (repeated below).' if sc['status'] == 'no slip' else
                                'The new estimate is not reliable (see inhand_pose); the targets of the post-lift estimate are repeated below.')
                tg = self.plug_targets(arm, ref)
                der['plug_targets'] = tg if tg else 'not computed: the outlets have not been located yet (locate_outlets)'
            der['slip_check'] = sc
            return dict(status='ok' if est.get('status') == 'ok' else {'uncertain': 'uncertain'}.get(est.get('status'), 'failed'),
                        measured=dict(arm=arm, step=self.ad.packet['step_id'], steps_used=0), derived=der)
        if h is not None and est.get('status') == 'ok': h['est'] = est  # first reliable estimate = the reference
        tg = self.plug_targets(arm, est)
        der['plug_targets'] = tg if tg else ('not computed: no reliable in-hand pose' if est.get('status') != 'ok' else 'not computed: the outlets have not been located yet (locate_outlets)')
        if tg: der['insertion_reach_precheck'] = self.reach_report(arm, est)
        return dict(status={'ok': 'ok', 'uncertain': 'uncertain'}.get(est.get('status'), 'failed'), measured=dict(arm=arm, step=self.ad.packet['step_id'], steps_used=0), derived=der)

    # ---------------- v1.4 insertion reach pre-check (no motion) ----------------
    def seat_poses(self):
        x, y, ztop, yaw = self.outlets['theta']; Ts = np.eye(4); Ts[:3, :3] = R.from_euler('z', yaw).as_matrix(); Ts[:3, 3] = [x, y, ztop - CALIB['socket_top_minus_origin_m']]
        out = []
        for ox in OFFS:
            S = np.eye(4); S[:3, :3] = SEAT_R; S[:3, 3] = [ox, OUT_Y, SEAT_P]; out.append(Ts @ S)
        return out

    def reach_report(self, arm, est):
        global _RM
        try:
            from . import reach_model as RMod
        except ImportError:  # plain-module import (offline smoke)
            import reach_model as RMod
        try:
            if _RM is None: _RM = RMod.Reach()
            i = ARMS.index(arm); q = np.asarray(self.ad.packet['current_state'][7 * i: 7 * i + 6], float)
            pc = RMod.precheck(_RM, arm, q, self.l6(arm), np.asarray(est['_G']), self.seat_poses(), HOVER_M)
        except Exception as e:  # noqa: BLE001
            return dict(verdict='unavailable', note='reach model failed (%s)' % type(e).__name__)
        adv = {'reachable': 'The hover and seated targets of outlet(s) %s are reachable from the current arm configuration with this hold.' % pc['reachable_outlets'],
               'turn needed': ('Not reachable directly; reachable after first turning the wrist in place to pre_turn (5 cm above the current hand '
                               'position, hover orientation), then going to the hover target of outlet %s.' % ((pc.get('pre_turn') or {}).get('outlet'))),
               'suggest regrasp': ('No outlet is reachable with this hold (joint limits): put the charger down on a free flat spot and %s.' % GRASP_HINT)}[pc['verdict']]
        rep = dict(verdict=pc['verdict'], advice=adv, reachable_outlets=pc['reachable_outlets'],
                   outlets=[dict(outlet=r['outlet'], hover_reachable=r['hover_reachable'], seated_reachable=r['seated_reachable'], wrist_turn_deg=r['wrist_turn_deg']) for r in pc['outlets']],
                   basis='controller model (URDF chain, joint limits) from the measured joints with the estimated in-hand pose and outlet positions; prediction, not a guarantee')
        if pc.get('pre_turn'): rep['pre_turn'] = pc['pre_turn']
        return rep

    def reach_check(self, arm=None):
        from .stage_tools import Refused
        busy = [a for a in ARMS if self.holding(a)]
        if arm is None:
            if len(busy) != 1: raise Refused('give arm: %s' % ('no gripper holds anything' if not busy else 'both grippers hold something'))
            arm = busy[0]
        if arm not in busy: raise Refused('the %s gripper does not hold anything' % arm)
        if not self.outlets: raise Refused('the outlets have not been located yet (call locate_outlets first)')
        h = self.hold[arm]; est = (h or {}).get('est')
        if not est or est.get('status') != 'ok':
            raise Refused('no reliable in-hand pose of the held charger yet (post-lift report missing or uncertain; call estimate_inhand_pose)')
        rep = self.reach_report(arm, est)
        return dict(status='ok' if rep['verdict'] != 'unavailable' else 'failed', measured=dict(arm=arm, step=self.ad.packet['step_id'], steps_used=0, inhand_pose_step=est.get('step')),
                    derived=rep)

    # ---------------- tool: locate_outlets ----------------
    def hv2_eligible(self, arm, strip_c, strip_top):
        """hv2 observer eligibility (deploy inputs only); raises Refused at 0 steps."""
        from .stage_tools import Refused
        g = self.grip(arm); tcp = self.tcp(arm); top = float(strip_top) if strip_top is not None else float(strip_c[2])
        d_xy = float(np.linalg.norm(tcp[:2] - np.asarray(strip_c[:2], float)))
        if g >= 0.9 and (d_xy >= 0.05 or tcp[2] - top >= 0.05): return
        if self.cmd_closed[arm] is True and g < 0.02 and tcp[2] - top >= 0.03: return
        raise Refused('abstain (0 steps): the %s hand is too close to the power strip to be moved safely (gripper %.2f, %.0f mm from the strip centre, %.0f mm above its top)'
                      % (arm, g, d_xy * 1000, (tcp[2] - top) * 1000))

    def hv2_reach_precheck(self, arm, gaze):
        """0-step native-controller pre-check of observation view A; raises Refused if it does not converge."""
        from .stage_tools import Refused
        global _RM
        from . import reach_model as RMod
        if _RM is None: _RM = RMod.Reach()
        tag, hc, yaw = POSES[0]; tgt, q = aim(np.asarray(CALIB['mount'][arm]), gaze, hc, yaw)
        M = T_of(tgt, q); i = ARMS.index(arm); q0 = np.asarray(self.ad.packet['current_state'][7 * i: 7 * i + 6], float)
        _, pe, re, ok = _RM.go(arm, q0, M, ticks=300, tol=(0.02, math.radians(10)))
        if not ok: raise Refused('abstain (0 steps): the observation pose above the strip is not reachable by the %s arm from its current joints (controller model residual %.0f mm)' % (arm, pe * 1000))

    def check_reach(self, arm, target):
        """hv2 tool: native-controller convergence fact for a caller-given link6 pose (0 steps)."""
        from .stage_tools import Refused
        global _RM
        from . import reach_model as RMod
        if _RM is None: _RM = RMod.Reach()
        if arm not in ARMS: raise Refused('arm must be left or right')
        try:
            pos = [float(x) for x in target['position']]; quat = [float(x) for x in target['quaternion_wxyz']]
            assert len(pos) == 3 and len(quat) == 4
        except Exception:  # noqa: BLE001
            raise Refused('target_pose must be {position: [x, y, z] (m, world frame of your actions), quaternion_wxyz: [w, x, y, z]}')
        i = ARMS.index(arm); q0 = np.asarray(self.ad.packet['current_state'][7 * i: 7 * i + 6], float)
        _, pe, re, ok = _RM.go(arm, q0, T_of(pos, quat))
        return dict(status='ok', measured=dict(arm=arm, step=self.ad.packet['step_id'], steps_used=0),
                    derived=dict(reachable_from_current=bool(ok), controller_residual_mm=round(pe * 1000, 1), controller_residual_deg=round(re, 1),
                                 note=('Native-controller model from the current joints (3 mm / 3 deg). On fresh plug states, "not reachable" was always right (26/26); '
                                       '"reachable" was really reached about 60 % of the time. Objects, the table and the held object are not modelled.')))

    def locate_outlets(self, tool, observer=None, return_arm=True):
        from .stage_tools import Refused
        pk = self.ad.packet; start = tool.steps
        def coarse(paths):
            req = dict(images_b64={c: _b64(p) for c, p in paths.items()}, states=pk['current_state'],
                       eef_positions=[pk['current_eef'][a]['position'] for a in ARMS], eef_quaternions_wxyz=[pk['current_eef'][a]['quaternion_wxyz'] for a in ARMS],
                       object='power strip', camera='cam_head')
            return _post(self.cfg['locate_url'], req, timeout=120)
        loc = coarse(_imgs(pk)); src = 'current'
        if self.hv2 and (not loc.get('found') or loc.get('center_xyz') is None):
            raise Refused('abstain (0 steps): the power strip is not visible in the current overhead image (no fallback to older images)')
        if (not loc.get('found') or loc.get('center_xyz') is None) and self.start_imgs:
            loc = coarse(self.start_imgs); src = 'episode start images'
        if not loc.get('found') or loc.get('center_xyz') is None:
            raise Refused('the power strip was not found in the overhead camera')
        c = np.asarray(loc['center_xyz'], float); gaze = np.array([c[0], c[1], loc.get('top_z') if loc.get('top_z') is not None else c[2]])
        busy = [a for a in ARMS if self.holding(a)]
        if observer is None:
            free = [a for a in ARMS if a not in busy]
            if not free: raise Refused('both grippers hold something; one arm must be free to look at the strip')
            observer = free[0] if len(free) == 1 else ('left' if c[0] < 0 else 'right')
        elif observer in busy:
            raise Refused('the %s gripper holds something; choose the free arm as observer' % observer)
        if self.hv2:
            self.hv2_eligible(observer, c, loc.get('top_z'))
            self.hv2_reach_precheck(observer, gaze)
        home = self.l6(observer); views = []; moves = []
        for tag, hc, yaw in POSES:
            tgt, q = aim(np.asarray(CALIB['mount'][observer]), gaze, hc, yaw)
            r = tool.line(observer, tgt, q, 60, 0.01, 'observe', 'locate_outlets'); moves.append(dict(view=tag, **r['measured']))
            if r['measured']['position_error_m'] > 0.02: continue
            p = _imgs(self.ad.packet)['cam_%s_wrist' % observer]
            try:
                d = _post(self.cfg['infer_url'].rstrip('/') + '/outlets', dict(image_b64=_b64(p)), timeout=60)['detections']
            except Exception as e:  # noqa: BLE001
                moves[-1]['detector_error'] = 'outlet detector not reachable (%s)' % type(e).__name__; continue
            Tc = self.l6(observer) @ np.asarray(CALIB['mount'][observer])
            views.append(dict(tag=tag, K=CALIB['K_wrist'], T=Tc.tolist(), det=d))
        f = fit(views, gaze.tolist(), math.radians(loc.get('long_axis_deg') or 0.0)) if views else None
        if return_arm:
            tool.line(observer, home[:3, 3], q_of(home), 60, 0.01, 'return', 'locate_outlets')
        if self.hv2: assert tool.steps - start <= HV2_STEP_CAP, ('step cap', tool.steps - start)
        meas = dict(coarse=dict(source=src, center_xyz=[round(float(v), 4) for v in c], long_axis_deg=loc.get('long_axis_deg'), views=loc.get('views')),
                    observer_arm=observer, observation_moves=moves, detections={v['tag']: [[round(x, 1) for x in dd[:2]] + [round(dd[2], 2)] for dd in v['det']] for v in views},
                    steps_used=tool.steps - start)
        if f is None:
            return dict(status='failed', measured=meas, derived=dict(found=False, note='no outlet position: the observation pose was not reached, the detector failed, or fewer than 3 outlets were detected'))
        if f['rms_px'] > RMS_UNCERTAIN_PX:
            return dict(status='uncertain', measured=dict(meas, fit_rms_px=round(f['rms_px'], 2)), derived=dict(found=False,
                        note='the outlet fit residual is large (%.1f px > %.0f px): positions not reported; try again with the other arm or after moving the hands out of the view' % (f['rms_px'], RMS_UNCERTAIN_PX)))
        self.outlets = f
        der = dict(found=True, outlets_xyz_m=[[round(float(v), 4) for v in p] for p in f['outlets']], strip_yaw_deg=round(math.degrees(f['theta'][3]), 1),
                   fit_rms_px=round(f['rms_px'], 2), n_detections=f['n_obs'],
                   accuracy_note='offline test of this locator: median outlet error 0.15 mm, all 3 outlets within 2 mm in 76 % of cases')
        for a in ([] if self.info_only else busy):  # v1.8: no targets in info_only
            h = self.hold[a]
            if h and h.get('est'):
                tg = self.plug_targets(a, h['est'])
                if tg:
                    der.setdefault('plug_targets', {})[a] = tg; der.setdefault('insertion_reach_precheck', {})[a] = self.reach_report(a, h['est'])
        return dict(status='ok', measured=meas, derived=der)


SPECS = [
    {'type': 'function', 'name': 'locate_outlets',
     'description': ('Locate the three outlets of the power strip to about a millimetre. The free arm (observer_arm, default: the arm that holds '
                     'nothing) is moved to an observation pose above the strip (wrist camera about 13 cm above it), a keypoint detector finds the '
                     'outlets in that wrist image and a model fit gives their 3-D positions; then the observer arm returns to where it was '
                     '(return_arm false leaves it there). Step budget: it costs about 105 of the episode\'s 400 native control steps (measured 100-105 '
                     'including the return), so call it once - first, at the start of the episode while both grippers are empty (the arm on the strip side then observes), before '
                     'grasping the charger - rather than repeatedly, and keep '
                     'about 40 steps at the end to reset the arms. If one gripper holds the charger and its in-hand pose has been estimated '
                     '(automatic report after the lift), the reply also gives link6 targets for that arm: hover (prongs 4 cm above the outlet) and '
                     'seated (prongs in the outlet) for each outlet. Returns status (ok / failed / refused), measured {coarse, observation_moves, '
                     'detections, steps_used} and derived {outlets_xyz_m, strip_yaw_deg, plug_targets}. Non-privileged: RGB images, robot '
                     'proprioception and the strip\'s outlet layout only. ' + CAVEAT),
     'inputSchema': {'type': 'object', 'properties': {
         'observer_arm': {'type': 'string', 'enum': list(ARMS), 'description': 'optional; must be an empty gripper'},
         'return_arm': {'type': 'boolean', 'description': 'move the observer arm back afterwards (default true)'},
         'reason': {'type': 'string'}}, 'required': ['reason'], 'additionalProperties': False}},
    {'type': 'function', 'name': 'estimate_inhand_pose',
     'description': ('Estimate the pose of the object held by one gripper relative to the gripper, from the current wrist and overhead camera '
                     'images (in-hand pose network v2-charger; no motion, no control steps). The pose is measured automatically once after the '
                     'lift (post-lift report); after that this tool is an optional slip check (e.g. at the hover target): it compares the new '
                     'estimate with the post-lift one and, if the charger moved more than 5 mm or 5 degrees in the fingers, says so and advises to '
                     'put it down and grasp it again (re-aiming with the new pose did not get the plug in during tests); otherwise it repeats the '
                     'post-lift link6 hover / seated targets. Returns status, measured {arm, step} and derived {inhand_pose (position / '
                     'orientation in the TCP frame, uncertainty, member agreement), slip_check, plug_targets}. Non-privileged. ' + CAVEAT),
     'inputSchema': {'type': 'object', 'properties': {
         'arm': {'type': 'string', 'enum': list(ARMS), 'description': 'the holding arm (default: the only arm that holds something)'},
         'reason': {'type': 'string'}}, 'required': ['reason'], 'additionalProperties': False}},
    {'type': 'function', 'name': 'check_insertion_reach',
     'description': ('Insertion reach pre-check (no motion, no control steps): predicts, with a model of the arm controller (joint limits) from the '
                     'current measured joints, whether the link6 hover and seated targets of each located outlet can be reached with the current '
                     'hold (post-lift in-hand pose). Returns status, measured {arm, step} and derived {verdict: reachable / turn needed / suggest '
                     'regrasp, advice, reachable_outlets, outlets, pre_turn (only for turn needed: a link6 pose 5 cm above the hand with the hover '
                     'orientation, to go to first)}. The same check is included in the automatic post-lift report. Needs locate_outlets first. '
                     'Non-privileged: robot model, proprioception and RGB estimates only. ' + CAVEAT),
     'inputSchema': {'type': 'object', 'properties': {
         'arm': {'type': 'string', 'enum': list(ARMS), 'description': 'the holding arm (default: the only arm that holds something)'},
         'reason': {'type': 'string'}}, 'required': ['reason'], 'additionalProperties': False}},
]


def state(ad, cfg):
    co = getattr(ad, '_co', None)
    if co is None:
        co = Co(ad, cfg); ad._co = co
    return co


def after_act(ad, cfg, args):
    """Called by NativeAdapter after every top-level robodojo_act / stage tool reply; returns extra English content items."""
    try:
        co = state(ad, cfg); items = co.after(args)
        if cfg.get('quiet_auto_reports'): items = []  # hv2f1: hold tracking runs, no automatic report text
        if isinstance(cfg.get('l3_policy'), dict):  # hv2e: intervention gate v2 (log / offer), per-step history
            from . import hv2e_ext as _x
            try: items = list(items or []) + _x.l3_after_act(ad, cfg)
            except Exception as e:  # noqa: BLE001
                (Path(ad.evidence) / 'l3-gate-error.txt').write_text('%s: %s' % (type(e).__name__, str(e)[:300]))
        if items:
            (Path(ad.evidence) / ('combined-%04d-%d.json' % (len(co.log), ad.packet['step_id']))).write_text(json.dumps(dict(step=ad.packet['step_id'], items=items)))  # evidence of what was shown
        return items
    except Exception as e:  # noqa: BLE001  (never break the episode because of this module)
        return [dict(type='inputText', text='=== AUTOMATIC POST-LIFT REPORT unavailable (%s) ===' % type(e).__name__)]


def run(ad, name, args, call_id, on_ack, cfg):
    from . import stage_tools as ST
    co = state(ad, cfg); t = ST._Tool(ad, name, call_id, on_ack, cfg)
    try:
        if ad.packet is None: raise ST.Refused('call robodojo_start first')
        if isinstance(cfg.get('tools'), list) and name not in cfg['tools']: raise ST.Refused('not available in this harness')  # hv2f1
        if name == 'estimate_inhand_pose':
            out = co.refresh(args.get('arm'))
        elif name == 'check_reach':
            if not co.hv2: raise ST.Refused('not available in this harness')
            out = co.check_reach(args.get('arm'), args.get('target_pose') or {})
        elif name == 'insert_charger':  # hv2e executor (explicit call only)
            if not isinstance(cfg.get('insert_charger'), dict): raise ST.Refused('not available in this harness')
            from . import hv2e_ext as _x
            out = _x.insert_charger(ad, co, t, args, cfg)
        elif name == 'check_insertion_reach':
            if co.info_only: raise ST.Refused('not available in this harness')
            out = co.reach_check(args.get('arm'))
        else:
            out = co.locate_outlets(t, args.get('observer_arm'), True if args.get('return_arm') is None else bool(args['return_arm']))
        out = dict(tool=name, privileged=False, **out, caveat=CAVEAT); ok = True
    except ST.Refused as e:
        out = dict(tool=name, privileged=False, status='refused', error=str(e)[:300], measured=dict(steps_used=t.steps)); ok = True
    except (ValueError, KeyError, TypeError, RuntimeError, OSError) as e:
        out = dict(tool=name, privileged=False, status='error', error='%s: %s' % (type(e).__name__, str(e)[:300]), measured=dict(steps_used=t.steps)); ok = False
    ST.assert_non_privileged(out)
    content = [dict(type='inputText', text=json.dumps(out))]
    if out.get('status') == 'ok' and isinstance(out['derived'].get('plug_targets'), (list, dict)):
        content.append(dict(type='inputText', text=RULE))
    if ad.packet is not None and name not in ('estimate_inhand_pose', 'check_insertion_reach', 'check_reach'):
        content += ad._content(ad.packet)
    return dict(success=ok, contentItems=content)


# v1.8 info_only tool descriptions (neutral; no workflow advice, no targets)
SPECS_INFO = [
    {'type': 'function', 'name': 'locate_outlets',
     'description': ('Locate the three outlets of the power strip to about a millimetre. The free arm (observer_arm, default: the arm that holds '
                     'nothing) is moved to an observation pose above the strip (wrist camera about 13 cm above it), a keypoint detector finds the '
                     'outlets in that wrist image and a model fit gives their 3-D positions; then the observer arm returns to where it was '
                     '(return_arm false leaves it there). It costs about 105 of the episode\'s 400 native control steps. Returns status '
                     '(ok / failed / refused / uncertain), measured {coarse, observation_moves, detections, steps_used} and derived {outlets_xyz_m, '
                     'strip_yaw_deg}. Offline test: median outlet error 0.15 mm. Non-privileged: RGB images, robot proprioception and the strip\'s '
                     'outlet layout only. ' + CAVEAT),
     'inputSchema': SPECS[0]['inputSchema']},
    {'type': 'function', 'name': 'estimate_inhand_pose',
     'description': ('Estimate the pose of the object held by one gripper relative to the gripper, from the current wrist and overhead camera '
                     'images (no motion, no control steps). The pose is also reported automatically once after a lift. If a post-lift estimate '
                     'exists, the reply also gives the measured change since then (shift in mm, rotation in degrees). Returns status, measured '
                     '{arm, step} and derived {inhand_pose (position / orientation in the TCP frame, uncertainty, member agreement), '
                     'change_since_post_lift}. ' + INFO_POSE_NOTE + '. Non-privileged. ' + CAVEAT),
     'inputSchema': SPECS[1]['inputSchema']},
]


# harness-v2 plug batch tool list (English, facts only; SKILL cards: experiments/harness-v2-20261028/skills)
SPECS_HV2 = [
    dict(SPECS_INFO[0], description=SPECS_INFO[0]['description'].replace("It costs about 105 of the episode\'s 400 native control steps.",
         "It costs at most 120 of the episode\'s 400 native control steps (measured 55-120) and abstains at 0 steps when the strip is not visible in the current overhead image, when the free hand is too close to the strip to be moved safely, or when the observation pose is not reachable.")),
    SPECS_INFO[1],
    {'type': 'function', 'name': 'check_reach',
     'description': ('Whether an end-effector (link6) pose can be reached by one arm from its current joints with the native controller. No motion, '
                     '0 control steps. Returns status, measured {arm, step, steps_used 0} and derived {reachable_from_current, controller_residual_mm, '
                     'controller_residual_deg, note}. Evidence: on fresh plug states, "not reachable" answers were 26/26 correct; "reachable" was '
                     'really reached about 60 % of the time. Objects, the table and the held object are not modelled. ' + CAVEAT),
     'inputSchema': {'type': 'object', 'properties': {
         'arm': {'type': 'string', 'enum': list(ARMS)},
         'target_pose': {'type': 'object', 'properties': {'position': {'type': 'array', 'items': {'type': 'number'}}, 'quaternion_wxyz': {'type': 'array', 'items': {'type': 'number'}}},
                         'required': ['position', 'quaternion_wxyz']},
         'reason': {'type': 'string'}}, 'required': ['arm', 'target_pose', 'reason'], 'additionalProperties': False}},
]
