"""Stage tools v2 (rgb-grasp-hint S1, 2026-10-25): PyRUA / RoboSimStudio-style primitives as Astra-callable tools, executed inside the
native adapter. Every physical step is a normal robodojo_act sub-call through NativeAdapter._invoke (same validation, guard, ACK and
ledger as m1_program sub-calls; call ids '<call>-stage-<k>'), so every control step counts against the episode budget.
Enabled only for aliases whose control/ablation.json entry has `stage_tools: {locate_url: ...}`.
v2 (RoboSimStudio interface ideas, Apache-2.0, design only - no code or simulator copied):
  * replies separate `measured` facts from `derived` judgements; `status` is ok / failed (tried, did not work) / refused (the question
    has no answer here, e.g. no position to grasp, or locate asked about a held object);
  * two motions: move_linear (Cartesian straight line, for insertion / pushing) and move_transfer (rise - travel - lower). RoboDojo's
    action contract is end-effector targets only (<= 5 cm / 0.35 rad per action): a true joint-space interpolation is NOT available,
    which the move_transfer description says;
  * set_gripper(object_width_m): close only to the object width minus 4 mm instead of all the way;
  * privilege boundary: every tool is non-privileged (public observation, robot proprioception incl. the measured finger joints, RGB
    images through the locate service); `assert_non_privileged` refuses any reply carrying simulator-truth keys.
  locate            no physical action: POSTs the current public observation (3 RGB images, 14D proprio, link6 poses) to the locate
                    service (SAM3 + PyRUA multi-view geometry, lane C), returns its answer.
  move_linear       bounded straight-line EEF steps toward a link6 target.
  move_transfer     rise to a clearance height, travel, lower; each segment a straight line.
  set_gripper       opening (0..1) or object_width_m; waits until the measured fingers stop; width and holding from the measured
                    finger joints (native `rgbhint_gripper` query, read-only, continuation entry wrapper).
  grasp_and_verify  top-down grasp + lift + check (finger gap; still detected at its old place on the table => did not follow)."""
import base64, json, math, urllib.request
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

NAMES = ('locate', 'move_linear', 'move_transfer', 'set_gripper', 'grasp_and_verify')
PRIVILEGED = {n: False for n in NAMES}
ARMS = ('left', 'right')
FINGER_PAD = 0.14; FINGERTIP = 0.1576; STEP_M = 0.049; STEP_RAD = 0.34; SERVO_POS = 0.005; SERVO_ROT = 0.05; SQUEEZE = 0.004
# commanded opening -> measured finger gap (m), empty gripper, calibrated offline 2026-10-25 (eval/stage-s1/gripper-calibration.json)
WIDTH_CAL = [(0.0, 0.0), (0.18, 0.0), (0.25, 0.007), (0.5, 0.034), (0.75, 0.061), (1.0, 0.088)]  # 0.1 -> 0.0 measured; 0.18 = linear extrapolation of the 0.25-1.0 points
CAVEAT = 'Returned values may be inaccurate; check them against the camera images yourself.'
BANNED = {'objects', 'object_poses', 'truth', 'success', 'reward', 'native_score', 'score', 'body_q', 'contacts'}
_POSE = {'position': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3, 'description': 'link6 target, metres, environment_origin'},
         'quaternion_wxyz': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 4, 'maxItems': 4, 'description': 'optional link6 target orientation'}}
_RET = ('Returns status (ok / failed / refused), phase (the segment where it stopped), measured {position_error_m, rotation_error_rad, '
        'achieved_pose, steps_used} and derived {reached, within_servo_tolerance (within 5 mm and 0.05 rad), stalled (the arm stopped improving '
        'before reaching: the pose may be outside the reachable workspace for that orientation)}, plus the new observation. ')
SPECS = [
    {'type': 'function', 'name': 'locate',
     'description': ('Locate an object in the current scene from the three RGB cameras (SAM3 segmentation + multi-view geometry; no depth sensor). '
                     'Non-privileged: uses only the images and robot proprioception. Returns status, measured {center_xyz (metres, environment_origin), '
                     'top_z, length_m, height_m, width_m, long_axis_deg (yaw of the long axis on the table), grasp_yaw_deg, views (camera views used; 1 = '
                     'single view, less reliable)} and derived {found, note, grippers_holding}. Small objects can be missed and long objects segmented only '
                     'in part. Positions are estimated for objects resting on the table: an object held in a gripper or lifted off the table is usually not '
                     'found, or its position is wrong (typically too low); if nothing is found while a gripper holds something, status is refused. A hand '
                     'right above an object can hide it from the overhead camera. If expected_size_m is given and the measured size is inconsistent with it, '
                     'no position is returned. No physical action, no control steps used. ' + CAVEAT),
     'inputSchema': {'type': 'object', 'properties': {
         'object': {'type': 'string', 'description': 'short English description, e.g. "the keyhole" or "the blue toy car"'},
         'expected_size_m': {'type': 'number', 'description': 'optional: your estimate of the object\'s longest dimension in metres'}},
         'required': ['object'], 'additionalProperties': False}},
    {'type': 'function', 'name': 'move_linear',
     'description': ('Move one arm\'s link6 along a Cartesian straight line to a target pose, as a sequence of bounded native EEF actions (each at most '
                     '5 cm / 0.35 rad, 5 control steps); the other arm is held still and both grippers keep their opening. For insertion, pushing and other '
                     'contact moves. orientation "lock" turns the wrist to quaternion_wxyz along the line; "keep" keeps the current wrist orientation. '
                     'Stops when reached, when the arm stops improving, or at max_steps. ' + _RET + 'Non-privileged. Uses native control steps. ' + CAVEAT),
     'inputSchema': {'type': 'object', 'properties': dict(_POSE, **{
         'arm': {'type': 'string', 'enum': list(ARMS)},
         'orientation': {'type': 'string', 'enum': ['lock', 'keep'], 'description': 'default: lock if quaternion_wxyz is given, else keep'},
         'max_steps': {'type': 'integer', 'minimum': 5, 'maximum': 100, 'description': 'control-step budget for this call (default 60)'},
         'tolerance_m': {'type': 'number', 'description': 'position tolerance for reached (default 0.01)'},
         'reason': {'type': 'string'}}),
         'required': ['arm', 'position', 'reason'], 'additionalProperties': False}},
    {'type': 'function', 'name': 'move_transfer',
     'description': ('Transfer one arm\'s link6 to a target pose in three straight segments: rise to clearance_m above the higher of the start and goal '
                     'heights, travel there (the wrist turns toward quaternion_wxyz during the travel), then lower onto the goal. For moving between places, '
                     'e.g. carrying an object over obstacles. Note: this robot interface accepts only end-effector targets (at most 5 cm / 0.35 rad per action), '
                     'so a joint-space interpolation is not available; every segment is a bounded Cartesian line. clearance_m 0 gives a single straight line. '
                     + _RET + 'Non-privileged. Uses native control steps. ' + CAVEAT),
     'inputSchema': {'type': 'object', 'properties': dict(_POSE, **{
         'arm': {'type': 'string', 'enum': list(ARMS)},
         'clearance_m': {'type': 'number', 'minimum': 0, 'maximum': 0.3, 'description': 'default 0.05'},
         'max_steps': {'type': 'integer', 'minimum': 5, 'maximum': 150, 'description': 'control-step budget for the whole transfer (default 100)'},
         'tolerance_m': {'type': 'number', 'description': 'position tolerance for reached (default 0.01)'},
         'reason': {'type': 'string'}}),
         'required': ['arm', 'position', 'reason'], 'additionalProperties': False}},
    {'type': 'function', 'name': 'set_gripper',
     'description': ('Command one gripper and wait until the fingers stop moving (at most 20 control steps); the arm pose is held. Give either opening '
                     '(0 closed, 1 fully open, about %.1f cm between the fingers) or object_width_m: then the fingers close only to the object width minus 4 mm '
                     'instead of all the way (closing all the way can squeeze a narrow object out of the grasp). Returns status, measured {width_m (gap between '
                     'the fingers from the finger joints), commanded_opening, steps_used} and derived {holding: the fingers stopped at least 2 mm wider than an empty gripper does for that command, i.e. '
                     'something is between them - can be a false positive if the fingers press on the table or another object}, plus the new observation. '
                     'Non-privileged. ' + CAVEAT) % (100 * WIDTH_CAL[-1][1]),
     'inputSchema': {'type': 'object', 'properties': {
         'arm': {'type': 'string', 'enum': list(ARMS)}, 'opening': {'type': 'number', 'minimum': 0, 'maximum': 1},
         'object_width_m': {'type': 'number', 'minimum': 0, 'maximum': 0.2}, 'reason': {'type': 'string'}},
         'required': ['arm', 'reason'], 'additionalProperties': False}},
    {'type': 'function', 'name': 'grasp_and_verify',
     'description': ('Top-down grasp of an object followed by a check: open, move above the object, descend (fingertips kept at least 5 mm above the '
                     'surface under the object), close, lift by lift_m, then measure the finger gap and re-locate the object. The fingers close across the '
                     'object\'s long axis. Position and yaw come from locate unless given; if neither is available the call is refused. Returns status, phase '
                     '(where it stopped: locate, approach, descend, close, lift), measured {grasp_pose, width_after_close_m, width_m, relocated_shift_m, steps_used} '
                     'and derived {holding (finger gap after the lift, and it did not shrink during the lift), object_followed (false if the object is still '
                     'detected at its original place on the table after the lift; null if this cannot be decided - a lifted object is usually not detected)}, '
                     'plus the new observation. One attempt per call; call again (e.g. another yaw_deg or depth_offset_m) to retry. Non-privileged. '
                     'Uses native control steps. ' + CAVEAT),
     'inputSchema': {'type': 'object', 'properties': {
         'arm': {'type': 'string', 'enum': list(ARMS)}, 'object': {'type': 'string'},
         'expected_size_m': {'type': 'number'},
         'position': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3, 'description': 'optional grasp centre; default from locate'},
         'yaw_deg': {'type': 'number', 'description': 'optional long-axis yaw of the object; default from locate'},
         'object_width_m': {'type': 'number', 'description': 'optional: close only to this width minus 4 mm'},
         'depth_offset_m': {'type': 'number', 'description': 'optional; negative grasps deeper'},
         'lift_m': {'type': 'number', 'description': 'lift height for the check (default 0.05)'},
         'reason': {'type': 'string'}},
         'required': ['arm', 'object', 'reason'], 'additionalProperties': False}},
]

rot = lambda q: R.from_quat([q[1], q[2], q[3], q[0]])
def wxyz(r):
    q = r.as_quat(); return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def opening_for_width(w):
    w = max(0.0, float(w)); cal = WIDTH_CAL[1:]; xs = [c[1] for c in cal]; ys = [c[0] for c in cal]  # strictly increasing widths
    return float(min(1.0, max(0.0, np.interp(w, xs, ys))))


def width_for_opening(o):
    return float(np.interp(o, [c[0] for c in WIDTH_CAL], [c[1] for c in WIDTH_CAL]))


def assert_non_privileged(v, path='reply'):
    if isinstance(v, dict):
        for k, x in v.items():
            if k in BANNED: raise RuntimeError('privileged key %r in %s' % (k, path))
            assert_non_privileged(x, path + '.' + k)
    elif isinstance(v, list):
        for x in v: assert_non_privileged(x, path)


class Refused(ValueError):
    """The question has no answer here (not a failed attempt)."""


class _Tool:
    def __init__(self, ad, name, call_id, on_ack, cfg):
        self.ad, self.name, self.call_id, self.on_ack, self.cfg, self.k, self.steps = ad, name, call_id, on_ack, cfg, 0, 0

    def eef(self, arm):
        e = self.ad.packet['current_eef'][arm]; return np.asarray(e['position'], float), list(e['quaternion_wxyz'])

    def measured_opening(self, arm):
        return float(self.ad.packet['current_state'][ARMS.index(arm) * 7 + 6])

    def opening(self, arm):
        """Gripper COMMAND to keep (v1.5 fix 2026-10-27, TAKEOVER-BUG-AUDIT.md): never echo the measured finger joint as a command (the
        simulator servos the continuous gripper_opening -> zero squeeze on a held object). Returns the last command this adapter sent for
        the arm; unknown (first call, e.g. right after Astra's own actions) -> inferred from the measured joint: < 0.9 = closed (Astra's
        native close command is 0.0; an open gripper reads >= 0.9), else the measured value (open)."""
        cmd = self.ad.__dict__.setdefault('_grip_cmd', {})
        if arm not in cmd:
            m = self.measured_opening(arm); cmd[arm] = 0.0 if m < 0.9 else m
        return cmd[arm]

    def terminal(self):
        return bool(self.ad.terminal_reason or self.ad.packet.get('rollout_finished'))

    def act(self, arm, pos, quat, opening, reason, steps=5):
        self.opening(arm); self.ad._grip_cmd[arm] = float(opening)  # v1.5: every sent opening is the remembered command for this arm
        pk = self.ad.packet; tgt = {}
        for a in ARMS:
            if a == arm:
                tgt[a] = dict(position=[float(x) for x in pos], quaternion_wxyz=[float(x) for x in quat], gripper_closed=bool(opening < 0.5), gripper_opening=float(opening))
            else:
                p, q = self.eef(a); o = self.opening(a)
                tgt[a] = dict(position=[float(x) for x in p], quaternion_wxyz=[float(x) for x in q], gripper_closed=bool(o < 0.5), gripper_opening=o)
        before = pk['step_id']; self.k += 1
        rep = self.ad._invoke('robodojo_act', dict(mode='eef', request_id=pk['request_id'], reason='[%s] %s' % (self.name, reason)[:400], steps=int(steps), target=tgt),
                              '%s-stage-%d' % (self.call_id, self.k), self.on_ack)
        if not rep.get('success'):
            raise RuntimeError('sub-action rejected: ' + ''.join(c.get('text', '') for c in rep.get('contentItems', []))[:300])
        self.steps += self.ad.packet['step_id'] - before
        return rep

    def width(self, arm):
        try:
            v = self.ad.client.sim.request('rgbhint_gripper'); return float(sum(abs(x) for x in v[arm]))
        except Exception:  # noqa: BLE001
            return None

    def holding_now(self):
        out = {}
        for a in ARMS:
            w = self.width(a); out[a] = bool(self.opening(a) < 0.5 and w is not None and w > 0.003)
        return out

    # ---- motion
    def line(self, arm, pos, quat, max_steps, tol, phase, reason):
        """Straight Cartesian line toward (pos, quat) in <= STEP_M / STEP_RAD waypoints measured from the CURRENT pose."""
        pos = np.asarray(pos, float); qn = np.asarray(quat, float); quat = list(qn / np.linalg.norm(qn)); start = self.steps; best = None; stall = 0; o = self.opening(arm)
        while self.steps - start < max_steps and not self.terminal():
            p, q = self.eef(arm); d = pos - p; dist = float(np.linalg.norm(d)); ang = float((rot(quat) * rot(q).inv()).magnitude())
            if dist <= tol and ang <= math.radians(3): break
            step_p = p + (d if dist <= STEP_M else d / dist * STEP_M)
            frac = 1.0 if ang <= STEP_RAD else STEP_RAD / ang; qs = wxyz(Slerp([0, 1], R.concatenate([rot(q), rot(quat)]))([frac])[0])
            self.act(arm, step_p, qs, o, '%s: %s' % (phase, reason))
            p2, q2 = self.eef(arm); e = (float(np.linalg.norm(pos - p2)), float((rot(quat) * rot(q2).inv()).magnitude()))
            if best is not None and e[0] > best[0] - 0.001 and e[1] > best[1] - 0.01: stall += 1
            else: stall = 0
            best = e if best is None or e[0] < best[0] else best
            if stall >= 3: break
        p, q = self.eef(arm); pe = float(np.linalg.norm(pos - p)); re = float((rot(quat) * rot(q).inv()).magnitude())
        reached = pe <= tol and re <= math.radians(3)
        return dict(phase=phase, measured=dict(position_error_m=round(pe, 4), rotation_error_rad=round(re, 4),
                                               achieved_pose=dict(position=[round(x, 4) for x in p.tolist()], quaternion_wxyz=[round(x, 5) for x in q]), steps_used=self.steps - start),
                    derived=dict(reached=bool(reached), within_servo_tolerance=bool(pe <= SERVO_POS and re <= SERVO_ROT), stalled=bool(not reached and stall >= 3)))

    @staticmethod
    def _status(r, done_phase=None):
        out = dict(status='ok' if r['derived']['reached'] else 'failed', **r)
        if r['derived']['reached'] and done_phase: out['phase'] = done_phase
        return out

    def move_linear(self, arm, pos, quat, orientation, max_steps, tol, reason):
        q0 = self.eef(arm)[1]
        if orientation is None: orientation = 'lock' if quat is not None else 'keep'
        if orientation == 'lock' and quat is None: raise ValueError('orientation lock needs quaternion_wxyz')
        q = list(quat) if orientation == 'lock' else q0
        return self._status(self.line(arm, pos, q, max_steps, tol, 'line', reason), 'done')

    def move_transfer(self, arm, pos, quat, clearance, max_steps, tol, reason):
        pos = np.asarray(pos, float); p0, q0 = self.eef(arm); q = list(quat) if quat is not None else q0; start = self.steps; segs = []
        if clearance <= 0:
            return self._status(self.line(arm, pos, q, max_steps, tol, 'travel', reason), 'done')
        z = max(p0[2], pos[2]) + clearance
        plan = [('rise', np.array([p0[0], p0[1], z]), q0, 0.01), ('travel', np.array([pos[0], pos[1], z]), q, 0.01), ('lower', pos, q, tol)]
        for phase, tp, tq, t in plan:
            left = max_steps - (self.steps - start)
            if left < 5: break
            r = self.line(arm, tp, tq, left, t, phase, reason); segs.append(dict(phase=phase, **{k: r[k] for k in ('measured', 'derived')}))
            if not r['derived']['reached']:
                out = dict(status='failed', phase=phase, measured=dict(r['measured'], steps_used=self.steps - start), derived=r['derived'], segments=segs); return out
        last = segs[-1] if segs else None
        if last is None or last['phase'] != 'lower':
            r = self.line(arm, pos, q, 0, tol, 'lower', reason)  # measure only (no budget left)
            return dict(status='failed', phase=(last or {}).get('phase', 'rise'), measured=dict(r['measured'], steps_used=self.steps - start), derived=r['derived'], segments=segs)
        return dict(status='ok', phase='done', measured=dict(last['measured'], steps_used=self.steps - start), derived=last['derived'], segments=segs)

    def set_gripper(self, arm, opening=None, object_width=None, reason='set_gripper'):
        if (opening is None) == (object_width is None): raise ValueError('give exactly one of opening or object_width_m')
        tw = None
        if object_width is not None:
            tw = max(0.0, float(object_width) - SQUEEZE); opening = opening_for_width(tw)
        opening = float(min(1.0, max(0.0, opening))); p, q = self.eef(arm); start = self.steps; prev = None; w = None
        for _ in range(4):
            if self.terminal(): break
            self.act(arm, p, q, opening, reason); w = self.width(arm)
            if w is not None and prev is not None and abs(w - prev) < 0.001: break
            prev = w
        cmd_w = width_for_opening(opening)
        holding = bool(w is not None and opening < 0.75 and w > cmd_w + 0.002)  # stopped wider than the empty-gripper width for this command
        return dict(status='ok', measured=dict(width_m=None if w is None else round(w, 4), commanded_opening=round(opening, 4), steps_used=self.steps - start),
                    derived=dict(holding=holding, commanded_width_m=round(cmd_w, 4)))

    def locate(self, obj, expected=None):
        if not str(obj).strip(): raise Refused('object description required')
        pk = self.ad.packet; imgs = {i['camera']: base64.b64encode(Path(i['path']).read_bytes()).decode() for i in pk['images']}
        req = dict(images_b64=imgs, states=pk['current_state'], eef_positions=[pk['current_eef'][a]['position'] for a in ARMS],
                   eef_quaternions_wxyz=[pk['current_eef'][a]['quaternion_wxyz'] for a in ARMS], object=obj, expected_size_m=expected)
        try:
            r = urllib.request.urlopen(urllib.request.Request(self.cfg['locate_url'], data=json.dumps(req).encode(), headers={'Content-Type': 'application/json'}), timeout=60)
            res = json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            return dict(status='failed', measured={}, derived=dict(found=False, note='locate service unavailable: %s' % type(e).__name__))
        held = self.holding_now(); found = bool(res.get('found'))
        meas = {k: res.get(k) for k in ('center_xyz', 'top_z', 'length_m', 'height_m', 'width_m', 'long_axis_deg', 'grasp_yaw_deg', 'views', 'raw_center_xyz')}
        note = res.get('note') or res.get('error')
        status = 'ok' if found else ('refused' if any(held.values()) else 'failed')
        if not found and any(held.values()):
            note = ((note + '; ') if note else '') + 'a gripper is holding something: locate does not work for held or lifted objects'
        return dict(status=status, measured=meas, derived=dict(found=found, note=note, grippers_holding=held))

    def grasp_and_verify(self, arm, obj, expected=None, position=None, yaw_deg=None, object_width=None, depth=0.0, lift=0.05, reason='grasp_and_verify'):
        start = self.steps; loc = None
        if position is None or yaw_deg is None:
            loc = self.locate(obj, expected)
            if loc['status'] != 'ok':
                return dict(status='refused', phase='locate', measured=dict(steps_used=0, locate=loc['measured']),
                            derived=dict(holding=False, object_followed=None, note='no position or yaw for the object: ' + str(loc['derived'].get('note'))))
            loc = loc['measured']
        c = np.asarray(position if position is not None else loc['center_xyz'], float)
        yaw = math.radians(yaw_deg if yaw_deg is not None else loc['long_axis_deg'])
        surface = (loc['top_z'] - (loc.get('height_m') or 0.0)) if loc and loc.get('top_z') is not None else c[2] - 0.01
        X = np.array([0, 0, -1.0]); Y = np.array([-math.sin(yaw), math.cos(yaw), 0.0]); Rm = np.column_stack([X, Y, np.cross(X, Y)])
        _, qnow = self.eef(arm)
        if (rot(wxyz(R.from_matrix(Rm))) * rot(qnow).inv()).magnitude() > math.pi / 2:  # the 180-deg-equivalent closer to the current wrist
            Rm = np.column_stack([X, -Y, np.cross(X, -Y)])
        q = wxyz(R.from_matrix(Rm)); tcp = c.copy(); tcp[2] += depth
        l6 = tcp - Rm @ np.array([FINGER_PAD, 0, 0]); l6[2] = max(l6[2], surface + FINGERTIP + 0.005)
        meas = dict(grasp_pose=dict(position=[round(x, 4) for x in l6.tolist()], quaternion_wxyz=[round(x, 5) for x in q]), yaw_deg=round(math.degrees(yaw), 1))
        self.set_gripper(arm, opening=1.0, reason=reason + ': open')
        m = self.line(arm, l6 + np.array([0, 0, 0.06]), q, 60, 0.01, 'approach', reason)
        if not m['derived']['reached'] and m['measured']['position_error_m'] > 0.03:
            meas.update(steps_used=self.steps - start, approach=m['measured'])
            return dict(status='failed', phase='approach', measured=meas, derived=dict(holding=False, object_followed=None, note='pre-grasp pose not reached'))
        m = self.line(arm, l6, q, 30, 0.01, 'descend', reason)
        g = self.set_gripper(arm, opening=0.0 if object_width is None else None, object_width=object_width, reason=reason + ': close'); w0 = g['measured']['width_m']
        p, qq = self.eef(arm); self.line(arm, p + np.array([0, 0, lift]), qq, 25, 0.01, 'lift', reason)
        cw = g['derived']['commanded_width_m']; w = self.width(arm)
        holding = bool(w is not None and w > cw + 0.002 and (w0 is None or w >= w0 - 0.002))  # wider than an empty gripper for this command, not shrinking
        followed = None; moved = None
        if loc is not None and loc.get('center_xyz') is not None:
            after = self.locate(obj, expected)
            if after['derived']['found']:
                moved = float(np.linalg.norm(np.asarray(after['measured']['center_xyz']) - np.asarray(loc['center_xyz'])))
                if moved < 0.03: followed = False
        meas.update(width_after_close_m=w0, width_m=None if w is None else round(w, 4), relocated_shift_m=None if moved is None else round(moved, 4), steps_used=self.steps - start)
        ok = holding and followed is not False
        return dict(status='ok' if ok else 'failed', phase='done' if ok else ('close' if not holding else 'lift'), measured=meas,
                    derived=dict(holding=holding, object_followed=followed,
                                 note=None if ok else ('nothing held: after closing and lifting, the finger gap is no wider than an empty gripper (or it shrank during the lift)' if not holding else 'object still on the table after the lift')))


def run(ad, name, args, call_id, on_ack, cfg):
    """Execute one stage tool; returns the public reply {success, contentItems}."""
    t = _Tool(ad, name, call_id, on_ack, cfg); physical = name != 'locate'
    try:
        if ad.packet is None: raise Refused('call robodojo_start first')
        r = str(args.get('reason') or name)
        if name == 'locate':
            out = t.locate(str(args['object']), args.get('expected_size_m'))
        elif name == 'move_linear':
            out = t.move_linear(args['arm'], args['position'], args.get('quaternion_wxyz'), args.get('orientation'), int(args.get('max_steps') or 60), float(args.get('tolerance_m') or 0.01), r)
        elif name == 'move_transfer':
            out = t.move_transfer(args['arm'], args['position'], args.get('quaternion_wxyz'), float(0.05 if args.get('clearance_m') is None else args['clearance_m']),
                                  int(args.get('max_steps') or 100), float(args.get('tolerance_m') or 0.01), r)
        elif name == 'set_gripper':
            out = t.set_gripper(args['arm'], args.get('opening'), args.get('object_width_m'), r)
        else:
            out = t.grasp_and_verify(args['arm'], str(args['object']), args.get('expected_size_m'), args.get('position'), args.get('yaw_deg'), args.get('object_width_m'),
                                     float(args.get('depth_offset_m') or 0.0), float(args.get('lift_m') or 0.05), r)
        out = dict(tool=name, privileged=PRIVILEGED[name], **out, caveat=CAVEAT); ok = True
    except Refused as e:
        out = dict(tool=name, privileged=PRIVILEGED[name], status='refused', error=str(e)[:300], measured=dict(steps_used=t.steps)); ok = True
    except (ValueError, KeyError, TypeError, RuntimeError) as e:
        out = dict(tool=name, privileged=PRIVILEGED[name], status='error', error='%s: %s' % (type(e).__name__, str(e)[:300]), measured=dict(steps_used=t.steps)); ok = False
    assert_non_privileged(out)
    content = [dict(type='inputText', text=json.dumps(out))]
    if physical and ad.packet is not None:
        content += ad._content(ad.packet)
    return dict(success=ok, contentItems=content)
