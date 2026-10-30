"""finepose shared library (host3 lizekai-c9, 2026-10-28) for experiments B-F (PLAN-BCDEF.md).
Frozen-able pieces that should transfer to a second task (plug_in_charger) with only an OBJECT DESCRIPTION change:
  - target_hand(): generic pose computation  link6_goal = T_world_objgoal . inv(G_object_in_link6)
  - Env.move(): the common action interface (dx, dy, dz in world mm, dyaw about the vertical through the TCP) used by scripts AND Astra
  - Env.move_to(): absolute link6 target with tracking-error report
  - Env.grip_check(): in-hand relation change vs a reference
Truth is read ONLY by Env.truth() / Env.log() for analysis, and by the 'truth' perception mode; decisions in reference scripts use
proprioception + whatever perception mode the experiment states."""
import json, math, shutil, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/lizekai/RSIws/experiments/regrasp-20261027/env/pkg'); sys.path.insert(0, '/lizekai/RSIws/experiments/regrasp-20261027/src')
import rg_sim as pe_sim  # noqa: E402
from rsi import combined_tools as CT, stage_tools as ST  # noqa: E402
TCP = np.array([0.14, 0.0, 0.0])
# ---- object description: insert_key (the only task-specific block; plug_in_charger gets its own)
KEY = dict(task='insert_key', obj_prefix='key', target_prefix='slot', eval_axis=[0.5, -math.sqrt(3) / 2, 0.0], keyway_yaw_deg=60.0, seat_depth=0.035,
           grasp=dict(beta=0.0, roll=90.0, d=0.025), turn_deg=-75.0,  # v3: -75 (the key lags ~half of the commanded turn)
           checks=[['is_A_depth_in_B', dict(label_A='key', label_B='slot', z_threshold=0.025)], ['is_AB_xy_distance_within_threshold', dict(label_A='key', label_B='slot', threshold=0.007)],
                   ['is_axis_up', dict(label='key', axis=[0, 0, 1], threshold=7)], ['is_axis_aligned', dict(label_A='key', axis_A=[1, 0, 0], label_B='slot', axis_B=[0.5, -math.sqrt(3) / 2, 0.0], align_threshold=30)]])


def rot_deg(A, B): return float(np.degrees((R.from_matrix(A[:3, :3]) * R.from_matrix(B[:3, :3]).inv()).magnitude()))
def yaw_of(M): return math.degrees(math.atan2(M[1, 0], M[0, 0]))
def wrap(d, period=360.0): return (d + period / 2) % period - period / 2


def grasp_in_obj(beta, roll, d):
    """link6 pose in the object frame (key_run family): approach = cos(beta)(-z) + sin(beta)(-y), TCP at +d on the long axis."""
    av = math.cos(math.radians(beta)) * np.array([0, 0, -1.0]) + math.sin(math.radians(beta)) * np.array([0, -1.0, 0])
    ref = np.array([0, 1.0, 0]) if abs(av[1]) < 0.9 else np.array([1.0, 0, 0]); yv = np.cross(ref, av); yv /= np.linalg.norm(yv); zv = np.cross(av, yv)
    Rl = np.column_stack([av, yv, zv]) @ R.from_euler('x', roll, degrees=True).as_matrix(); H = np.eye(4); H[:3, :3] = Rl; H[:3, 3] = np.array([0, 0, d]) - Rl @ TCP; return H


def target_hand(T_obj_goal, G_obj_in_link6):
    """Generic pose computation tool: where link6 must be so that the held object reaches T_obj_goal."""
    return T_obj_goal @ np.linalg.inv(G_obj_in_link6)


class Env:
    def __init__(self, out, gpu, desc=KEY, arm='right', extend=900, scratch='/root/fp-scratch'):  # v4: local disk (NAS copy of the 2.5 GB render cache was the bottleneck)
        self.out = Path(out); self.out.mkdir(parents=True, exist_ok=True); (self.out / 'img').mkdir(exist_ok=True); self.D = desc; self.arm = arm; self.close = 0.0
        self.simdir = Path(scratch) / ('KEY-%s' % self.out.name); self.sim = pe_sim.RA.Sim(desc['task'], self.simdir, gpu); self.rpc = self.sim.rpc.request
        self.rpc('reset', layout_id=0); self.rpc('extend_limit', extra=int(extend)); objs = self.rpc('objects')
        self.kn = next(k for k in objs if k.startswith(desc['obj_prefix']) and desc['target_prefix'] not in k); self.sn = next(k for k in objs if desc['target_prefix'] in k)
        self.ad = pe_sim.FakeAdapter(self.sim, self.simdir / 'obs'); self.tool = ST._Tool(self.ad, 'fp', 'fp', None, {})
        g = self.rpc('rgbhint_geom', bbox=[self.sn, self.kn])['bbox']; self.Ts = CT.T_of(g[self.sn]['pose'][:3], g[self.sn]['pose'][3:]); self.top = g[self.sn]['z_max']
        LV = np.asarray(g[self.kn]['local_vertices']); self.cl = (LV.max(0) + LV.min(0)) / 2; self.half = float((LV.max(0) - LV.min(0))[2]) / 2
        self.events = []; self.G_ref = None; self.t0 = time.time()

    # ---- truth (analysis + 'truth' perception mode)
    def T_obj(self):
        o = self.rpc('objects')[self.kn]; T = CT.T_of(o['position'], o['quaternion_wxyz']); K = T.copy(); K[:3, 3] = T[:3, :3] @ self.cl + T[:3, 3]; return K  # bbox-centre frame
    def T_link6(self): return CT.T_of(*self.tool.eef(self.arm))
    def G(self): return np.linalg.inv(self.T_link6()) @ self.T_obj()
    def slot_goal(self, hover=0.0, eyaw=None):
        """object (bbox-centre) goal frame: seated, long axis up, x along the keyway (= eval axis rotated by keyway_yaw), raised by hover."""
        ax = self.Ts[:3, :3] @ np.asarray(self.D['eval_axis']); yaw = math.atan2(ax[1], ax[0]) + math.radians(self.D['keyway_yaw_deg'] if eyaw is None else eyaw)
        T = np.eye(4); T[:3, :3] = R.from_euler('z', yaw).as_matrix(); T[:3, 3] = [self.Ts[0, 3], self.Ts[1, 3], self.top - self.D['seat_depth'] + self.half + hover]; return T
    def truth(self):
        K = self.T_obj(); S = self.slot_goal(); L = self.T_link6(); G = np.linalg.inv(L) @ K
        d = dict(step=int(self.ad.packet['step_id']), link6=[round(float(x), 4) for x in L[:3, 3]],
                 key_vs_slot_xy_mm=[round(float(v) * 1000, 2) for v in (K[:2, 3] - S[:2, 3])],
                 key_yaw_vs_keyway_deg=round(wrap(yaw_of(K) - yaw_of(S), 180.0), 2), key_yaw_vs_keyway_deg360=round(wrap(yaw_of(K) - yaw_of(S), 360.0), 2), key_yaw_vs_evalaxis_deg=round(wrap(yaw_of(K) - yaw_of(S) + self.D['keyway_yaw_deg'], 180.0), 2),
                 key_tilt_deg=round(math.degrees(math.acos(max(-1.0, min(1.0, float(K[2, 2]))))), 2),
                 insertion_depth_mm=round((self.top - (K[2, 3] - self.half)) * 1000, 2))
        if self.G_ref is not None: d.update(inhand_shift_mm=round(float(np.linalg.norm(G[:3, 3] - self.G_ref[:3, 3])) * 1000, 2), inhand_rot_deg=round(rot_deg(G, self.G_ref), 2))
        return d
    def log(self, tag, **kw):
        e = dict(tag=tag, **self.truth(), **kw); self.events.append(e); (self.out / 'events.json').write_text(json.dumps(self.events, indent=0)); return e
    def snap(self, tag):
        for im in self.ad.packet.get('images', []): shutil.copy(im['path'], self.out / 'img' / ('%s-%s.png' % (tag, im['camera'])))
        return {im['camera']: str(self.out / 'img' / ('%s-%s.png' % (tag, im['camera']))) for im in self.ad.packet.get('images', [])}
    def checks(self): return pe_sim.RA.plain(self.rpc('rgbhint_check', checks=self.D['checks']))
    def terminal(self): return self.tool.terminal()

    # ---- action interface (shared by scripts and Astra)
    def move_to(self, L, tag, max_steps=150, tol=0.001):
        r = self.tool.line(self.arm, L[:3, 3], CT.q_of(L), max_steps, tol, tag, 'fp'); A = self.T_link6()
        return self.log(tag, cmd_link6=[round(float(x), 4) for x in L[:3, 3]], track_pos_err_mm=round(float(np.linalg.norm(A[:3, 3] - L[:3, 3])) * 1000, 2), track_rot_err_deg=round(rot_deg(A, L), 2))
    def move(self, dx_mm=0.0, dy_mm=0.0, dz_mm=0.0, dyaw_deg=0.0, tag='move', steps=3, src='script'):
        dx, dy, dz = [max(-5.0, min(5.0, float(v))) / 1000 for v in (dx_mm, dy_mm, dz_mm)]; dyw = max(-15.0, min(15.0, float(dyaw_deg)))
        L0 = self.T_link6(); tcp = L0[:3, :3] @ TCP + L0[:3, 3]; Rz = R.from_euler('z', dyw, degrees=True).as_matrix()
        L = np.eye(4); L[:3, :3] = Rz @ L0[:3, :3]; L[:3, 3] = tcp + Rz @ (L0[:3, 3] - tcp) + np.array([dx, dy, dz])
        self.tool.act(self.arm, L[:3, 3], CT.q_of(L), self.close, '%s:%s' % (src, tag), steps=steps); A = self.T_link6()
        return self.log(tag, src=src, cmd=dict(dx_mm=round(dx * 1000, 2), dy_mm=round(dy * 1000, 2), dz_mm=round(dz * 1000, 2), dyaw_deg=round(dyw, 1)),
                        track_pos_err_mm=round(float(np.linalg.norm(A[:3, 3] - L[:3, 3])) * 1000, 2), track_rot_err_deg=round(rot_deg(A, L), 2))

    # ---- setups
    def setup_held(self, eyaw=None, ex=0.0, ey=0.0, etilt=0.0, hover=0.045):
        """dev setup (truth): official grasp, key pinned into the open fingers at the hover pose (+ deliberate error)."""
        gp = self.D['grasp']; H = grasp_in_obj(gp['beta'], gp['roll'], gp['d']); Kh = self.slot_goal(hover, eyaw)
        Kh[:3, :3] = Kh[:3, :3] @ R.from_euler('x', etilt, degrees=True).as_matrix(); Kh[:3, 3] += np.array([ex / 1000, ey / 1000, 0])
        L = target_hand(Kh, np.linalg.inv(H))  # H = link6 in the object frame -> G = inv(H)
        self.tool.set_gripper(self.arm, opening=1.0, reason='open'); pre = L.copy(); pre[2, 3] += 0.08
        self.tool.line(self.arm, pre[:3, 3], CT.q_of(pre), 200, 0.003, 'pre', 'fp'); e = self.move_to(L, 'setup_hover', 80)
        if e['track_pos_err_mm'] > 2: return False
        L6 = self.T_link6(); Kp = L6 @ np.linalg.inv(H); Ro = Kp[:3, :3]; po = Kp[:3, 3] - Ro @ self.cl; q = R.from_matrix(Ro).as_quat(); qw = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
        p, qq = self.tool.eef(self.arm)
        for _ in range(12): self.rpc('place_obj', inst=self.kn, position=[float(x) for x in po], quaternion_wxyz=qw); self.tool.act(self.arm, p, qq, self.close, 'close-pin', steps=1)
        self.tool.act(self.arm, p, qq, self.close, 'settle', steps=5); self.G_ref = self.G(); self.log('held'); return True

    # ---- reference controllers (decisions from proprio + the stated perception; truth only logged)
    def settle(self, steps=10):
        p, q = self.tool.eef(self.arm)
        for _ in range(max(1, steps // 5)): self.tool.act(self.arm, p, q, self.close, 'settle', steps=5)  # act accepts 1..5 steps
        return self.log('settled')
    goal_err = None  # TOL: 4x4 error applied to the object goal (world translation, rotation about the goal centre) = combined perception bias
    def goal(self, hover=0.045):
        T = self.slot_goal(hover)
        if self.goal_err is not None: E = self.goal_err; T = T.copy(); T[:3, :3] = E[:3, :3] @ T[:3, :3]; T[:3, 3] = T[:3, 3] + E[:3, 3]
        return T
    def realign(self, perc='truth', hover=0.045, tag='realigned'):
        G = self.G() if perc == 'truth' else perc(self)  # perc: callable returning the in-hand G estimate (experiment F)
        return self.move_to(target_hand(self.goal(hover), G), tag, 150)
    def ref_insert(self, perc='truth', retries=2, src='ref'):
        """v3 reference: settle -> re-measure + realign -> straight descent; if link6 went down less than 38 mm (proprio: not seated; full stroke = 45 mm),
        lift 3 mm, settle, realign at the current height and retry."""
        for t in range(retries + 1):
            self.settle(); z0 = self.T_link6()[2, 3]; self.realign(perc, tag='realign_try%d' % t); z1 = self.T_link6()[2, 3]
            self.ref_descend(src=src); drop = (z1 - self.T_link6()[2, 3]) * 1000
            self.log('descend_end', try_=t, link6_drop_mm=round(drop, 1))
            if drop >= 38 or self.terminal(): return t  # v3.1: hover is 45 mm above the seat; depth >= 25 mm needs ~35 mm of drop
            for _ in range(int(drop // 5) + 1): self.move(dz_mm=5, tag='retry_lift', src=src)
        return -1
    def ref_descend(self, depth_mm=None, step_mm=3.0, src='ref'):
        """straight descent through the common interface until link6 has gone down hover+seat or stops following (proprio stall)."""
        z0 = self.T_link6()[2, 3]; goal = (depth_mm or 52.0) / 1000; n = 0
        while self.T_link6()[2, 3] > z0 - goal and not self.terminal() and n < 40:
            e = self.move(dz_mm=-step_mm, tag='descend', src=src); n += 1
            if e['track_pos_err_mm'] > 0.5: break  # v2: the arm no longer follows (seated or blocked) -> stop pushing
        return n
    def ref_turn(self, deg=None, step=5.0, src='ref'):  # v2: 5 deg steps (15 deg steps let the key lag ~30 deg in the fingers)
        deg = self.D['turn_deg'] if deg is None else deg; n = int(math.ceil(abs(deg) / step))
        for _ in range(n):
            if self.terminal(): break
            self.move(dyaw_deg=deg / n, tag='turn', src=src)
    # ---- J (user 10-28): start right after the two-hand handover; 6-DoF delta moves for Astra/scripts
    HAND = np.array([-0.075, -0.185, 0.935])  # handover key-centre position (mirror of official ep 0)
    def setup_handover(self, seed=-1):
        """dev setup (truth): key pinned into the RIGHT fingers with the official grasp (beta 0, bow-face pinch) at the handover pose:
        key horizontal, bow toward +x (toward the right arm), thickness vertical; seed >= 0 adds key-centre noise U[-10, 10] mm and yaw U[-10, 10] deg."""
        Kh = np.eye(4); Kh[:3, :3] = np.column_stack([[0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0]]); Kh[:3, 3] = self.HAND
        if seed >= 0:
            rng = np.random.default_rng(seed); Kh[:3, 3] += rng.uniform(-0.01, 0.01, 3); Kh[:3, :3] = R.from_euler('z', rng.uniform(-10, 10), degrees=True).as_matrix() @ Kh[:3, :3]
        gp = self.D['grasp']; self.tool.set_gripper(self.arm, opening=1.0, reason='open')
        for rr in (gp['roll'], -gp['roll']):  # same bow-face pinch, pick the reachable roll
            H = grasp_in_obj(gp['beta'], rr, gp['d']); L = Kh @ H; pre = L.copy(); pre[:3, 3] -= 0.07 * L[:3, 0]
            e = self.move_to(pre, 'setup_pre_roll%d' % rr, 250, 0.002)
            if e['track_pos_err_mm'] < 3 and e['track_rot_err_deg'] < 3: break
        e = self.move_to(L, 'setup_handover', 120)
        if e['track_pos_err_mm'] > 2 or e['track_rot_err_deg'] > 3: return False
        L6 = self.T_link6(); Kp = L6 @ np.linalg.inv(H); Ro = Kp[:3, :3]; po = Kp[:3, 3] - Ro @ self.cl; q = R.from_matrix(Ro).as_quat(); qw = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
        p, qq = self.tool.eef(self.arm)
        for _ in range(12): self.rpc('place_obj', inst=self.kn, position=[float(x) for x in po], quaternion_wxyz=qw); self.tool.act(self.arm, p, qq, self.close, 'close-pin', steps=1)
        self.tool.act(self.arm, p, qq, self.close, 'settle', steps=5); self.G_ref = self.G(); self.handover_roll = rr; self.log('held'); return True
    def move6(self, dx_mm=0.0, dy_mm=0.0, dz_mm=0.0, droll_deg=0.0, dpitch_deg=0.0, dyaw_deg=0.0, tag='move6', steps=3, src='script', lim_mm=20.0):
        """world-frame delta about the TCP: translation (clamped lim_mm) + rotation Rz(dyaw) Ry(dpitch) Rx(droll) about world axes (each clamped 15 deg)."""
        t = [max(-lim_mm, min(lim_mm, float(v))) / 1000 for v in (dx_mm, dy_mm, dz_mm)]; a = [max(-15.0, min(15.0, float(v))) for v in (droll_deg, dpitch_deg, dyaw_deg)]
        L0 = self.T_link6(); tcp = L0[:3, :3] @ TCP + L0[:3, 3]; Rd = R.from_euler('ZYX', [a[2], a[1], a[0]], degrees=True).as_matrix()
        L = np.eye(4); L[:3, :3] = Rd @ L0[:3, :3]; L[:3, 3] = tcp + Rd @ (L0[:3, 3] - tcp) + np.array(t)
        n = max(1, int(math.ceil(max(abs(x) for x in t) / 0.005)), int(math.ceil(max(abs(x) for x in a) / 5.0)))  # <= 5 mm / 5 deg per sim waypoint
        for i in range(1, n + 1):
            f = i / n; Ri = R.from_euler('ZYX', [a[2] * f, a[1] * f, a[0] * f], degrees=True).as_matrix(); P = tcp + Ri @ (L0[:3, 3] - tcp) + np.array(t) * f
            Li = np.eye(4); Li[:3, :3] = Ri @ L0[:3, :3]; Li[:3, 3] = P; self.tool.act(self.arm, Li[:3, 3], CT.q_of(Li), self.close, '%s:%s' % (src, tag), steps=steps)
        A = self.T_link6()
        return self.log(tag, src=src, cmd=dict(dx_mm=round(t[0] * 1000, 1), dy_mm=round(t[1] * 1000, 1), dz_mm=round(t[2] * 1000, 1), droll_deg=round(a[0], 1), dpitch_deg=round(a[1], 1), dyaw_deg=round(a[2], 1)),
                        track_pos_err_mm=round(float(np.linalg.norm(A[:3, 3] - L[:3, 3])) * 1000, 2), track_rot_err_deg=round(rot_deg(A, L), 2))
    def close_sim(self):
        try: self.sim.close()
        finally: shutil.rmtree(self.simdir, ignore_errors=True)
