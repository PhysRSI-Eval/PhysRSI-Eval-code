"""combined-online v1.4 insertion reach pre-check (copied into the release as rsi/reach_model.py).
Deploy-time robot knowledge only: URDF chain (X5A.urdf, sha256 checked), soft joint limits and arm base poses (robot_kin.json, robot
model constants dumped once from the simulator, like a calibration), measured joints (14-D state). No object truth.
ArmFK = verbatim copy of physrsi_eval.sim_server.ArmFK (the native controller: every control tick ONE bounded damped-least-squares
step from the measured joints toward the link6 target; task error capped 2 cm / 0.1 rad, joint step 0.05 rad, clipped to the soft
limits). Reach / precheck = plug-offline-eval/v14/src/reach.py (prototype used in plug-retest p2), plus the pre-turn pose in the reply.
Iterating the controller step from the current joints predicts where the arm ends for a fixed target; residual > tol = not reachable."""
import hashlib, json, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

KIN = json.loads(Path(__file__).with_name('robot_kin.json').read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def transform(position, quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, float); t = np.eye(4); t[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix(); t[:3, 3] = position; return t


def q_of(M):
    q = Rotation.from_matrix(M[:3, :3]).as_quat(); return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


class ArmFK:  # verbatim logic of physrsi_eval.sim_server.ArmFK
    def __init__(self, urdf, joint_names, base='base_link', tip='link6'):
        self.names = list(joint_names)
        edges = {j.find('child').get('link'): j for j in ET.parse(urdf).getroot().findall('joint')}
        self.chain = []
        while tip != base:
            j = edges[tip]
            origin = j.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            t = np.eye(4)
            t[:3, :3], t[:3, 3] = Rotation.from_euler('xyz', rpy).as_matrix(), xyz
            axis = j.find('axis')
            axis = np.fromstring(axis.get('xyz'), sep=' ') if axis is not None else np.array([1., 0, 0])
            self.chain.insert(0, (j.get('name'), j.get('type'), t, axis))
            tip = j.find('parent').get('link')
        moving = [n for n, kind, _, _ in self.chain if kind != 'fixed']
        require(set(moving) == set(self.names), f'URDF chain/arm joint mismatch: {moving} vs {self.names}')

    def matrix(self, q):
        values, t = dict(zip(self.names, np.asarray(q, float))), np.eye(4)
        for name, kind, origin, axis in self.chain:
            motion = np.eye(4)
            if kind in ('revolute', 'continuous'):
                motion[:3, :3] = Rotation.from_rotvec(axis * values[name]).as_matrix()
            t = t @ origin @ motion
        return t

    def bounded_step(self, q, limits, root, target):
        q = np.asarray(q, float)
        now = root @ self.matrix(q)
        goal = transform(target['position'], target['quaternion_wxyz'])
        bounded = lambda x, cap: x * min(1., cap / max(np.linalg.norm(x), 1e-12))
        error = np.r_[bounded(goal[:3, 3] - now[:3, 3], .02), bounded(Rotation.from_matrix(goal[:3, :3] @ now[:3, :3].T).as_rotvec(), .1)]
        jacobian = np.zeros((6, len(q)))
        for i in range(len(q)):
            shifted = q.copy(); shifted[i] += 1e-5
            nxt = root @ self.matrix(shifted)
            jacobian[:3, i] = (nxt[:3, 3] - now[:3, 3]) / 1e-5
            jacobian[3:, i] = Rotation.from_matrix(nxt[:3, :3] @ now[:3, :3].T).as_rotvec() / 1e-5
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .05 ** 2 * np.eye(6), error)
        limits = np.asarray(limits, float)
        low, high = np.maximum(limits[:, 0], q - .05), np.minimum(limits[:, 1], q + .05)
        require(np.all(low <= high), 'joint outside its valid interval')
        return np.clip(q + delta, low, high).astype(np.float32)


class Reach:
    def __init__(self, kin=None):
        kin = KIN if kin is None else kin
        h = hashlib.sha256(Path(kin['urdf']).read_bytes()).hexdigest()
        require(h == kin['urdf_sha256'], 'robot URDF differs from the frozen one')
        self.fk = {a: ArmFK(kin['urdf'], k['joints'], k.get('base', 'base_link'), k.get('tip', 'link6')) for a, k in kin['arms'].items()}
        self.lim = {a: np.asarray(k['limits'], float) for a, k in kin['arms'].items()}
        self.root = {a: np.asarray(k['root'], float) for a, k in kin['arms'].items()}

    def pose(self, arm, q):
        return self.root[arm] @ self.fk[arm].matrix(q)

    def go(self, arm, q0, M, ticks=300, tol=(0.003, np.radians(3))):
        q = np.asarray(q0, float); goal = dict(position=M[:3, 3], quaternion_wxyz=q_of(M))
        for t in range(ticks):
            q = np.asarray(self.fk[arm].bounded_step(q, self.lim[arm], self.root[arm], goal), float)
            if t % 10 == 9:
                P = self.pose(arm, q); pe = float(np.linalg.norm(P[:3, 3] - M[:3, 3])); re = float((Rotation.from_matrix(P[:3, :3]) * Rotation.from_matrix(M[:3, :3]).inv()).magnitude())
                if pe < tol[0] * 0.5 and re < tol[1] * 0.5: break
        P = self.pose(arm, q); pe = float(np.linalg.norm(P[:3, 3] - M[:3, 3])); re = float((Rotation.from_matrix(P[:3, :3]) * Rotation.from_matrix(M[:3, :3]).inv()).magnitude())
        return q, pe, float(np.degrees(re)), bool(pe <= tol[0] and re <= tol[1])

    def path(self, arm, q0, targets, ticks=300):
        q = q0; out = []
        for M in targets:
            q, pe, re, ok = self.go(arm, q, M, ticks); out.append((round(pe * 1000, 1), round(re, 1), ok))
        return out, q


def precheck(reach, arm, q_now, L6_now, G_obj_in_link6, seat_poses, hover_m=0.04):
    """For each outlet's seated charger pose (world 4x4): link6 hover / seated targets = pose . inv(G), predicted from the CURRENT
    joints with the controller model. verdict: 'reachable' | 'turn needed' (reachable only after an in-place wrist pre-turn to the hover
    orientation 5 cm above the current position) | 'suggest regrasp' (no outlet reachable)."""
    Ginv = np.linalg.inv(np.asarray(G_obj_in_link6, float)); res = []
    for k, Ts in enumerate(seat_poses):
        Hs = np.asarray(Ts, float) @ Ginv; Hh = Hs.copy(); Hh[2, 3] += hover_m
        (h, s), _ = reach.path(arm, q_now, [Hh, Hs])
        turn = float(np.degrees((Rotation.from_matrix(Hh[:3, :3]) * Rotation.from_matrix(L6_now[:3, :3]).inv()).magnitude()))
        res.append(dict(outlet=k, hover_reachable=h[2], hover_residual_mm=h[0], seated_reachable=s[2], seated_residual_mm=s[0], wrist_turn_deg=round(turn, 1)))
    ok = [r for r in res if r['hover_reachable'] and r['seated_reachable']]
    pre_turn = None
    if ok:
        verdict = 'reachable'
    else:
        verdict = 'suggest regrasp'
        for r, Ts in zip(res, seat_poses):
            Hs = np.asarray(Ts, float) @ Ginv; Hh = Hs.copy(); Hh[2, 3] += hover_m
            Mt = Hh.copy(); Mt[:3, 3] = L6_now[:3, 3] + np.array([0, 0, 0.05])
            (t, h, s), _ = reach.path(arm, q_now, [Mt, Hh, Hs])
            if t[2] and h[2] and s[2]:
                verdict = 'turn needed'; r['reachable_after_pre_turn'] = True
                pre_turn = dict(outlet=r['outlet'], link6=dict(position=[round(float(v), 4) for v in Mt[:3, 3]], quaternion_wxyz=[round(v, 5) for v in q_of(Mt)]))
                break
    return dict(verdict=verdict, reachable_outlets=[r['outlet'] for r in ok], outlets=res, pre_turn=pre_turn)
