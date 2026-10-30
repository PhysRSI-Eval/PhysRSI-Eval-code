"""self_coll.py (C1, 2026-10-30): self-collision-aware reach for the X5 arms.
Finding (c1/tools/self_coll_stalls.py on census v2): 241/297 simulator stalls ('controller model converges, the sim arm stops') have
arm SELF-penetration in the cuRobo sphere model (link3-link5 229, link4-link6 52; median -6.8 mm) vs 30/300 random joints from the same
ranges (median +3.4 mm). The native-controller model (reach_model v15) ignores self-collision, so it calls these targets reachable.
SelfReach.trace_ok(arm, q0, M): native controller trace (reach_cu.controller_trace) toward M, sub-sampled; ok if it converges AND the
minimum self-distance along it stays >= margin. Deploy-legal: robot model + proprioception only."""
import importlib.util as ilu
import numpy as np

_sp = ilu.spec_from_file_location('reach_cu_v2', '/lizekai/RSIws/experiments/macro-tools-20261027/pkg-v2/rsi/reach_cu.py')
RC = ilu.module_from_spec(_sp); _sp.loader.exec_module(RC)


class SelfReach:
    def __init__(self, width=0.0, margin=0.0):
        self.cr = RC.CollisionReach(width=width); k = RC.robot_cfg(width)['robot_cfg']['kinematics']; self.margin = margin
        names = [ln for ln in k['collision_link_names'] for _ in k['collision_spheres'][ln]]; ign = k['self_collision_ignore']; n = len(names)
        A = np.zeros((n, n), bool)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = names[i], names[j]
                if a != b and b not in ign.get(a, []) and a not in ign.get(b, []): A[i, j] = True
        self.mask = A; self.names = names

    def self_dist(self, arm, Q):
        S = self.cr.spheres_world(arm, Q); c, r = S[..., :3], S[..., 3]; valid = r > 0
        D = np.linalg.norm(c[:, :, None, :] - c[:, None, :, :], axis=-1) - r[:, :, None] - r[:, None, :]
        m = self.mask[None] & valid[:, :, None] & valid[:, None, :]
        return np.where(m, D, np.inf).reshape(len(S), -1).min(1)

    def trace_ok(self, arm, q0, M, every=5):
        tr, pe, re, conv = self.cr.controller_trace(arm, q0, M); d = self.self_dist(arm, tr[::every] if len(tr) > every else tr)
        return bool(conv and d.min() >= self.margin), tr[-1], float(d.min())
