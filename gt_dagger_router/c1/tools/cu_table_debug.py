"""C1 (2026-10-29): why does cuRobo (reach_cu v2) put the census holds 50-200 mm 'in the table'? Per-link sphere z vs table plane, and cuRobo link6 FK vs ArmFK."""
import importlib.util as ilu, sys, json
import numpy as np
sp = ilu.spec_from_file_location('rc', '/lizekai/RSIws/experiments/macro-tools-20261027/pkg-v2/rsi/reach_cu.py'); RC = ilu.module_from_spec(sp); sp.loader.exec_module(RC)
cr = RC.CollisionReach(width=0.0)
arm = sys.argv[1]; q = np.array(json.loads(sys.argv[2]), float)
print('joint_names', cr.joint_names)
st = cr.kin.compute_kinematics(cr.JS.from_position(cr.torch.tensor(q[None], dtype=cr.torch.float32, device='cuda'), joint_names=cr.joint_names))
root = cr.rm.root[arm]
print('state attrs', [a for a in dir(st) if not a.startswith('_')][:40])
try:
    tp = st.tool_poses if hasattr(st, 'tool_poses') else None; print('tool_poses', tp)
except Exception as e: print('tp err', e)
print('ArmFK link6 world', cr.rm.pose(arm, q)[:3, 3])
S = cr.spheres_world(arm, q)[0]
names = []
k = RC.robot_cfg(0.0)['robot_cfg']['kinematics']
for ln in k['collision_link_names']:
    for s in k['collision_spheres'][ln]: names.append(ln)
print('n spheres', len(S), 'n names', len(names), 'table_z', cr.table_z(arm))
for i, s in enumerate(S):
    if s[3] > 0 and s[2] - s[3] < cr.table_z(arm) + 0.02:
        print('%-10s c=(%.3f %.3f %.3f) r=%.3f  clear=%.1f mm' % (names[i] if i < len(names) else '?', s[0], s[1], s[2], s[3], (s[2] - s[3] - cr.table_z(arm)) * 1000))
# trace probe: targets = current link6 shifted 30 mm along +-x/y/z (same orientation)
M0 = cr.rm.pose(arm, q)
for v in ((0.03, 0, 0), (-0.03, 0, 0), (0, 0.03, 0), (0, -0.03, 0), (0, 0, 0.03), (0, 0, -0.03)):
    M = M0.copy(); M[:3, 3] += v; tr, pe, re, conv = cr.controller_trace(arm, q, M); d, who = cr.clearance(arm, tr)
    Sx = cr.spheres_world(arm, tr); i = int(np.argmin(d)); j = int(np.argmin(Sx[i, :, 2] - Sx[i, :, 3]))
    print(v, 'conv', conv, 'n', len(tr), 'min clear %.1f mm at tick %d by %s' % (d.min() * 1000, i, names[j]), 'q_end-q0 max %.2f rad' % np.abs(tr[-1] - q).max())
