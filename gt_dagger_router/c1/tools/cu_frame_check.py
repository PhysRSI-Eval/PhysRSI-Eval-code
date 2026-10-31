"""C1: does cuRobo's link6 frame ORIENTATION equal ArmFK's? (position already matches < 0.1 mm). Prints the relative rotation."""
import importlib.util as ilu, sys, json
import numpy as np
from scipy.spatial.transform import Rotation as R
sp = ilu.spec_from_file_location('rc', '/lizekai/RSIws/experiments/macro-tools-20261027/pkg-v2/rsi/reach_cu.py'); RC = ilu.module_from_spec(sp); sp.loader.exec_module(RC)
cr = RC.CollisionReach(width=0.0); arm = sys.argv[1]; q = np.array(json.loads(sys.argv[2]), float)
st = cr.kin.compute_kinematics(cr.JS.from_position(cr.torch.tensor(q[None], dtype=cr.torch.float32, device='cuda'), joint_names=cr.joint_names))
tq = st.tool_poses.quaternion.cpu().numpy().reshape(-1)[:4]; Rc = cr.rm.root[arm][:3, :3] @ R.from_quat([tq[1], tq[2], tq[3], tq[0]]).as_matrix()
Ra = cr.rm.pose(arm, q)[:3, :3]; D = Ra.T @ Rc
print('rel rot deg', round(R.from_matrix(D).magnitude() * 57.3, 2), 'D=\n', D.round(3))
print('ArmFK x-axis (approach) world', Ra[:, 0].round(3), ' cuRobo x-axis world', Rc[:, 0].round(3))
S = cr.spheres_world(arm, q)[0]; P = cr.rm.pose(arm, q); loc = (S[:, :3] - P[:3, 3]) @ P[:3, :3]
k = RC.robot_cfg(0.0)['robot_cfg']['kinematics']; nm = [ln for ln in k['collision_link_names'] for _ in k['collision_spheres'][ln]]
for n, l, s in zip(nm, loc, S):
    if n == 'link6': print('link6 sphere in ArmFK link6 frame', l.round(3), 'r', round(float(s[3]), 3), 'world z', round(float(s[2]), 3))
