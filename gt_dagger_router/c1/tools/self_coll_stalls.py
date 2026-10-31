"""C1 (10-30): are the sim stalls ('model says reachable, sim stops') arm SELF-collisions? cuRobo X5 spheres (open jaws), pairwise min
distance between spheres of link pairs NOT in the yml self_collision_ignore list, at the stalled joints logged by census v2."""
import importlib.util as ilu, json, glob, collections
import numpy as np
sp = ilu.spec_from_file_location('rc', '/lizekai/RSIws/experiments/macro-tools-20261027/pkg-v2/rsi/reach_cu.py'); RC = ilu.module_from_spec(sp); sp.loader.exec_module(RC)
cr = RC.CollisionReach(width=0.0); k = RC.robot_cfg(0.0)['robot_cfg']['kinematics']
names = [ln for ln in k['collision_link_names'] for _ in k['collision_spheres'][ln]]; ign = k['self_collision_ignore']; buf = k.get('self_collision_buffer', {})
def selfd(arm, q):
    S = cr.spheres_world(arm, q)[0]; best = (9.0, None)
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = names[i], names[j]
            if a == b or b in ign.get(a, []) or a in ign.get(b, []) or S[i, 3] <= 0 or S[j, 3] <= 0: continue
            d = float(np.linalg.norm(S[i, :3] - S[j, :3]) - S[i, 3] - S[j, 3])
            if d < best[0]: best = (d, (a, b))
    return best
C = collections.Counter(); ds = []
for f in glob.glob('/lizekai/RSIws/experiments/macro-tools-20261027/out/rgcensus2/*/result.json'):
    for d in (json.load(open(f)).get('census2') or {}).get('diag') or []:
        if not d.get('q') or 'stalled' not in d['label']: continue
        v, pair = selfd(d['arm'], d['q']); ds.append(v); C[pair if v < 0.005 else 'clear'] += 1
ds = np.array(ds); print('stalled entries', len(ds), 'self-distance <5 mm:', int((ds < 0.005).sum()), ' <0:', int((ds < 0).sum()), ' median mm %.1f' % (np.median(ds) * 1000))
print(C.most_common(8))
# control: random joints inside the per-joint range spanned by the stalled set (same arms)
rng = np.random.default_rng(0); Q = [(d['arm'], d['q']) for f in glob.glob('/lizekai/RSIws/experiments/macro-tools-20261027/out/rgcensus2/*/result.json')
                                    for d in (json.load(open(f)).get('census2') or {}).get('diag') or [] if d.get('q') and 'stalled' in d['label']]
cs = []
for arm in ('left', 'right'):
    A = np.array([q for a, q in Q if a == arm])
    for _ in range(150): cs.append(selfd(arm, rng.uniform(A.min(0), A.max(0)))[0])
cs = np.array(cs); print('control random', len(cs), '<5 mm:', int((cs < 0.005).sum()), ' <0:', int((cs < 0).sum()), ' median mm %.1f' % (np.median(cs) * 1000))
