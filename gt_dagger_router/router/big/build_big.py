"""ROUTER-BIG v1 dataset (capability line, Track D, 2026-10-30): one sequence format for all sources, PCA-compressed visual features.
Per frame: V (768,) fp16 = PCA(3 x 1536 DINOv3 [CLS, mean patch] of head / left / right, same router_rgb.dino_features), P (14,) proprio
(per arm: gripper reading, gripper command (demos: reading < 0.9 -> 0), steps since close / 100, eef z, z velocity x100/4, xy speed x100/4, z since close x10).
Labels per frame (-1 / nan = unknown): task (42-way, global list), stage (0 approach 1 grasp 2 lift 3 transport 4 align 5 inserted/placed),
progress t/T, success (episode outcome: demo 1, Astra native), takeover (fitmap rows only).
Sources: official sim_train demos (router-rgb/official/feat; stage from proprio: approach / grasp / lift / placed, carry frames unknown),
Astra K1 stage episodes (stage/stage_k1_v2.npz), Astra task-ID frames (taskid/feat, 5 frames per episode, task only), ROUTER-v0 plug rows
(gate/ROUTER-DATA-v0, takeover label).  -> big_v1.npz (+ pca.npz, tasks.json); split = 20 % of groups held out (ctx / demo episode)."""
import glob, json, random
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; RR = H.parent
TASKS = sorted({e['task'] for e in json.load(open(RR / 'taskid/episode_index.json'))}); TI = {t: i for i, t in enumerate(TASKS)}
def prop_from(g, z, xy, t):
    """g, z (L,) and xy (L,2) per arm over sampled frames with step indices t (L,)."""
    out = []
    for gi, zi, xyi in zip(g, z, xy):
        L = len(gi); closed = gi < 0.9; since = np.full(L, -1.0); zc = np.zeros(L); cur = None
        for k in range(L):
            if closed[k] and cur is None: cur = k
            if not closed[k]: cur = None
            since[k] = (t[k] - t[cur]) / 100.0 if cur is not None else -1.0; zc[k] = (zi[k] - zi[cur]) * 10 if cur is not None else 0.0
        dz = np.r_[0, np.diff(zi)] * 100 / np.maximum(np.r_[1, np.diff(t)], 1); dxy = np.r_[0, np.linalg.norm(np.diff(xyi, axis=0), axis=1)] * 100 / np.maximum(np.r_[1, np.diff(t)], 1)
        out.append(np.c_[gi, (~closed).astype(float), since, zi, dz, dxy, zc])   # col 1 = 'last command' convention of add_prop (0 = closed)
    return np.concatenate(out, 1).astype(np.float32)
EPI = []   # dicts: V3 (L,3,1536) fp16, P (L,14), task, stage (L,), prog (L,), succ, take (L,), group, src
for f in sorted(glob.glob(str(RR / 'official/feat/*/*.npz'))):
    z = np.load(f, allow_pickle=True); t = z['t']; T = int(z['T']); task = str(z['task'])
    g = [z['%s_ee_joint_states' % a][:, 0] for a in ('left', 'right')]; zz = [z['%s_ee_poses' % a][:, 2] for a in ('left', 'right')]; xy = [z['%s_ee_poses' % a][:, :2] for a in ('left', 'right')]
    P = prop_from(g, zz, xy, t); st = np.full(len(t), -1, int); held_before = False
    for k in range(len(t)):
        cl = [P[k, 1] < 0.5, P[k, 8] < 0.5]; zc = max(P[k, 6] if cl[0] else -9, P[k, 13] if cl[1] else -9) / 10
        if not any(cl): st[k] = 5 if held_before else 0
        else:
            held_before = held_before or zc >= 0.03; st[k] = 1 if zc < 0.01 else 2 if zc < 0.03 else -1
    EPI.append(dict(V3=z['X'], P=P, task=TI.get(task, -1), stage=st, prog=t / max(T - 1, 1), succ=1.0, take=np.full(len(t), np.nan), group='demo:' + Path(f).stem + task, src='demo'))
Z = np.load(RR / 'stage/stage_k1_v2.npz', allow_pickle=True); M = json.loads(str(Z['meta']))
for X, Y, P, m in zip(Z['X'], Z['Y'], Z['P'], M):
    if not len(X): continue
    st = np.asarray(m['steps']); EPI.append(dict(V3=X, P=np.asarray(P, np.float32), task=TI[m['task']], stage=np.asarray(Y, int), prog=np.full(len(X), np.nan), succ=float(bool(m.get('native_success'))),
                                                 take=np.full(len(X), np.nan), group=m['ctx'], src='astra_k1'))
idx = {e['episode']: e for e in json.load(open(RR / 'taskid/episode_index.json'))}
for f in sorted(glob.glob(str(RR / 'taskid/feat/shard-*.npz'))):
    z = np.load(f)
    for X, ok, ep in zip(z['X'], z['ok'], z['episodes']):
        X = X[ok]
        if not len(X): continue
        EPI.append(dict(V3=X, P=np.zeros((len(X), 14), np.float32), task=TI[idx[ep]['task']], stage=np.full(len(X), -1), prog=np.full(len(X), np.nan), succ=np.nan,
                        take=np.full(len(X), np.nan), group='-'.join(ep.split('-')[:3]), src='astra_taskid'))
D = json.load(open(RR.parent / 'gate/ROUTER-DATA-v0.json'))['rows']; XV = np.load(RR.parent / 'gate/ROUTER-DATA-v0.npz')['dino']; seen = set()
for r, x in zip(D, XV):
    if r['kind'] != 'rg3f' or (r['ctx'], r['t']) in seen: continue
    seen.add((r['ctx'], r['t'])); y = float(r['int_success'] and not r['astra_success'] and not r['abstained'])
    EPI.append(dict(V3=x[None].astype(np.float16), P=np.zeros((1, 14), np.float32), task=TI['plug_in_charger'], stage=np.full(1, -1), prog=np.full(1, np.nan), succ=np.nan, take=np.array([y]), group=r['ctx'], src='fitmap_plug'))
# PCA (fit on a sample of frames from all sources), 3 x 1536 -> 768
rng = np.random.default_rng(0); S = np.concatenate([e['V3'][rng.integers(0, len(e['V3']), 2)].reshape(-1, 3 * 1536) for e in EPI]).astype(np.float32)
mu = S.mean(0); U, s_, Vt = np.linalg.svd(S - mu, full_matrices=False); W = Vt[:768].T.astype(np.float32)
np.savez(H / 'pca.npz', mu=mu, W=W, explained=float((s_[:768] ** 2).sum() / (s_ ** 2).sum()))
gr = sorted({e['group'] for e in EPI}); random.Random(20261030).shuffle(gr); dev_g = set(gr[:len(gr) // 5])
out = dict(V=[], P=[], task=[], stage=[], prog=[], succ=[], take=[], src=[], dev=[], group=[])
for e in EPI:
    out['V'].append(((e['V3'].reshape(len(e['V3']), -1).astype(np.float32) - mu) @ W).astype(np.float16)); out['P'].append(e['P']); out['task'].append(e['task']); out['stage'].append(e['stage'].astype(np.int8))
    out['prog'].append(np.asarray(e['prog'], np.float32)); out['succ'].append(e['succ']); out['take'].append(np.asarray(e['take'], np.float32)); out['src'].append(e['src']); out['dev'].append(e['group'] in dev_g); out['group'].append(e['group'])
np.savez_compressed(H / 'big_v1.npz', **{k: np.array(v, dtype=object) for k, v in out.items()})
json.dump(TASKS, open(H / 'tasks.json', 'w'))
import collections; print(len(EPI), collections.Counter(out['src']), 'frames', sum(len(v) for v in out['V']), 'pca explained', round(float((s_[:768] ** 2).sum() / (s_ ** 2).sum()), 3))
