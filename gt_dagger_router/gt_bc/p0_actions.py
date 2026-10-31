"""GT-DAgger P0 (tool line ip4, 2026-10-31). Official RoboDojo sim_train demos (9 tasks x 100, GT.md):
(1) per-step action / state cache next to the router's DINOv3 cache (router-rgb/official/feat, every 2nd step; read-only here):
    act/<task>/<episode>.npz = state14, action14 (left arm 6, left grip 1, right arm 6, right grip 1; GT.md order), state/action ee poses
    (7, as stored), T, freq. File indices are used as stored (fix_obs_shift_rgb is already applied; no shifting).
(2) split by episode index: 0-79 train, 80-89 val, 90-99 test -> split.json; normalisation statistics from train only -> norm.json.
(3) action semantics, offline, on TRAIN episodes: for each dim and lag k = 0..12, the error of
    ABS:   action[t]            vs state[t+k]
    DELTA: action[t]            vs state[t+k] - state[t]
    and the same for ee poses (position part); gripper: value ranges of action vs state, and the best linear map state[t+k] ~ a*action[t]+b.
    -> results/P0-semantics.json (per task and pooled), the verdict per channel = the hypothesis/lag with the smallest median error.
  /opt/PhysRSI/ENV/robodojo/bin/python -I p0_actions.py [--procs 16]"""
import argparse, json
from multiprocessing import Pool
from pathlib import Path
import h5py
import numpy as np
SRC = Path('/lizekai/RSIws/data/official/sim_train'); G = Path('/lizekai/RSIws/experiments/gt-bc-20261031')
FIELDS = ['left_arm_joint_states', 'left_ee_joint_states', 'right_arm_joint_states', 'right_ee_joint_states']
KS = list(range(13)); ARM = list(range(0, 6)) + list(range(7, 13)); GRIP = [6, 13]


def split_of(i): return 'train' if i < 80 else ('val' if i < 90 else 'test')


def one(p):
    with h5py.File(p, 'r') as f:
        st = np.concatenate([f['state/' + k][:] for k in FIELDS], -1).astype(np.float64); ac = np.concatenate([f['action/' + k][:] for k in FIELDS], -1).astype(np.float64)
        sp = {'%s_%s' % (g, s): f['%s/%s_ee_poses' % (g, s)][:].astype(np.float64) for g in ('state', 'action') for s in ('left', 'right')}
        fr = int(f['additional_info/frequency'][()]); attr = str(f.attrs.get('fix_obs_shift_rgb', ''))
    o = G / 'act' / p.parent.name / (p.stem + '.npz'); o.parent.mkdir(parents=True, exist_ok=True)
    np.savez(o, state=st.astype(np.float32), action=ac.astype(np.float32), T=len(st), freq=fr, **{'ee_' + k: v.astype(np.float32) for k, v in sp.items()})
    # 语义统计：每个滞后 k、每维的绝对误差（中位数用样本拼接后再算，这里先返回逐步误差的和与分位所需的小样本）
    T = len(st); r = dict(task=p.parent.name, ep=p.stem, T=T, freq=fr, attr=attr, abs={}, delta={}, eeabs={}, grip=dict(act_min=ac[:, GRIP].min(0).tolist(), act_max=ac[:, GRIP].max(0).tolist(),
                                                                                                         st_min=st[:, GRIP].min(0).tolist(), st_max=st[:, GRIP].max(0).tolist()))
    for k in KS:
        if T - k < 5: continue
        r['abs'][k] = np.abs(ac[:T - k] - st[k:]).mean(0).tolist(); r['delta'][k] = np.abs(ac[:T - k] - (st[k:] - st[:T - k])).mean(0).tolist()
        r['eeabs'][k] = [float(np.linalg.norm(sp['action_' + s][:T - k, :3] - sp['state_' + s][k:, :3], axis=1).mean()) for s in ('left', 'right')]
    # 夹爪线性映射：state[t+k] ~ a*action[t] + b（k 取 0..12 中残差最小者）
    r['grip_fit'] = {}
    for j in GRIP:
        best = None
        for k in KS:
            if T - k < 5: continue
            x, y = ac[:T - k, j], st[k:, j]
            if x.std() < 1e-6: continue
            a_, b_ = np.polyfit(x, y, 1); e = float(np.abs(a_ * x + b_ - y).mean())
            if best is None or e < best[0]: best = (e, k, float(a_), float(b_))
        r['grip_fit'][j] = best
    # 动作是否“领先”状态：action[t] 与 state[t] 完全相同的比例（若动作=当前状态则无信息）
    r['act_eq_state_frac'] = float((np.abs(ac - st).max(1) < 1e-6).mean())
    return r


def main(a):
    eps = sorted(SRC.glob('*/episode_*.hdf5')); split = {}
    for p in eps: split.setdefault(p.parent.name, {}).setdefault(split_of(int(p.stem.split('_')[1])), []).append(p.stem)
    (G / 'split.json').write_text(json.dumps(split, indent=1))
    with Pool(a.procs) as pool: R = pool.map(one, eps)
    # 归一化：只用 train
    norm = {}
    for task in sorted({r['task'] for r in R}) + ['K1', 'ALL']:
        tasks = {'K1': ['plug_in_charger', 'insert_key', 'insert_tubes', 'deposit_coin']}.get(task, None if task == 'ALL' else [task])
        S, A_ = [], []
        for p in eps:
            if (tasks is None or p.parent.name in tasks) and split_of(int(p.stem.split('_')[1])) == 'train':
                z = np.load(G / 'act' / p.parent.name / (p.stem + '.npz')); S.append(z['state']); A_.append(z['action'])
        S, A_ = np.concatenate(S), np.concatenate(A_)
        norm[task] = dict(state_mean=S.mean(0).tolist(), state_std=(S.std(0) + 1e-6).tolist(), action_mean=A_.mean(0).tolist(), action_std=(A_.std(0) + 1e-6).tolist())
    (G / 'norm.json').write_text(json.dumps(norm))
    out = {}
    for task in sorted({r['task'] for r in R}) + ['ALL']:
        rr = [r for r in R if (task == 'ALL' or r['task'] == task) and split_of(int(r['ep'].split('_')[1])) == 'train']
        e = dict(episodes=len(rr), T_median=float(np.median([r['T'] for r in rr])), freq=sorted({r['freq'] for r in rr}), attr=sorted({r['attr'] for r in rr}),
                 act_eq_state_frac=round(float(np.mean([r['act_eq_state_frac'] for r in rr])), 3))
        for h in ('abs', 'delta'):
            M = {k: np.mean([r[h][k] for r in rr if k in r[h]], 0) for k in KS}
            e[h + '_arm_err_by_k'] = {k: round(float(M[k][ARM].mean()), 5) for k in KS}; e[h + '_grip_err_by_k'] = {k: round(float(M[k][GRIP].mean()), 5) for k in KS}
            e[h + '_arm_best_k'] = int(min(KS, key=lambda k: M[k][ARM].mean()))
        e['ee_pos_abs_err_m_by_k'] = {k: round(float(np.mean([np.mean(r['eeabs'][k]) for r in rr if k in r['eeabs']])), 5) for k in KS}
        e['grip_ranges'] = dict(act_min=np.min([r['grip']['act_min'] for r in rr], 0).tolist(), act_max=np.max([r['grip']['act_max'] for r in rr], 0).tolist(),
                                st_min=np.min([r['grip']['st_min'] for r in rr], 0).tolist(), st_max=np.max([r['grip']['st_max'] for r in rr], 0).tolist())
        gf = [r['grip_fit'][j] for r in rr for j in GRIP if r['grip_fit'].get(j)]
        if gf: e['grip_linear_fit_median'] = dict(err=float(np.median([g[0] for g in gf])), k=float(np.median([g[1] for g in gf])), a=float(np.median([g[2] for g in gf])), b=float(np.median([g[3] for g in gf])))
        ba, bd = min(e['abs_arm_err_by_k'].values()), min(e['delta_arm_err_by_k'].values())
        e['verdict_arm'] = ('ABS joint target, lag %d steps' % e['abs_arm_best_k']) if ba < bd else ('DELTA, lag %d steps' % e['delta_arm_best_k'])
        out[task] = e
    (G / 'results' / 'P0-semantics.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out['ALL'], indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--procs', type=int, default=16); main(ap.parse_args())
