"""GT-DAgger P1: train the K1 insert-skill BC policy (bcnet.BC) with teacher forcing on the official GT demos (split.json 80/10/10),
then evaluate on val and test: (1) teacher-forced action error (GT history), (2) free-running offline rollout on the episodes
(own executed actions as history; state from the P0 tracking model state[t+k] = action[t] for arm joints, linear map for grippers; images
stay GT because nothing is rendered offline) -> the gap = exposure bias from the history/state channel. Reported fit-style per task and
per stage (reach = before the first gripper close of the active arm, hold = closed, release = after it reopens) and per step of the chunk.
  python train_bc.py --tasks plug_in_charger[,insert_key,...] --tag NAME --gpu G [--H 16 --hist 8 --target res --epochs 40]"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import bcnet as B
G = B.G


def stages(E):
    st = E['state']; act = int(np.argmax([np.abs(np.diff(st[:, 0:6], axis=0)).sum(), np.abs(np.diff(st[:, 7:13], axis=0)).sum()])); g = st[:, 6 if act == 0 else 13]
    lo, hi = g.min(), g.max(); closed = g < lo + 0.5 * (hi - lo) if hi - lo > 1e-3 else np.zeros(len(g), bool); lab = np.zeros(len(g), int)
    if closed.any():
        c0 = int(np.argmax(closed)); lab[c0:] = 1; op = np.where(~closed[c0:])[0]
        if len(op): lab[c0 + op[0]:] = 2
    return lab  # 0 reach, 1 hold, 2 release


def batches(eps, nm, a, shuffle, rng):
    idx = [(k, i) for k, E in enumerate(eps) for i in range(len(E['tX']))]
    if shuffle: rng.shuffle(idx)
    for j in range(0, len(idx), a.bs):
        S = [B.sample(eps[k], i, nm, a.hist, a.H, a.target) for k, i in idx[j:j + a.bs]]; tk = [B.TASKS.index(eps[k]['task']) + (a.ntask * int(eps[k]['stage'][int(eps[k]['tX'][i])]) if a.stage_token else 0) for k, i in idx[j:j + a.bs]]  # 阶段 token：任务 x 阶段（reach/hold/release）合成一个嵌入
        yield [torch.from_numpy(np.stack(x)).float() for x in zip(*S)] + [torch.tensor(tk)], idx[j:j + a.bs]


def tf_eval(net, eps, nm, a, dev):
    net.eval(); R = []
    with torch.no_grad():
        for (X, s, h, hm, y, m, tk), ids in batches(eps, nm, a, False, None):
            p = net(X.to(dev), s.to(dev), h.to(dev), hm.to(dev), tk.to(dev)).cpu().numpy()
            for b, (k, i) in enumerate(ids):
                E = eps[k]; t = int(E['tX'][i]); s0 = E['state'][t]; pa = nm.dec_y(p[b], s0, a.target); ga = nm.dec_y(y[b].numpy(), s0, a.target); mm = m[b].numpy() > 0
                R.append(dict(task=E['task'], ep=E['ep'], t=t, stage=int(E['stage'][t]), err=np.abs(pa - ga)[mm], err0=np.abs(pa[0] - ga[0])))
    return R


def free_run(net, E, nm, a, dev, sem):
    """offline closed loop on one episode from t = 0: predict a chunk every EXEC steps, execute its first EXEC actions; own actions fill the
    history; state = tracking model of own actions (lag k); images GT (nearest cached even step)."""
    T = E['T']; k = sem['k']; ga, gs = E['action'], E['state']; pa = np.zeros_like(ga); ps = gs.copy(); t = 0
    while t < T:
        i = min(len(E['tX']) - 1, t // 2); s = ps[t]
        hs = np.stack([ps[max(0, t - j)] for j in range(a.hist, 0, -1)] or [gs[0]])[:a.hist]; ha = np.stack([pa[t - j] if t - j >= 0 else gs[0] for j in range(a.hist, 0, -1)] or [gs[0]])[:a.hist]
        hm = np.array([1.0 if t - j >= 0 else 0.0 for j in range(a.hist, 0, -1)], np.float32); h = np.concatenate([(hs - nm.sm) / nm.ss, (ha - nm.am) / nm.as_], 1)
        with torch.no_grad():
            p = net(torch.from_numpy(E['X'][i][None]).float().to(dev), torch.from_numpy(((s - nm.sm) / nm.ss)[None]).float().to(dev), torch.from_numpy(h[None]).float().to(dev),
                    torch.from_numpy(hm[None]).to(dev), torch.tensor([B.TASKS.index(E['task']) + (a.ntask * int(E['stage'][t]) if a.stage_token else 0)]).to(dev)).cpu().numpy()[0]
        ch = nm.dec_y(p, s, a.target)
        for j in range(a.exec_steps):
            if t + j >= T: break
            pa[t + j] = ch[j]
            for u in range(t + j + 1, min(T, t + j + 1 + 1)):  # 下一步状态：state[u] = f(action[u-k])
                src = u - k
                if src >= 0 and src <= t + j: ps[u] = sem['map'](pa[src], gs[u])
        t += a.exec_steps
    return pa, ps


def summarize(R, key):
    out = {}
    for name, sel in key:
        rr = [r for r in R if sel(r)]
        if not rr: continue
        e0 = np.stack([r['err0'] for r in rr]); ec = np.concatenate([r['err'] for r in rr])
        out[name] = dict(n=len(rr), first_arm_rad=round(float(e0[:, B.ARMM > 0].mean()), 4), first_grip=round(float(e0[:, B.GRIP].mean()), 4),
                         chunk_arm_rad=round(float(ec[:, B.ARMM > 0].mean()), 4), chunk_grip=round(float(ec[:, B.GRIP].mean()), 4))
    return out


def main(a):
    split = json.load(open(G / 'split.json')); tasks = a.tasks.split(','); nm = B.Norm(a.norm); dev = torch.device('cuda:%d' % a.gpu); torch.manual_seed(0); rng = np.random.default_rng(0)
    D = {sp: [B.load_episode(t, e) for t in tasks for e in split[t][sp]] for sp in ('train', 'val', 'test')}
    for sp in D:
        for E in D[sp]: E['stage'] = stages(E)
    # DAgger 数据源（dagger_feats.py 的 npz）：每个被访问状态 = 一条只有 1 个查询点的伪示范（无历史，掩码全 0；动作 = 专家纠正块），
    # 混入 train；val/test 仍是官方示范，便于与纯 BC 直接比较。需配合 --hist 0 或历史 dropout 训练（否则空历史在分布外）。
    nd = 0
    for f in [x for x in a.dagger.split(',') if x]:
        z = np.load(f, allow_pickle=True)
        for i in range(len(z['X'])):
            if a.dagger_phases and str(z['phase'][i]) not in a.dagger_phases.split(','): continue  # 例：去掉错误的 release 标签（DAGGER-gripcheck：残差 473 mm 时就教松手）
            ex = z['expert'][i]; s0 = z['state'][i]
            for _ in range(a.dagger_weight): D['train'].append(dict(task='plug_in_charger', ep='dagger', X=z['X'][i][None].astype(np.float32), tX=np.array([0]), state=s0[None], action=ex, T=len(ex), stage=np.full(len(ex), 2 if str(z['phase'][i]) == 'release' else 1, int)))  # 专家阶段：align/transport = 持握(1)，release = 2
            nd += 1
    print(json.dumps(dict(dagger_states=nd, weight=a.dagger_weight)), flush=True)
    net = B.BC(a.H, a.hist, ntask=a.ntask * (3 if a.stage_token else 1)).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    nb = sum(len(E['tX']) for E in D['train']) // a.bs + 1; sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * nb + 10); best = None; log = []
    for ep in range(a.epochs):
        net.train(); tot = 0; n = 0; t0 = time.time()
        for (X, s, h, hm, y, m, tk), _ in batches(D['train'], nm, a, True, rng):
            X, s, h, hm, y, m, tk = [x.to(dev) for x in (X, s, h, hm, y, m, tk)]
            if a.state_noise > 0:  # DART 式输入噪声：只扰动本体/历史输入（归一化单位），监督仍是 GT 动作 -> 学会从偏离状态拉回
                s = s + a.state_noise * torch.randn_like(s); h = h + a.state_noise * torch.randn_like(h)
            if a.hist and a.hist_drop > 0: hm = hm * (torch.rand(hm.shape[0], 1, device=dev) > a.hist_drop).float()
            l = ((net(X, s, h, hm, tk) - y).abs().mean(-1) * m).sum() / m.sum(); opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            tot += float(l); n += 1
        V = tf_eval(net, D['val'], nm, a, dev); v = float(np.mean([r['err'][:, B.ARMM > 0].mean() for r in V]))
        log.append(dict(epoch=ep, train=round(tot / n, 4), val_chunk_arm_rad=round(v, 5), sec=round(time.time() - t0, 1))); print(json.dumps(log[-1]), flush=True)
        if best is None or v < best: best = v; torch.save(dict(state=net.state_dict(), H=a.H, hist=a.hist, target=a.target, tasks=tasks, norm=a.norm, ntask=a.ntask * (3 if a.stage_token else 1), ntask_base=a.ntask, stage_token=a.stage_token, epoch=ep), G / 'models' / ('bc-%s.pt' % a.tag))
    ck = torch.load(G / 'models' / ('bc-%s.pt' % a.tag), map_location=dev); net.load_state_dict(ck['state'])
    P0 = json.load(open(G / 'results' / 'P0-semantics.json'))['ALL']; k = int(P0['abs_arm_best_k']); gf = P0.get('grip_linear_fit_median', dict(a=1.0, b=0.0))
    def mp(act, gs_u):  # 跟踪模型：手臂 state = action（绝对关节目标）；夹爪线性映射
        o = act.copy(); o[B.GRIP] = gf['a'] * act[B.GRIP] + gf['b']; return o
    sem = dict(k=k, map=mp); res = dict(tag=a.tag, tasks=tasks, args=vars(a), best_val=best, log=log, semantics_used=dict(k=k, grip=gf))
    for sp in ('val', 'test'):
        R = tf_eval(net, D[sp], nm, a, dev)
        keys = [('all', lambda r: True)] + [(t, (lambda r, t=t: r['task'] == t)) for t in tasks] + [('%s|%s' % (t, sn), (lambda r, t=t, si=si: r['task'] == t and r['stage'] == si)) for t in tasks for si, sn in enumerate(('reach', 'hold', 'release'))]
        res['teacher_forced_' + sp] = summarize(R, keys)
        res['teacher_forced_%s_by_chunk_step' % sp] = [round(float(np.mean([r['err'][j][B.ARMM > 0].mean() for r in R if len(r['err']) > j])), 5) for j in range(a.H)]
    # 自由运行（val）：与同一条示范、同一步上的 GT 动作比较；同时报告教师强制的逐步首动作误差作对照
    FR = []
    for E in D['val']:
        pa, ps = free_run(net, E, nm, a, dev, sem); e = np.abs(pa - E['action']); es = np.abs(ps - E['state'])
        for t in range(E['T']): FR.append(dict(task=E['task'], stage=int(E['stage'][t]), arm=float(e[t, B.ARMM > 0].mean()), grip=float(e[t, B.GRIP].mean()), state_arm=float(es[t, B.ARMM > 0].mean()), prog=t / E['T']))
    TF = tf_eval(net, D['val'], nm, a, dev); fr = {}
    for t in tasks + ['all']:
        for si, sn in [(-1, 'all'), (0, 'reach'), (1, 'hold'), (2, 'release')]:
            rr = [r for r in FR if (t == 'all' or r['task'] == t) and (si < 0 or r['stage'] == si)]; tt = [r for r in TF if (t == 'all' or r['task'] == t) and (si < 0 or r['stage'] == si)]
            if rr: fr['%s|%s' % (t, sn)] = dict(free_arm_rad=round(float(np.mean([r['arm'] for r in rr])), 4), free_grip=round(float(np.mean([r['grip'] for r in rr])), 4),
                                               free_state_drift_rad=round(float(np.mean([r['state_arm'] for r in rr])), 4),
                                               tf_first_arm_rad=round(float(np.mean([r['err0'][B.ARMM > 0].mean() for r in tt])), 4) if tt else None)
    res['free_running_val'] = fr
    zb = [np.abs(E['action'][t:t + a.H] - E['state'][t])[:, B.ARMM > 0].mean() for E in D['test'] for t in E['tX']]; res['zero_motion_baseline_test_chunk_arm_rad'] = round(float(np.mean(zb)), 4)
    res['free_running_val_by_progress'] = [round(float(np.mean([r['arm'] for r in FR if lo <= r['prog'] < lo + 0.1])), 4) for lo in np.arange(0, 1, 0.1)]
    (G / 'results' / ('BC-%s.json' % a.tag)).write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k.startswith(('teacher_forced_test', 'free_running_val')) and 'by' not in k}, indent=1)[:3000])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--tasks', required=True); ap.add_argument('--tag', required=True); ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--H', type=int, default=16); ap.add_argument('--hist', type=int, default=8); ap.add_argument('--target', default='res'); ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=128); ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--exec-steps', dest='exec_steps', type=int, default=8)
    ap.add_argument('--hist-drop', dest='hist_drop', type=float, default=0.0); ap.add_argument('--state-noise', dest='state_noise', type=float, default=0.0); ap.add_argument('--dagger', default=''); ap.add_argument('--norm', default='K1'); ap.add_argument('--stage-token', dest='stage_token', action='store_true'); ap.add_argument('--dagger-phases', dest='dagger_phases', default=''); ap.add_argument('--ntask', type=int, default=4); ap.add_argument('--dagger-weight', dest='dagger_weight', type=int, default=1); main(ap.parse_args())
