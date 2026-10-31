"""DAgger relabel check (dispatcher 10-31 04:4xZ): on the policy-visited states of C1's rollouts, the HOLDING arm's gripper: the state reading,
the expert's chunk (first tick and chunk mean) and each BC model's predicted chunk, per expert phase. Also, for the 'release' rows, the
expert's residual (dxy / dz / checks) to see whether the expert opens before the charger is inserted. BC input = the stored DINOv3
features + state with a stationary history (= policy_v2 at a fresh start). Gripper: 1 open .. 0 closed; holding the charger reads ~0.36.
  python dagger_gripcheck.py OUT.json GPU npz[,npz] ckpt[,ckpt]"""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent)); import bcnet as B  # noqa: E402
out, gpu = sys.argv[1], int(sys.argv[2]); npzs = sys.argv[3].split(','); cks = sys.argv[4].split(','); dev = torch.device('cuda:%d' % gpu)
Z = [np.load(f, allow_pickle=True) for f in npzs]
X = np.concatenate([z['X'] for z in Z]).astype(np.float32); S = np.concatenate([z['state'] for z in Z]); E = np.concatenate([z['expert'] for z in Z])
PH = np.concatenate([z['phase'] for z in Z]); HA = np.concatenate([z['hold_arm'] for z in Z]); INF = [json.loads(x) for z in Z for x in z['info']]; SRC = np.concatenate([[Path(f).stem] * len(z['X']) for f, z in zip(npzs, Z)])
gi = np.array([6 if a == 'left' else 13 for a in HA]); n = len(X); R = dict(n=n, by_source={s: int((SRC == s).sum()) for s in set(SRC)})
pred = {}
for ck in cks:
    c = torch.load(ck, map_location=dev); nm = B.Norm(c['norm']); net = B.BC(c['H'], c['hist'], ntask=c.get('ntask', 4)).to(dev); net.load_state_dict(c['state']); net.eval(); P = []
    with torch.no_grad():
        for i in range(0, n, 256):
            s = S[i:i + 256]; sn = (s - nm.sm) / nm.ss; hist = c['hist']
            # 阶段 token 模型：用部署时的本体规则（任一夹爪读数 0.05..0.6 = 持握 -> hold=1，否则按 release=2 处理）；plug 任务索引 0
            tk = [(c.get('ntask_base', 4) * (1 if ((0.05 < x[6] < 0.6) or (0.05 < x[13] < 0.6)) else 2)) if c.get('stage_token') else 0 for x in s]
            h = np.concatenate([np.repeat(sn[:, None], hist, 1), np.repeat(((s - nm.am) / nm.as_)[:, None], hist, 1)], 2) if hist else np.zeros((len(s), 0, 28), np.float32); hm = np.ones((len(s), hist), np.float32)
            y = net(*[torch.from_numpy(np.asarray(v, np.float32)).to(dev) for v in (X[i:i + 256], sn, h, hm)], torch.tensor(tk, dtype=torch.long, device=dev)).cpu().numpy()
            P += [nm.dec_y(y[j], s[j], c['target']) for j in range(len(s))]
    pred[Path(ck).stem] = np.stack(P)
for ph in sorted(set(PH)) + ['ALL']:
    m = np.ones(n, bool) if ph == 'ALL' else PH == ph; idx = np.where(m)[0]
    if not len(idx): continue
    sg = S[idx, gi[idx]]; eg0 = E[idx, 0, gi[idx]]; egm = E[idx, :, gi[idx]].mean(1) if False else np.array([E[k, :, gi[k]].mean() for k in idx]); eglast = np.array([E[k, -1, gi[k]] for k in idx])
    e = dict(n=len(idx), state_grip_mean=round(float(sg.mean()), 3), expert_grip_first=round(float(eg0.mean()), 3), expert_grip_last=round(float(eglast.mean()), 3),
             expert_opens_frac=round(float((eglast > sg + 0.2).mean()), 3))
    for k, Pk in pred.items():
        pg = np.array([Pk[j, :, gi[j]].mean() for j in idx]); pl = np.array([Pk[j, -1, gi[j]] for j in idx])
        e[k] = dict(pred_grip_mean=round(float(pg.mean()), 3), pred_grip_last=round(float(pl.mean()), 3), pred_opens_frac=round(float((pl > sg + 0.2).mean()), 3),
                    abs_err_vs_expert=round(float(np.mean([np.abs(Pk[j, :, gi[j]] - E[j, :, gi[j]]).mean() for j in idx])), 3),
                    arm_err_vs_expert_rad=round(float(np.mean([np.abs(Pk[j][:, B.ARMM > 0] - E[j][:, B.ARMM > 0]).mean() for j in idx])), 4))
    if ph == 'release':
        res = [INF[j].get('residual', {}) if INF[j] else {} for j in idx]; ck_ = [INF[j].get('checks') if INF[j] else None for j in idx]
        for q in ('dxy_mm', 'dz_mm', 'dyaw_deg'):
            v = [r[q] for r in res if q in r]
            if v: e['residual_' + q] = (round(float(np.median(v)), 2), round(float(np.percentile(v, 90)), 2))
        e['checks_all_pass_frac'] = round(float(np.mean([bool(c) and min(c) >= 1.0 for c in ck_])), 3); e['info_keys'] = sorted({k for j in idx if INF[j] for k in INF[j]})
    R[ph] = e
Path(out).write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1))
