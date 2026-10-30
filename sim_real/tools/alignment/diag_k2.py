"""K2-v1 判定后的失败归因分析（不改变判定）：按任务、按偏离类型拆开看 K 模型在 HOLD-T2 / HOLD-T 偏离样本上的表现。
偏离类型：release = 动作块内该臂夹爪指令由闭合变张开（松手后物体下落）；held = 全程闭合（滑脱、碰撞、推动）；open = t 时刻已张开（推、碰）。
  python -I diag_k2.py <ckpt> <out_json>"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_pw as P
import train_k2 as K


def main(ckpt, out):
    dev = torch.device('cuda'); ck = torch.load(ckpt, map_location='cpu')
    model = K.K2(ck['mode']).to(dev); model.load_state_dict(ck['state_dict']); model.eval()
    S = K.targets(P.load(['astra_v1'])); hold2 = {'pack_objects_into_box', 'fill_pen_holder'}
    for s in S:
        if s['task'] in hold2: s['split'] = 'HOLD-T2'
    arrs = {}; G = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for i, s in enumerate(S):
            if s['split'] not in ('HOLD-T2', 'HOLD-T', 'TEST') or s['m'].sum() == 0: continue
            b = [x.to(dev) for x in K.batch(S, [i])]; pk = K.run(model, b)[1][0].cpu(); kp, fol = s['kp'], s['fol']
            key = (s['task'], s['ep'])
            if key not in arrs: arrs[key] = np.load(P.SRC['astra_v1'] / 'arrays' / s['task'] / ('%s.npz' % s['ep']))
            d = arrs[key]; T = int(d['T'])
            for a, arm in enumerate(P.ARMS):
                if s['m'][a] == 0: continue
                ea = dict(model=(pk[a] - kp[a]).norm(dim=-1).mean().item(), follow=(fol[a] - kp[a]).norm(dim=-1).mean().item(), nochange=kp[a].norm(dim=-1).mean().item(), disp=0.0)
                for k, v in ea.items(): G[('all', '%s/%s' % (s['split'], s['task']))][k].append(v)   # 全部样本（含正常持握）：看门控在哪些任务上有用
                if (fol[a] - kp[a]).norm(dim=-1).mean().item() <= K.DEV_CM: continue
                g = d['action/%s_ee_joint_states' % arm][s['t']:min(s['t'] + P.H, T), 0]; g0 = d['state/%s_ee_joint_states' % arm][s['t'], 0]
                typ = 'open' if g0 >= 0.85 else 'release' if (g >= 0.5).any() else 'held'
                e = dict(model=(pk[a] - kp[a]).norm(dim=-1).mean().item(), follow=(fol[a] - kp[a]).norm(dim=-1).mean().item(),
                         nochange=kp[a].norm(dim=-1).mean().item(), disp=kp[a][-1].norm(dim=-1).mean().item())
                for grp in (('task', '%s/%s' % (s['split'], s['task'])), ('type', '%s/%s' % (s['split'], typ)), ('task_type', '%s/%s/%s' % (s['split'], s['task'], typ))):
                    for k, v in e.items(): G[grp][k].append(v)
    res = {'%s=%s' % k: dict(n=len(v['model']), **{m: float(np.median(x)) for m, x in v.items()}) for k, v in sorted(G.items())}
    for r in res.values(): r['ratio'] = r['model'] / min(r['follow'], r['nochange'])
    json.dump(res, open(out, 'w'), indent=1)
    for k, r in res.items(): print('%-48s n %5d  model %.2f  follow %.2f  nochange %.2f  ratio %.2f  |disp16| %.2f' % (k, r['n'], r['model'], r['follow'], r['nochange'], r['ratio'], r['disp']))


if __name__ == '__main__':
    main(*sys.argv[1:3])
