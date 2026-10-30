"""K2-f: alignment in the SAME functional-keypoint coordinates the KPNet relative servo uses (Track C, system rule 11:5xZ; dev).
原理：工具族（finepose/kpfactory 的 KPNet 相对视觉伺服）把"被插件功能点 -> 目标功能点"的相对向量 r 伺服到零；
对齐模型若在同一坐标下预测 Astra 的动作块会让 r 怎么变（Δr，t+4..t+16），就能直接给 harness 一个信号
（"这个动作块让对准变好还是变坏"），也能给路由器（Track D）当"该不该交给 KP 伺服"的特征。
部署时当前 r 由 KPNet 工具给出（RGB），模型只预测 Δr；训练目标 r 用仿真真值 × CAD 功能点表计算（训练时可用）。
功能点表与 finepose README §2 完全一致；新任务（充电器插脚/插座孔、硬币/投币口）等 kpfactory 的点表定稿后加进 FUNC。
基线：不变；跟随（被插件随最近的夹爪按动作块中的末端位姿刚体移动，目标不动）。指标：Δr 误差中位数（cm）、
"是否改善对准"（|r| 是否变小）的判别准确率（对比跟随基线）。
  python -I train_k2f.py --epochs 40 --device cuda --out ../out/k2f-dev.json"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import train_pw as P
from train_k2 import Tm, TCP

# 功能点表：被插件（moving）与目标（target）各一点，单位 m，坐标系同 finepose README §2
FUNC = {'insert_key': dict(mov=('key_', np.array([0.0021, 0.0, -0.02135]) + np.array([-0.0021, 0.0, -0.0314])),   # 资产原点 -> bbox 中心 -> 钥匙尖
                           tgt=('key_slot', np.array([-0.002, 0.0, 0.055])))}                                       # 锁坐标系 -> 圆孔中心


def pt(o, ofs): M = Tm(o['position'], o['quaternion_wxyz']); return M[:3, :3] @ ofs + M[:3, 3]


def find(objs, prefix, excl=None):
    return next((n for n, v in objs.items() if n.startswith(prefix) and 'position' in v and not (excl and n.startswith(excl))), None)


def func_targets(S):
    """只保留有功能点表的任务；每样本 r0 (3,) cm、dr (K,3) cm、跟随基线 fol (K,3) cm、m 有效标记。"""
    out, cache = [], {}
    for s in S:
        if s['src'] != 'data-astra' or s['task'] not in FUNC: continue
        key = (s['task'], s['ep'])
        if key not in cache:
            root = P.SRC['astra']; cache[key] = (np.load(root / 'arrays' / s['task'] / ('%s.npz' % s['ep'])),
                                                   json.load(open(root / 'truth' / s['task'] / ('%s.json' % s['ep']))))
        d, tr = cache[key]; t = s['t']; T = int(d['T']); fd = FUNC[s['task']]; o0 = tr.get(str(t))
        if o0 is None: continue
        mn = find(o0, fd['mov'][0], fd['tgt'][0]); tn = find(o0, fd['tgt'][0])
        if mn is None or tn is None: continue
        m0, g0 = pt(o0[mn], fd['mov'][1]), pt(o0[tn], fd['tgt'][1]); r0 = g0 - m0
        # 跟随基线：被插件跟最近的夹爪走（只在夹爪闭合且 TCP 在 5 cm 内时），目标不动
        Es = {arm: Tm(d['state/%s_ee_poses' % arm][t, :3], d['state/%s_ee_poses' % arm][t, 3:7]) for arm in P.ARMS}
        dist = {arm: np.linalg.norm(E[:3, :3] @ TCP + E[:3, 3] - m0) for arm, E in Es.items()}; arm = min(dist, key=dist.get)
        held = dist[arm] < 0.05 and d['state/%s_ee_joint_states' % arm][t, 0] < 0.85
        dr, fol, ok = np.zeros((P.K, 3)), np.zeros((P.K, 3)), True
        for k in range(P.K):
            tk = t + (k + 1) * P.TSTRIDE; ok_ = tr.get(str(tk))
            if ok_ is None or mn not in ok_ or tn not in ok_: ok = False; break
            dr[k] = (pt(ok_[tn], fd['tgt'][1]) - pt(ok_[mn], fd['mov'][1]) - r0) * 100
            if held:
                ea = d['action/%s_ee_poses' % arm][min(tk - 1, T - 1)]; D = Tm(ea[:3], ea[3:7]) @ np.linalg.inv(Es[arm])
                fol[k] = -((D[:3, :3] @ m0 + D[:3, 3]) - m0) * 100     # 目标不动，r 的变化 = -被插件位移
        if not ok: continue
        s.update(r0=torch.tensor(r0 * 100, dtype=torch.float32), dr=torch.tensor(dr, dtype=torch.float32), fol=torch.tensor(fol, dtype=torch.float32))
        out.append(s)
    return out


class K2F(P.PW):
    def __init__(s):
        super().__init__(); s.r_in = nn.Linear(3, 768); s.head = nn.Linear(768, 3)
        nn.init.zeros_(s.head.weight); nn.init.zeros_(s.head.bias)   # 初始输出 = 跟随基线

    def forward2(s, vis, prop, chunk, r0, fol):
        # 当前相对向量 r0（部署时来自 KPNet 工具）作为一个额外的本体 token 加到本体 token 上
        B = vis.shape[0]
        ctx = torch.cat([s.vis(vis) + s.typ[0], (s.prop(prop) + s.r_in(r0))[:, None] + s.typ[1], s.act(chunk) + s.typ[2]], 1) + s.pos
        h = s.enc(torch.cat([ctx, s.slot.expand(B, -1, -1) + s.typ[3]], 1), mask=s.mask)[:, -2 * P.K:].view(B, 2, P.K, -1)
        # 残差式：在"跟随夹爪"基线上只学修正量（v0 直接回归 Δr，在钥匙静止的样本上凭空预测运动，输给了"不变"基线）
        return h, fol + s.head(h.mean(1))                         # (B,K,3)


def batch(S, idx):
    return P.batch(S, idx) + [torch.stack([S[i][k] for i in idx]) for k in ('r0', 'dr', 'fol')]


@torch.no_grad()
def evaluate(model, S, split, dev):
    ix = [i for i, s in enumerate(S) if s['split'] == split]
    if not ix: return None
    model.eval(); E = {k: [] for k in ('model', 'nochange', 'follow')}; dec = {k: [] for k in ('model', 'follow')}
    for i0 in range(0, len(ix), 64):
        b = [x.to(dev) for x in batch(S, ix[i0:i0 + 64])]; _, pr = model.forward2(b[0], b[1], b[2], b[5], b[7]); dr, fol, r0 = b[6], b[7], b[5]
        for k, p in (('model', pr), ('nochange', torch.zeros_like(pr)), ('follow', fol)): E[k] += (p - dr)[:, -1].norm(dim=-1).tolist()
        # 改善判别：t+16 时 |r| 是否比现在小（真值）vs 预测
        truth = ((r0 + dr[:, -1]).norm(dim=-1) < r0.norm(dim=-1))
        for k, p in (('model', pr), ('follow', fol)): dec[k] += ((r0 + p[:, -1]).norm(dim=-1) < r0.norm(dim=-1)).eq(truth).float().tolist()
    model.train()
    return dict(n=len(ix), dr16_err_cm_median={k: float(np.median(v)) for k, v in E.items()}, improve_acc={k: float(np.mean(v)) for k, v in dec.items()})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--bs', type=int, default=32); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', required=True)
    a = ap.parse_args(); torch.manual_seed(a.seed); dev = torch.device(a.device)
    S = func_targets(P.load(['astra'])); tr = [i for i, s in enumerate(S) if s['split'] == 'TRAIN']
    cnt = {sp: sum(s['split'] == sp for s in S) for sp in ('TRAIN', 'DEV', 'TEST')}; print('samples', cnt, 'episodes', len({s['ep'] for s in S}), flush=True)
    model = K2F().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0); rng = np.random.default_rng(a.seed); log = []
    for ep in range(a.epochs):
        rng.shuffle(tr); tot = []
        for s in range(0, len(tr), a.bs):
            b = [x.to(dev) for x in batch(S, tr[s:s + a.bs])]; _, pr = model.forward2(b[0], b[1], b[2], b[5], b[7])
            L = F.smooth_l1_loss(pr, b[6]); opt.zero_grad(); L.backward(); opt.step(); tot.append(L.item())
        log.append(dict(epoch=ep, loss=float(np.mean(tot)))); print(log[-1], flush=True)
    res = dict(seed=a.seed, counts=cnt, train_log=log, level='dev, no judgement', tasks=sorted(FUNC),
               episodes=sorted({'%s/%s' % (s['task'], s['ep']) for s in S}), eval={sp: evaluate(model, S, sp, dev) for sp in ('DEV', 'TEST')})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, 'w'), indent=1); print(json.dumps(res['eval'], indent=1))


if __name__ == '__main__':
    main()
