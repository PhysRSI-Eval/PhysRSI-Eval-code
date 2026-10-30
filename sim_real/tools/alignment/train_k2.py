"""K2: keypoint-space action–video alignment (v2 plan rung K2, assigned to rsiws-dc 09:4xZ; dev until PREREG-K2 is frozen).
原理：PW-v1 的对齐目标是头部画面池化潜变量的变化，粗、以场景为中心；K2 改为以物体为中心、毫米级的目标——
每只手臂在 t 时刻离夹爪中心（TCP）15 cm 内最近的可动物体，取 4 个关键点（原点 + 3 个 5 cm 轴向点，K1 的 CAD 点出来后替换），
预测它们在 t+4/8/12/16 相对 t 的 3D 位移（cm）。词槽隐状态经线性头读出位移，同时可保留动词与潜变量监督（PhysWord 式）。
训练目标用仿真真值（训练时可用）；部署输入不变：RGB token、本体、动作块。
基线：不变（0）；跟随夹爪（夹爪闭合且物体在 5 cm 内时，物体按动作块中末端位姿的刚体变换移动）——只用部署可得的信息。
重点子集 "deviate"：真实位移与跟随夹爪基线相差 > 1 cm（松手、滑脱、推动、碰撞），这才是需要"懂后果"的样本。
  python -I train_k2.py --variant WK --epochs 40 --device cuda --out ../out/k2-dev-WK.json"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import train_pw as P

TCP = np.array([0.14, 0.0, 0.0]); S_AX = 0.05; NEAR, HELD, DEV_CM = 0.15, 0.05, 1.0
OFS = np.array([[0, 0, 0], [S_AX, 0, 0], [0, S_AX, 0], [0, 0, S_AX]])


def Tm(p, q):   # q 为 wxyz；preview 环境没有 scipy，直接用四元数公式
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q); M = np.eye(4); M[:3, 3] = p
    M[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                 [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                 [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    return M


def kps(o): M = Tm(o['position'], o['quaternion_wxyz']); return (M[:3, :3] @ OFS.T).T + M[:3, 3]


def targets(S):
    """给每个样本加 kp (2,K,4,3) cm、mask (2,)、follow 基线 (2,K,4,3) cm。只处理 astra 来源（有真值）。"""
    cache = {}
    for s in S:
        s['kp'] = torch.zeros(2, P.K, 4, 3); s['fol'] = torch.zeros(2, P.K, 4, 3); s['m'] = torch.zeros(2)
        if s['src'] not in ('data-astra', 'astra_v1'): continue
        key = (s['task'], s['ep'])
        if key not in cache:
            root = P.SRC['astra' if s['src'] == 'data-astra' else 'astra_v1']; cache[key] = (np.load(root / 'arrays' / s['task'] / ('%s.npz' % s['ep'])),
                                                   json.load(open(root / 'truth' / s['task'] / ('%s.json' % s['ep']))))
        d, tr = cache[key]; t = s['t']; T = int(d['T']); o0 = tr.get(str(t))
        if o0 is None: continue
        rig = [n for n, v in o0.items() if v.get('kind') == 'rigid' and 'position' in v]
        for a, arm in enumerate(P.ARMS):
            e = d['state/%s_ee_poses' % arm]; E0 = Tm(e[t, :3], e[t, 3:7]); tcp = E0[:3, :3] @ TCP + E0[:3, 3]
            dist = {n: np.linalg.norm(np.asarray(o0[n]['position']) - tcp) for n in rig}
            if not dist or min(dist.values()) > NEAR: continue
            n = min(dist, key=dist.get); k0 = kps(o0[n]); closed = d['state/%s_ee_joint_states' % arm][t, 0] < 0.85
            for k in range(P.K):
                tk = t + (k + 1) * P.TSTRIDE; ok = tr.get(str(tk)) or tr.get(str(min(tk, T - (T % 2))))
                if ok is None or n not in ok or 'position' not in ok[n]: break
                s['kp'][a, k] = torch.from_numpy((kps(ok[n]) - k0) * 100).float()
                if closed and dist[n] < HELD:   # 跟随夹爪：用动作块中的末端目标位姿（部署可得）
                    ea = d['action/%s_ee_poses' % arm][min(tk - 1, T - 1)]; Ek = Tm(ea[:3], ea[3:7]); D = Ek @ np.linalg.inv(E0)
                    s['fol'][a, k] = torch.from_numpy(((D[:3, :3] @ k0.T).T + D[:3, 3] - k0) * 100).float()
            else:
                s['m'][a] = 1.0
    return S


class K2(P.PW):
    def __init__(s, mode='direct'):
        super().__init__(); s.kp = nn.Linear(768, 12); s.gate = nn.Linear(768, 1); s.mode = mode
        if mode == 'res': nn.init.zeros_(s.kp.weight); nn.init.zeros_(s.kp.bias)   # 残差式：初始预测 = 跟随基线


def run(model, b):
    # 三种方式：direct 直接预测；res 在跟随基线上学残差（开发结果：被基线当捷径，偏离样本学不到）；
    # gate 两个专家（跟随夹爪 / 直接预测）+ 门控，门控学"此刻物体会不会偏离跟随"（用户的 MoE 思路）
    vis, prop, chunk, word, dlat, kp, m, fol = b; h = model(vis, prop, chunk); d = model.kp(h).view(*h.shape[:3], 4, 3)
    if model.mode == 'res': return h, fol + d
    if model.mode == 'gate': g = torch.sigmoid(model.gate(h))[..., None]; return h, g * d + (1 - g) * fol, d, g
    return h, d


def batch(S, idx):
    return P.batch(S, idx) + [torch.stack([S[i][k] for i in idx]) for k in ('kp', 'm', 'fol')]


def losses(model, b, variant):
    out = run(model, b); h, pk = out[0], out[1]; word, dlat, kp, m, fol = b[3], b[4], b[5], b[6], b[7]
    ml = lambda x: (x.mean((-1, -2, -3)) * m).sum() / m.sum().clamp(min=1)
    lk = ml(F.smooth_l1_loss(pk, kp, reduction='none'))
    if model.mode == 'gate':
        d, g = out[2], out[3]; dev = ((fol - kp).norm(dim=-1).mean(-1) > DEV_CM).float()          # (B,2,K) 偏离标签
        lk = lk + ml(F.smooth_l1_loss(d, kp, reduction='none')) + ml(F.binary_cross_entropy(g[..., 0, 0], dev, reduction='none')[..., None, None])
    ce = F.cross_entropy(model.word(h).flatten(0, 2), word.flatten())
    lat = (1 - F.cosine_similarity(h, dlat[:, None].expand(-1, 2, -1, -1), dim=-1)).mean()
    return {'K': lk, 'KR': lk, 'KG': lk, 'WK': lk + ce, 'WOK': lk + ce + lat}[variant]


@torch.no_grad()
def evaluate(model, S, split, dev, n_cand=16, seed=0):
    ix = [i for i, s in enumerate(S) if s['split'] == split and s['m'].sum() > 0]
    if not ix: return None
    model.eval(); rng = np.random.default_rng(seed); E = {k: [] for k in ('model', 'nochange', 'follow')}; Dv = {k: [] for k in E}; hit = []
    for i in ix:
        b = [x.to(dev) for x in batch(S, [i])]; pk = run(model, b)[1]; kp, m, fol = b[5], b[6], S[i]['fol'].to(dev)[None]
        for a in range(2):
            if m[0, a] == 0: continue
            err = lambda p: (p[0, a] - kp[0, a]).norm(dim=-1).mean().item()
            e = dict(model=err(pk), nochange=err(torch.zeros_like(pk)), follow=err(fol))
            dv = (fol[0, a] - kp[0, a]).norm(dim=-1).mean().item() > DEV_CM
            for k in E: E[k].append(e[k]); (Dv[k].append(e[k]) if dv else None)
        pool = [j for j in range(len(S)) if S[j]['task'] == S[i]['task'] and S[j]['ep'] == S[i]['ep'] and abs(S[j]['t'] - S[i]['t']) >= P.H]
        if len(pool) < n_cand - 1: continue
        cand = [i] + list(rng.choice(pool, n_cand - 1, replace=False)); ch = torch.stack([S[j]['chunk'] for j in cand]).to(dev)
        # 检索：每个候选动作块用它自己时刻的跟随值（跟随基线依赖动作块）
        bc = [b[0].expand(n_cand, -1, -1), b[1].expand(n_cand, -1), ch] + [x.expand(n_cand, *x.shape[1:]) for x in b[3:7]] + [torch.stack([S[j]['fol'] for j in cand]).to(dev)]
        pc = run(model, bc)[1]
        sc = -((pc - kp).norm(dim=-1).mean((-1, -2)) * m).sum(-1); hit.append(float(sc.argmax().item() == 0))
    model.train(); med = lambda v: float(np.median(v)) if v else None
    return dict(n_arm_samples=len(E['model']), n_deviate=len(Dv['model']), retrieval_top1=float(np.mean(hit)) if hit else None,
                err_cm_median={k: med(v) for k, v in E.items()}, err_cm_median_deviate={k: med(v) for k, v in Dv.items()})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--variant', default='WK', choices=['K', 'KR', 'KG', 'WK', 'WOK'])
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--bs', type=int, default=32); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--sources', default='astra'); ap.add_argument('--hold2', default=''); ap.add_argument('--tasks', default=''); ap.add_argument('--test-new', action='store_true')
    ap.add_argument('--save', default=''); ap.add_argument('--device', default='cuda'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', required=True)
    a = ap.parse_args(); torch.manual_seed(a.seed); dev = torch.device(a.device)
    S = targets(P.load(a.sources.split(',')))
    if a.tasks: S = [x for x in S if x['task'] in set(a.tasks.split(','))]   # 过拟合一类：只用指定类别的任务
    # 正式判定的划分（PREREG-K2）：--hold2 中的任务整任务留出（开发时从未按任务看过）；TEST-new = TEST 中第一轮导出没有的新回合
    hold2 = set(filter(None, a.hold2.split(',')))
    old = {f.stem for f in (P.SRC['astra'] / 'episodes').glob('*.json')} if a.test_new else set()
    for s_ in S:
        if s_['task'] in hold2: s_['split'] = 'HOLD-T2'
        elif s_['split'] == 'TEST' and a.test_new and str(s_['ep']) not in old: s_['split'] = 'TEST-new'
    tr = [i for i, s in enumerate(S) if s['split'] == 'TRAIN']
    SPL = ('TRAIN', 'DEV', 'TEST', 'TEST-new', 'HOLD-T', 'HOLD-T2')
    cnt = {sp: [sum(s['split'] == sp for s in S), int(sum(s['m'].sum().item() for s in S if s['split'] == sp))] for sp in SPL}
    print('samples [n, engaged arm-samples]', cnt, flush=True)
    model = K2({'KR': 'res', 'KG': 'gate'}.get(a.variant, 'direct')).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0); rng = np.random.default_rng(a.seed); log = []
    for ep in range(a.epochs):
        rng.shuffle(tr); tot = []
        for s in range(0, len(tr), a.bs):
            L = losses(model, [x.to(dev) for x in batch(S, tr[s:s + a.bs])], a.variant); opt.zero_grad(); L.backward(); opt.step(); tot.append(L.item())
        log.append(dict(epoch=ep, loss=float(np.mean(tot)))); print(log[-1], flush=True)
    res = dict(variant=a.variant, seed=a.seed, sources=a.sources, residual=True, counts=cnt, train_log=log, level='dev, no judgement',
               episodes=sorted({'%s/%s' % (s['task'], s['ep']) for s in S}), eval={sp: evaluate(model, S, sp, dev) for sp in SPL[1:]})
    if a.save: torch.save(dict(state_dict=model.state_dict(), variant=a.variant, seed=a.seed, mode=model.mode), a.save)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, 'w'), indent=1); print(json.dumps(res['eval'], indent=1))


if __name__ == '__main__':
    main()
