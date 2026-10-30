"""E1: RGB router to the C1 (precision insertion) tool chain (rsiws-dc, 2026-10-29; dev until PREREG-R1 is frozen).
原理：用户要求"过拟合一类实验，并保证 router 能只凭 RGB 路由到该类"。路由器只看部署输入（三路相机 DINOv3 池化 token + 本体），
输出两个概率：c1（当前场景属于精密插入类）和 route_c1（属于 C1 且手里已握住物体 = C1 插入工具此刻可用）。
标签只在训练时用仿真真值算：held = 夹爪闭合且最近的 rigid 物体在 TCP 5 cm 内。
泛化检验：留一 C1 任务（每折整任务留出 1 个 C1 任务 + 1 个非 C1 任务），看留出任务上的 AUROC 和误路由率——
路由器要认的是"这一类的视觉特征"，不是"这几个任务"。
  python -I train_router.py --fold 0 --epochs 15 --device cuda --out ../out/router-dev-f0.json"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import train_pw as P
from train_k2 import Tm, TCP

C1 = ['deposit_coin', 'fill_pen_holder', 'insert_key', 'insert_tubes', 'plug_in_charger']   # TASK-CATEGORIES-20261029 C1
HOLD_L = {3, 8, 13}
HELD = None   # --held-from-census：按普查 v1 的被插件名判"已握住"（R1 归因：笔筒任务里握笔筒也被算成已握住）


def labels(S):
    """给每个样本加 c1、held（任一臂）、route_c1；只用 astra_v1（有真值）。"""
    cache, out = {}, []
    for s in S:
        if s['src'] != 'astra_v1': continue
        key = (s['task'], s['ep'])
        if key not in cache:
            root = P.SRC['astra_v1']; cache[key] = (np.load(root / 'arrays' / s['task'] / ('%s.npz' % s['ep'])),
                                                      json.load(open(root / 'truth' / s['task'] / ('%s.json' % s['ep']))))
        d, tr = cache[key]; o = tr.get(str(s['t']))
        if o is None: continue
        rig = [np.asarray(v['position']) for v in o.values() if v.get('kind') == 'rigid' and 'position' in v]
        held = False
        if HELD is not None and s['task'] in C1:   # 只认该回合的被插件
            hn = HELD.get(str(s['ep'])); rig = [np.asarray(o[hn]['position'])] if hn and hn in o and 'position' in o[hn] else []
        for arm in P.ARMS:
            e = d['state/%s_ee_poses' % arm][s['t']]; E = Tm(e[:3], e[3:7]); tcp = E[:3, :3] @ TCP + E[:3, 3]
            if d['state/%s_ee_joint_states' % arm][s['t'], 0] < 0.85 and rig and min(np.linalg.norm(p - tcp) for p in rig) < 0.05: held = True
        c1 = s['task'] in C1
        s.update(c1=float(c1), held=float(held), route=float(c1 and held)); out.append(s)
    return out


def folds(tasks):
    non = sorted(t for t in tasks if t not in C1)
    return [(c, non[i % len(non)]) for i, c in enumerate(C1)]


class Router(nn.Module):
    def __init__(s, d=256, layers=2):
        super().__init__(); s.vis = nn.Linear(768, d); s.prop = nn.Linear(16, d); s.cls = nn.Parameter(torch.zeros(1, 1, d))
        s.pos = nn.Parameter(torch.zeros(1, 152, d)); s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True), layers)
        s.head = nn.Linear(d, 2)                                   # [c1, route_c1] 的 logit

    def forward(s, vis, prop, use_prop=True):
        B = vis.shape[0]; p = s.prop(prop) if use_prop else torch.zeros(B, s.prop.out_features, device=vis.device)
        x = torch.cat([s.cls.expand(B, -1, -1), s.vis(vis), p[:, None]], 1) + s.pos
        return s.head(s.enc(x)[:, 0])


THR = 0.5


def calibrate(e, max_fr=0.05):
    # 在 DEV 上取误路由 <= max_fr 的最小阈值（R1 归因：0.5 从未校准）
    neg = np.sort(e['scores'][e['labels'] == 0]); return float(neg[int(np.ceil((1 - max_fr) * len(neg))) - 1]) if len(neg) else 0.5


def auroc(y, p):
    y, p = np.asarray(y), np.asarray(p); pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0: return None
    r = np.argsort(np.argsort(np.r_[pos, neg])) + 1; return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def evaluate(model, S, ix, dev, use_prop):
    if not ix: return None
    model.eval(); P1, P2 = [], []
    for i0 in range(0, len(ix), 256):
        b = [S[i] for i in ix[i0:i0 + 256]]
        lo = model(torch.stack([x['vis'] for x in b]).float().to(dev), torch.stack([x['prop'] for x in b]).to(dev), use_prop)
        pr = torch.sigmoid(lo).cpu().numpy(); P1 += pr[:, 0].tolist(); P2 += pr[:, 1].tolist()
    model.train(); y1 = [S[i]['c1'] for i in ix]; y2 = [S[i]['route'] for i in ix]; p2 = np.asarray(P2)
    yr = np.asarray(y2); th = THR
    fr = float((p2[yr == 0] >= th).mean()) if (yr == 0).any() else None; hit = float((p2[yr == 1] >= th).mean()) if (yr == 1).any() else None
    return dict(n=len(ix), n_route=int(yr.sum()), auroc_c1=auroc(y1, P1), auroc_route=auroc(y2, P2), route_recall=hit, false_route=fr, threshold=th, scores=p2, labels=yr)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--fold', type=int, default=-1); ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--bs', type=int, default=64); ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-prop', action='store_true', help='只用 RGB（不给本体）'); ap.add_argument('--device', default='cuda'); ap.add_argument('--out', required=True)
    ap.add_argument('--held-from-census', action='store_true'); ap.add_argument('--calibrate', action='store_true')
    ap.add_argument('--save', default=''); ap.add_argument('--judge-new', default='', help='PREREG-R1：这些回合整体作判定集 NEW，其余全部训练')
    a = ap.parse_args(); torch.manual_seed(a.seed); dev = torch.device(a.device); use_prop = not a.no_prop
    global HELD, THR
    if a.held_from_census:
        HELD = {r['episode']: r.get('held') for r in json.load(open(Path(__file__).parent.parent / 'out' / 'c1-census-v1.json'))['rows']}
    S = labels(P.load(['astra_v1'])); fs = folds(sorted({s['task'] for s in S})); hc, hn = fs[a.fold] if a.fold >= 0 else ('', '')
    new = set(json.load(open(a.judge_new))) if a.judge_new else set()
    lay = {}
    for f in (P.SRC['astra_v1'] / 'episodes').glob('*.json'):
        r = json.load(open(f)); lay[r['episode']] = r.get('layout')
    for s in S:
        s['rsplit'] = 'NEW' if str(s['ep']) in new else 'TRAIN' if new and s['split'] != 'DEV' else 'HOLD-C1' if s['task'] == hc else 'HOLD-NON' if s['task'] == hn else 'HOLD-L' if lay.get(str(s['ep'])) in HOLD_L else \
            'TEST' if s['split'] == 'TEST' else 'DEV' if s['split'] == 'DEV' else 'TRAIN'
    tr = [i for i, s in enumerate(S) if s['rsplit'] == 'TRAIN']; ix = lambda sp: [i for i, s in enumerate(S) if s['rsplit'] == sp]
    SP = ('TRAIN', 'DEV', 'NEW') if new else ('TRAIN', 'DEV', 'TEST', 'HOLD-L', 'HOLD-C1', 'HOLD-NON')
    cnt = {sp: [len(ix(sp)), int(sum(S[i]['route'] for i in ix(sp)))] for sp in SP}
    print('fold', a.fold, 'held-out', hc, hn, 'counts [n, route]', cnt, flush=True)
    model = Router().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01); rng = np.random.default_rng(a.seed); log = []
    for ep in range(a.epochs):
        rng.shuffle(tr); tot = []
        for i0 in range(0, len(tr), a.bs):
            b = [S[i] for i in tr[i0:i0 + a.bs]]
            lo = model(torch.stack([x['vis'] for x in b]).float().to(dev), torch.stack([x['prop'] for x in b]).to(dev), use_prop)
            y = torch.tensor([[x['c1'], x['route']] for x in b], device=dev); L = F.binary_cross_entropy_with_logits(lo, y)
            opt.zero_grad(); L.backward(); opt.step(); tot.append(L.item())
        log.append(dict(epoch=ep, loss=float(np.mean(tot)))); print(log[-1], flush=True)
    # HOLD-ALL = 留出的 C1 任务 + 留出的非 C1 任务合在一起：误路由率与召回在这里看
    res = dict(fold=a.fold, held_out=[hc, hn], use_prop=use_prop, seed=a.seed, counts=cnt, train_log=log, level='dev, no judgement',
               eval={})
    if a.calibrate: THR = calibrate(evaluate(model, S, ix('DEV'), dev, use_prop)); res['threshold'] = THR
    res['eval'] = {sp: evaluate(model, S, ix(sp), dev, use_prop) for sp in SP[1:]}
    if new:   # 逐任务报告（不参与判定）
        for t in sorted({S[i]['task'] for i in ix('NEW')}): res['eval']['NEW/' + t] = evaluate(model, S, [i for i in ix('NEW') if S[i]['task'] == t], dev, use_prop)
    else: res['eval']['HOLD-ALL'] = evaluate(model, S, ix('HOLD-C1') + ix('HOLD-NON'), dev, use_prop)
    if a.save: torch.save(dict(state_dict=model.state_dict(), fold=a.fold, use_prop=use_prop, C1=C1), a.save)
    for e in res['eval'].values():
        if e: e.pop('scores', None); e.pop('labels', None)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, 'w'), indent=1); print(json.dumps(res['eval'], indent=1))


if __name__ == '__main__':
    main()
