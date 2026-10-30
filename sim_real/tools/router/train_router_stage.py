"""Router v2: RGB stage router over the C1 insertion funnel (rsiws-dc, 2026-10-29; dev until PREREG).
原理：普查显示插入类是一条漏斗、各任务卡在不同阶段，所以路由器不只判断"是不是插入类"，而是只凭 RGB 判断当前处于哪一阶段，
再路由到该阶段的工具（抓取 / 重抓与可达 / KP 伺服）。标签用仿真真值（被插件、接收方取自 c1_census 的逐回合结果），只在训练时用。
类别：0 未抓起，1 握着但离目标 >10 cm，2 近目标 3–10 cm，3 悬停 <=3 cm 且在上方，4 已对准 <=1 cm，5 非插入任务。
试管的目标点是槽的原点而非孔位（普查测量待修），所以 insert_tubes 最多标到 2。
评估：TEST、HOLD-L（布局 3/8/13）与 --hold-task 指定的整任务留出。  python -I train_router_stage.py --hold-task insert_key --out ../out/rs-dev-key.json"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import train_pw as P
from train_k2 import Tm, TCP
from train_router import Router, HOLD_L

NAMES = ['pre_grasp', 'held_far', 'near', 'hover', 'aligned', 'not_c1']
CEN = json.load(open(Path(__file__).parent.parent / 'out' / 'c1-census-v1.json'))
ROW = {r['episode']: r for r in CEN['rows']}


def stage_labels(S):
    cache, out = {}, []
    for s in S:
        if s['src'] != 'astra_v1': continue
        ep = str(s['ep']); r = ROW.get(ep)
        if r is None: s['stage'] = 5; out.append(s); continue                          # 非 C1 任务
        if not r.get('target') or not r.get('held'): continue
        if ep not in cache:
            root = P.SRC['astra_v1']; cache[ep] = (np.load(root / 'arrays' / s['task'] / ('%s.npz' % ep)), json.load(open(root / 'truth' / s['task'] / ('%s.json' % ep))))
        d, tr = cache[ep]; o = tr.get(str(s['t'])); o0 = tr[min(tr, key=int)]
        if o is None or r['held'] not in o or 'position' not in o[r['held']]: continue
        p = np.asarray(o[r['held']]['position']); tp = np.asarray(o[r['target']]['position']); z0 = o0[r['held']]['position'][2]
        Ps = []
        for arm in P.ARMS:
            e = d['state/%s_ee_poses' % arm][s['t']]; E = Tm(e[:3], e[3:7]); Ps.append(E[:3, :3] @ TCP + E[:3, 3])
        held = min(np.linalg.norm(p - q) for q in Ps) <= 0.06 and p[2] - z0 >= 0.02; dxy = float(np.linalg.norm(p[:2] - tp[:2]))
        if not held and dxy > 0.01: st = 0
        elif dxy <= 0.01: st = 4
        elif dxy <= 0.03 and p[2] > tp[2]: st = 3
        elif dxy <= 0.10: st = 2
        else: st = 1
        if s['task'] == 'insert_tubes': st = min(st, 2)
        s['stage'] = st; out.append(s)
    return out


@torch.no_grad()
def evaluate(model, S, ix, dev):
    if not ix: return None
    model.eval(); pred = []
    for i0 in range(0, len(ix), 256):
        b = [S[i] for i in ix[i0:i0 + 256]]; pred += model(torch.stack([x['vis'] for x in b]).float().to(dev), None).argmax(-1).cpu().tolist()
    model.train(); y = np.array([S[i]['stage'] for i in ix]); p = np.array(pred)
    rec = {NAMES[c]: (float((p[y == c] == c).mean()) if (y == c).any() else None) for c in range(6)}
    cm = [[int(((y == a) & (p == b)).sum()) for b in range(6)] for a in range(6)]
    present = [v for v in rec.values() if v is not None]
    return dict(n=len(ix), acc=float((p == y).mean()), macro_recall=float(np.mean(present)), recall=rec, confusion=cm)


class StageRouter(Router):
    def __init__(s):
        super().__init__(); s.head = torch.nn.Linear(256, 6)

    def forward(s, vis, prop=None, use_prop=False):
        return super().forward(vis, torch.zeros(vis.shape[0], 16, device=vis.device), False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--hold-task', default=''); ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--bs', type=int, default=64); ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--out', required=True); ap.add_argument('--save', default='')
    a = ap.parse_args(); torch.manual_seed(a.seed); dev = torch.device(a.device)
    S = stage_labels(P.load(['astra_v1'])); lay = {}
    for f in (P.SRC['astra_v1'] / 'episodes').glob('*.json'):
        r = json.load(open(f)); lay[r['episode']] = r.get('layout')
    for s in S:
        s['rsplit'] = 'HOLD-TASK' if s['task'] == a.hold_task else 'HOLD-L' if lay.get(str(s['ep'])) in HOLD_L else s['split'] if s['split'] in ('DEV', 'TEST') else 'TRAIN'
    tr = [i for i, s in enumerate(S) if s['rsplit'] == 'TRAIN']; ix = lambda sp: [i for i, s in enumerate(S) if s['rsplit'] == sp]
    cnt = {sp: np.bincount([S[i]['stage'] for i in ix(sp)], minlength=6).tolist() for sp in ('TRAIN', 'DEV', 'TEST', 'HOLD-L', 'HOLD-TASK')}
    print('hold-task', a.hold_task, 'class counts', cnt, flush=True)
    w = torch.tensor(1.0 / np.maximum(np.bincount([S[i]['stage'] for i in tr], minlength=6), 1), dtype=torch.float32); w = (w / w.sum() * 6).to(dev)   # 类别平衡
    model = StageRouter().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01); rng = np.random.default_rng(a.seed); log = []
    for ep in range(a.epochs):
        rng.shuffle(tr); tot = []
        for i0 in range(0, len(tr), a.bs):
            b = [S[i] for i in tr[i0:i0 + a.bs]]
            L = F.cross_entropy(model(torch.stack([x['vis'] for x in b]).float().to(dev)), torch.tensor([x['stage'] for x in b], device=dev), weight=w)
            opt.zero_grad(); L.backward(); opt.step(); tot.append(L.item())
        log.append(dict(epoch=ep, loss=float(np.mean(tot)))); print(log[-1], flush=True)
    res = dict(hold_task=a.hold_task, seed=a.seed, counts=cnt, names=NAMES, train_log=log, level='dev, no judgement',
               eval={sp: evaluate(model, S, ix(sp), dev) for sp in ('DEV', 'TEST', 'HOLD-L', 'HOLD-TASK')})
    if a.save: torch.save(dict(state_dict=model.state_dict(), hold_task=a.hold_task, names=NAMES), a.save)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps({sp: (e and {k: e[k] for k in ('n', 'acc', 'macro_recall', 'recall')}) for sp, e in res['eval'].items()}, indent=1))


if __name__ == '__main__':
    main()
