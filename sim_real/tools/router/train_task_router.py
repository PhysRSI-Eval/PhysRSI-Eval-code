"""Router 第 0 层：只凭 RGB（一个决策点的三路相机 DINOv3 池化 token）判定是 RoboDojo 42 个任务中的哪一个，并给出技能类别 A–G。
数据：data-native/pooled（extract_native_turns.py，Astra 每个回合均匀 16 个决策点）。不用任务 id、文本、本体或真值。
划分：回合按 sha1(episode)%5（4 TEST，3 DEV，其余 TRAIN）；布局 3/8/13 整体留作 HOLD-L（只有 6 个任务覆盖多布局，
这是"换布局还认得出"的检验）。报告：任务准确率、类别准确率、首个决策点（开局一眼）准确率、按类别分开的准确率、混淆最多的任务对。
  python -I train_task_router.py --out ../out/task-router-dev.json [--save ...]"""
import argparse, hashlib, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

D = Path('/lizekai/RSIws/experiments/ee-physword-20261029/data-native/pooled'); DO = Path('/lizekai/RSIws/experiments/ee-physword-20261029/data-official/pooled')
HOLD_L = {3, 8, 13}
CAT = {}   # 任务 -> 类别（TASK-TAXONOMY-ROBODOJO.md，按主难点）
for c, ts in dict(A='plug_in_charger insert_key insert_tubes deposit_coin fasten_screws fill_pen_holder make_toast hang_mugs store_tools_in_toolbox store_laptop_and_headphones',
                  B='pour_liquid_into_cup pour_by_language pour_balls_into_vase', C='play_Xylophone sweep_blocks align_blocks', D='push_T',
                  E='classify_objects classify_objects_by_language pack_objects_into_box organize_table stack_blocks stack_blocks_by_language build_tower stack_bowls play_stacking_toy sort_nesting_dolls_by_size fill_egg_holder arrange_largest_number solve_equation put_bottles_into_dustbin general_pickup swap_T swap_blocks',
                  F='cover_blocks imitate_sorting_sequence match_and_pick_from_conveyor pick_from_conveyor_by_image make_kong play_tic_tac_toe press_by_number',
                  G='fold_clothes').items():
    for t in ts.split(): CAT[t] = c
CATS = sorted(set(CAT.values()))


def split_of(ep, lay):
    h = int(hashlib.sha1(ep.encode()).hexdigest(), 16) % 5
    if ep.startswith('official/'): return 'OFF-TEST' if h >= 3 else 'OFF-TRAIN'     # 官方示范：场景随机化，换场景检验
    if lay in HOLD_L: return 'HOLD-L'
    return 'TEST' if h == 4 else 'DEV' if h == 3 else 'TRAIN'


def load(official):
    X, Y, meta = [], [], []
    for f in sorted(list(D.glob('*/*.npy')) + (list(DO.glob('*/*.npy')) if official != 'none' else [])):
        if '.tmp' in f.name: continue
        m = json.load(open(str(f)[:-4] + '.json')); a = np.load(f)
        for k in range(len(a)):
            X.append(a[k].reshape(75, 768)); Y.append(m['task']); meta.append((m['episode'], split_of(m['episode'], m.get('layout')), k))
    return np.stack(X), Y, meta


class TaskRouter(nn.Module):
    def __init__(s, n_task, n_cat, d=256):
        super().__init__(); s.inp = nn.Linear(768, d); s.cls = nn.Parameter(torch.zeros(1, 1, d)); s.pos = nn.Parameter(torch.zeros(1, 76, d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True), 2)
        s.task = nn.Linear(d, n_task); s.cat = nn.Linear(d, n_cat)

    def forward(s, x):
        h = s.enc(torch.cat([s.cls.expand(x.shape[0], -1, -1), s.inp(x)], 1) + s.pos)[:, 0]; return s.task(h), s.cat(h)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--bs', type=int, default=256); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--official', default='eval', choices=['none', 'eval', 'train'], help='eval：官方示范只作检验；train：OFF-TRAIN 也参与训练')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--device', default='cuda'); ap.add_argument('--out', required=True); ap.add_argument('--save', default='')
    a = ap.parse_args(); torch.manual_seed(a.seed); dev = torch.device(a.device)
    X, Y, meta = load(a.official); tasks = sorted(set(Y)); ti = {t: i for i, t in enumerate(tasks)}
    yt = np.array([ti[t] for t in Y]); yc = np.array([CATS.index(CAT.get(t, 'E')) for t in Y]); sp = np.array([m[1] for m in meta]); first = np.array([m[2] == 0 for m in meta])
    tr = np.flatnonzero((sp == 'TRAIN') | ((sp == 'OFF-TRAIN') & (a.official == 'train'))); print('samples', len(Y), Counter(sp), 'tasks', len(tasks), flush=True)
    cw = np.bincount(yt[tr], minlength=len(tasks)); w = torch.tensor(1.0 / np.maximum(cw, 1), dtype=torch.float32); w = (w / w.mean()).to(dev)   # 回合数差 100 倍，类别平衡
    model = TaskRouter(len(tasks), len(CATS)).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01); rng = np.random.default_rng(a.seed); log = []
    Xt = torch.from_numpy(X)
    for ep in range(a.epochs):
        rng.shuffle(tr); tot = []
        for i0 in range(0, len(tr), a.bs):
            b = tr[i0:i0 + a.bs]; lt, lc = model(Xt[b].float().to(dev))
            L = F.cross_entropy(lt, torch.from_numpy(yt[b]).to(dev), weight=w) + F.cross_entropy(lc, torch.from_numpy(yc[b]).to(dev))
            opt.zero_grad(); L.backward(); opt.step(); tot.append(L.item())
        log.append(dict(epoch=ep, loss=float(np.mean(tot)))); print(log[-1], flush=True)
    model.eval(); P, PC = [], []
    with torch.no_grad():
        for i0 in range(0, len(Y), 1024):
            lt, lc = model(Xt[i0:i0 + 1024].float().to(dev)); P += lt.argmax(-1).cpu().tolist(); PC += lc.argmax(-1).cpu().tolist()
    P, PC = np.array(P), np.array(PC); ev = {}
    for s in ('DEV', 'TEST', 'HOLD-L', 'OFF-TEST'):
        ix = sp == s
        if not ix.any(): continue
        conf = Counter((tasks[yt[j]], tasks[P[j]]) for j in np.flatnonzero(ix & (P != yt)))
        ev[s] = dict(n=int(ix.sum()), task_acc=float((P[ix] == yt[ix]).mean()), cat_acc=float((PC[ix] == yc[ix]).mean()),
                     first_turn_task_acc=float((P[ix & first] == yt[ix & first]).mean()) if (ix & first).any() else None,
                     per_cat_task_acc={c: float((P[ix & (yc == k)] == yt[ix & (yc == k)]).mean()) for k, c in enumerate(CATS) if (ix & (yc == k)).any()},
                     per_task_acc={tasks[k]: float((P[ix & (yt == k)] == k).mean()) for k in range(len(tasks)) if (ix & (yt == k)).any()},
                     top_confusions=[[a_, b_, n_] for (a_, b_), n_ in conf.most_common(8)])
    res = dict(seed=a.seed, official=a.official, tasks=tasks, cats=CATS, counts=dict(Counter(sp)), train_log=log, level='dev', eval=ev)
    if a.save: torch.save(dict(state_dict=model.state_dict(), tasks=tasks, cats=CATS, cat_map=CAT), a.save)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps({s: {k: e[k] for k in ('n', 'task_acc', 'cat_acc', 'first_turn_task_acc', 'per_cat_task_acc', 'top_confusions')} for s, e in ev.items()}, indent=1))


if __name__ == '__main__':
    main()
