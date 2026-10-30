"""PW arm of MVA v0: PhysWord-style word slots on frozen DINOv3 tokens (2026-10-29, rsiws-dc for the ee line; dev until PREREG frozen).
原理（PhysWord, NVIDIA 2026）：每只手臂在 t+4/8/12/16 各放一个"词槽"；词槽隐状态同时受两种监督——
  (1) 动词交叉熵（approach/pick/move/insert...，标签见 label_words.py），(2) 与真实未来帧潜变量的余弦对齐（不经 MLP，直接对隐状态）。
动词作为跨任务共享的语义锚，论文中让同一动作的潜表示更集中、对未见场景更稳。我们检验的是：它能否让"动作块 -> 视频变化"的预测迁移到未见任务。
与论文的差异（有意）：潜变量目标取"未来头部帧池化 token 减当前帧"（变化量 Δ），否则相邻帧几乎相同，"照抄当前帧"即可得到近 1 的余弦，测不出动作的作用。
部署输入：当前与 4 步前三路相机 DINOv3 token、本体、候选动作块；未来帧只作训练目标。
变体：O = 只有潜变量损失；W = 只有动词损失；WO = 两者（论文主方法）。
评估（HOLD-T 为整任务留出）：动词逐槽准确率；未来 Δ 余弦；动作检索（同一状态下换成同回合其他时刻的 15 个动作块，看真动作块预测的 Δ 是否最接近真实 Δ）。
  python -I train_pw.py --variant WO --epochs 20 --device cpu --out ../out/pw-smoke-WO.json"""
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from label_words import VOCAB, label_arm

SRC = {'demo': Path('/lizekai/RSIws/experiments/mva-20261029/data'), 'astra': Path('/lizekai/RSIws/experiments/ee-physword-20261029/data-astra'),
       'astra_v1': Path('/lizekai/RSIws/experiments/mva-20261029/astra_v1')}
HOLD_T = {'insert_tubes', 'play_Xylophone'}
TSTRIDE, K, H = 4, 4, 16          # token 每 4 步一帧；每臂 K=4 个词槽，覆盖 t+4..t+16
ARMS = ('left', 'right')


def split_of(task, ep):
    # 与 prep_mva.split_of 相同，划分在任何模型之前已固定
    # 示范回合号为 int（与 prep_mva 相同的 '%s/%d'），Astra 回合为 ctx-... 字符串
    if task in HOLD_T: return 'HOLD-T'
    h = int(hashlib.sha1(('%s/%s' % (task, ep)).encode()).hexdigest(), 16) % 5; return 'TEST' if h == 4 else 'DEV' if h == 3 else 'TRAIN'


def pool_cam(tok):
    # (300,768) 的 20x15 patch 网格 -> 5x5=25 个 token（平均池化），压缩序列长度
    g = torch.from_numpy(tok.astype(np.float32)).view(15, 20, 768).permute(2, 0, 1)
    return F.adaptive_avg_pool2d(g, (5, 5)).flatten(1).T


def chunk_feats(d, t, T):
    # 动作块：t+1..t+H 每步两臂 [末端位置相对当前位置、四元数、夹爪]，超出回合末尾的步重复最后一步
    rows = []
    for a in ARMS:
        p0 = d['state/%s_ee_poses' % a][t, :3]
        idx = np.minimum(np.arange(t + 1, t + H + 1), T - 1)
        ap = d['action/%s_ee_poses' % a][idx]; g = d['action/%s_ee_joint_states' % a][idx]
        rows.append(np.c_[ap[:, :3] - p0, ap[:, 3:7], g])
    return np.concatenate(rows, 1).astype(np.float32)            # (H, 16)


def load(sources):
    S = []
    for tf in sorted(f for s in sources for f in SRC[s].glob('tokens/*/*.npy') if '.tmp' not in f.name):
        task, src = tf.parent.name, tf.parents[2].name; ep = int(tf.stem) if tf.stem.isdigit() else tf.stem
        af = tf.parents[2] / 'arrays' / task / ('%s.npz' % tf.stem)
        if not af.exists(): continue
        d = np.load(af); T = int(d['T']); steps = json.load(open(str(tf)[:-4] + '.steps.json'))
        # 池化缓存：完整 token 每回合约 140 MB，多个作业各读一遍会压垮 NFS；池化后 (N,3,25,768) 只算一次
        pf = tf.parents[2] / 'pooled' / task / tf.name
        if not pf.exists():
            tok = np.load(tf, mmap_mode='r'); pf.parent.mkdir(parents=True, exist_ok=True)
            pooled = np.stack([np.stack([pool_cam(tok[i, c]).numpy() for c in range(3)]) for i in range(len(steps))]).astype(np.float16)
            tmp = pf.with_suffix('.%d.tmp.npy' % os.getpid()); np.save(tmp, pooled); os.replace(tmp, pf)
        pooled = np.load(pf)
        words = {a: label_arm(d, a, o)[0] for a, o in (('left', 'right'), ('right', 'left'))}
        head = pooled[:, 0].astype(np.float32).mean(1)   # 头部视角池化潜变量（15x20 等分为 5x5，均值与全 token 均值相同）
        for i in range(1, len(steps) - K):
            t = steps[i]
            if steps[i + K] != t + H: continue
            prop = np.concatenate([np.r_[d['state/%s_ee_poses' % a][t], d['state/%s_ee_joint_states' % a][t]] for a in ARMS]).astype(np.float32)
            S.append(dict(task=task, ep=ep, src=src, split=split_of(task, ep), t=t,
                          vis=torch.from_numpy(pooled[[i - 1, i]].reshape(150, 768)),     # (150,768) fp16，顺序同前：帧 -> 相机 -> patch
                          prop=torch.from_numpy(prop), chunk=torch.from_numpy(chunk_feats(d, t, T)),
                          word=torch.tensor([[VOCAB.index(words[a][min(t + (k + 1) * TSTRIDE, T - 1)]) for k in range(K)] for a in ARMS]),
                          dlat=torch.from_numpy(np.stack([head[i + k + 1] - head[i] for k in range(K)]))))  # (K,768)
    return S


class PW(nn.Module):
    def __init__(s, d=768, layers=4):
        super().__init__()
        s.vis = nn.Linear(768, d); s.prop = nn.Linear(16, d); s.act = nn.Linear(16, d)
        s.typ = nn.Parameter(torch.zeros(4, d)); s.pos = nn.Parameter(torch.zeros(150 + 1 + H, d))
        s.slot = nn.Parameter(torch.randn(2 * K, d) * 0.02)       # 左臂 K 个槽在前，右臂在后
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 4 * d, 0.1, batch_first=True, norm_first=True), layers)
        s.word = nn.Linear(d, len(VOCAB))
        n = 150 + 1 + H; m = torch.zeros(n + 2 * K, n + 2 * K, dtype=torch.bool)
        m[:n, n:] = True                                          # 上下文不看词槽
        for a in range(2):
            for k in range(K):
                r = n + a * K + k
                m[r, n:] = True; m[r, n + a * K: r + 1] = False   # 同臂内对更早的槽因果可见，跨臂不可见
        s.register_buffer('mask', m)

    def forward(s, vis, prop, chunk):
        B = vis.shape[0]
        ctx = torch.cat([s.vis(vis) + s.typ[0], s.prop(prop)[:, None] + s.typ[1], s.act(chunk) + s.typ[2]], 1) + s.pos
        h = s.enc(torch.cat([ctx, s.slot.expand(B, -1, -1) + s.typ[3]], 1), mask=s.mask)[:, -2 * K:]
        return h.view(B, 2, K, -1)                                # 词槽隐状态，直接与 Δ 潜变量对齐


def batch(S, idx):
    return [torch.stack([S[i][k] for i in idx]).float() if k == 'vis' else torch.stack([S[i][k] for i in idx]) for k in ('vis', 'prop', 'chunk', 'word', 'dlat')]


def losses(model, b, variant):
    vis, prop, chunk, word, dlat = b; h = model(vis, prop, chunk)
    ce = F.cross_entropy(model.word(h).flatten(0, 2), word.flatten())
    lat = (1 - F.cosine_similarity(h, dlat[:, None].expand(-1, 2, -1, -1), dim=-1)).mean()
    return {'O': lat, 'W': ce, 'WO': ce + lat}[variant], ce, lat


@torch.no_grad()
def evaluate(model, S, split, dev, maj, n_cand=16, seed=0):
    ix = [i for i, s in enumerate(S) if s['split'] == split]
    if not ix: return None
    model.eval(); rng = np.random.default_rng(seed); acc, cos, cos_copy, hit = [], [], [], []
    for i in ix:
        b = [x.to(dev) for x in batch(S, [i])]; h = model(*b[:3])
        acc.append((model.word(h).argmax(-1) == b[3]).float().mean().item())
        cos.append(F.cosine_similarity(h.mean(1), b[4], dim=-1).mean().item())
        # 动作检索：同一回合其他时刻的动作块作干扰项
        pool = [j for j in range(len(S)) if S[j]['task'] == S[i]['task'] and S[j]['ep'] == S[i]['ep'] and abs(S[j]['t'] - S[i]['t']) >= H]
        if len(pool) < n_cand - 1: continue
        cand = [i] + list(rng.choice(pool, n_cand - 1, replace=False))
        ch = torch.stack([S[j]['chunk'] for j in cand]).to(dev)
        hc = model(b[0].expand(n_cand, -1, -1), b[1].expand(n_cand, -1), ch).mean(1)
        sc = F.cosine_similarity(hc, b[4].expand(n_cand, -1, -1), dim=-1).mean(-1)
        hit.append(float(sc.argmax().item() == 0))
    model.train()
    # 多数类基线：所有槽都猜训练集最常见的词
    base = float(np.mean([(S[i]['word'] == maj).float().mean().item() for i in ix]))
    return dict(n=len(ix), word_acc=float(np.mean(acc)), word_acc_majority=base, dlat_cos=float(np.mean(cos)),
                retrieval_top1=float(np.mean(hit)) if hit else None, retrieval_n=len(hit))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--variant', default='WO', choices=['O', 'W', 'WO'])
    ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--bs', type=int, default=32); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--sources', default='demo', help='comma list of demo,astra')
    ap.add_argument('--save', default='', help='保存权重（交付用）')
    ap.add_argument('--device', default='cpu'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', required=True)
    a = ap.parse_args(); torch.manual_seed(a.seed); dev = torch.device(a.device)
    S = load(a.sources.split(',')); tr = [i for i, s in enumerate(S) if s['split'] == 'TRAIN']
    cnt = {sp: sum(s['split'] == sp for s in S) for sp in ('TRAIN', 'DEV', 'TEST', 'HOLD-T')}
    print('samples', cnt, 'episodes', len({(s['task'], s['ep']) for s in S}), flush=True)
    maj = int(torch.cat([S[i]['word'].flatten() for i in tr]).mode().values)
    model = PW().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    rng = np.random.default_rng(a.seed); log = []
    for ep in range(a.epochs):
        rng.shuffle(tr); tot = []
        for s in range(0, len(tr), a.bs):
            b = [x.to(dev) for x in batch(S, tr[s:s + a.bs])]; L, ce, lat = losses(model, b, a.variant)
            opt.zero_grad(); L.backward(); opt.step(); tot.append((L.item(), ce.item(), lat.item()))
        m = np.mean(tot, 0); log.append(dict(epoch=ep, loss=m[0], ce=m[1], lat=m[2])); print(log[-1], flush=True)
    res = dict(variant=a.variant, seed=a.seed, sources=a.sources, counts=cnt, train_log=log, level='dev smoke, no judgement',
               episodes=sorted({'%s/%s/%s' % (s['src'], s['task'], s['ep']) for s in S}),
               eval={sp: evaluate(model, S, sp, dev, maj) for sp in ('DEV', 'TEST', 'HOLD-T')})
    if a.save: torch.save(dict(state_dict=model.state_dict(), variant=a.variant, seed=a.seed, sources=a.sources, vocab=VOCAB, K=K, H=H, TSTRIDE=TSTRIDE), a.save)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, 'w'), indent=1); print(json.dumps(res['eval'], indent=1))


if __name__ == '__main__':
    main()
