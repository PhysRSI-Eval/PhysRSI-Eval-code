"""Task-ID router dev: 42-way task classification from DINOv3 features of head + wrists at step t (and the mean over steps <= t), grouped
5-fold CV by context (ctx-<hash>-<n> = seed/layout context; random.Random(20261030)). MLP (4608 -> 512 -> 42, dropout 0.2), AdamW 1e-3,
40 epochs, class-balanced. Reports accuracy per t, per task, top confusion pairs.  -> RESULT-taskid-dev.json"""
import glob, json, random, collections
import numpy as np, torch, torch.nn as nn
T_STEPS = (0, 10, 20, 40, 80)
X, OK, EP = [], [], []
for f in sorted(glob.glob('feat/shard-*.npz')):
    z = np.load(f); X.append(z['X'].astype(np.float32)); OK.append(z['ok']); EP += list(z['episodes'])
X = np.concatenate(X); OK = np.concatenate(OK); idx = {e['episode']: e for e in json.load(open('episode_index.json'))}
task = np.array([idx[e]['task'] for e in EP]); ctx = np.array(['-'.join(e.split('-')[:3]) for e in EP]); T = sorted(set(task)); ti = {t: i for i, t in enumerate(T)}; y = np.array([ti[t] for t in task])
nctx = collections.Counter(); [nctx.update([t]) for t in {(t_, c_) for t_, c_ in zip(task, ctx)}]
lay = np.array([idx[e]['layout'] for e in EP]); nlay = {t_: len(set(lay[task == t_])) for t_ in set(task)}
# v2 grouping (measurement fix): 16 tasks have ONE context -> ctx-grouped CV held them out entirely (acc 0 = unseen task, not a router error).
# group = context if the task has >= 5 contexts; else (context, layout) if >= 2 layouts; else the episode (within-context split, flagged)
grp = np.array([c_ if nctx[t_] >= 5 else ('%s|L%d' % (c_, l_) if nlay[t_] >= 2 else e_) for t_, c_, l_, e_ in zip(task, ctx, lay, EP)])
WITHIN = sorted(t_ for t_ in set(task) if nctx[t_] < 5 and nlay[t_] < 2)
cs = sorted(set(grp)); random.Random(20261030).shuffle(cs); fold = {c: i % 5 for i, c in enumerate(cs)}; fo = np.array([fold[c] for c in grp])
dev = torch.device('cuda:0')
def fit_pred(Xa, tr, te):
    mu, sd = Xa[tr].mean(0), Xa[tr].std(0) + 1e-6; A = torch.tensor(((Xa - mu) / sd).astype(np.float32), device=dev); Y = torch.tensor(y, device=dev)
    cnt = np.bincount(y[tr], minlength=len(T)); w = torch.tensor(1.0 / np.maximum(cnt, 1), dtype=torch.float32, device=dev); w = w / w.mean()
    torch.manual_seed(0); m = nn.Sequential(nn.Linear(Xa.shape[1], 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, len(T))).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
    trt = torch.tensor(tr, device=dev)
    for ep in range(40):
        perm = trt[torch.randperm(len(tr), device=dev)]
        for i0 in range(0, len(tr), 256):
            b = perm[i0:i0 + 256]; opt.zero_grad(); l = nn.functional.cross_entropy(m(A[b]), Y[b], weight=w); l.backward(); opt.step()
    m.eval()
    with torch.no_grad(): return m(A[torch.tensor(te, device=dev)]).argmax(1).cpu().numpy()
res = {}
for j, t in enumerate(T_STEPS):
    for mode in ('at_t', 'mean_upto_t'):
        if mode == 'mean_upto_t' and j == 0: continue
        Xa = X[:, j].reshape(len(X), -1) if mode == 'at_t' else (X[:, :j + 1] * OK[:, :j + 1, None, None]).sum(1).reshape(len(X), -1) / np.maximum(OK[:, :j + 1].sum(1), 1)[:, None]
        valid = OK[:, j] if mode == 'at_t' else OK[:, :j + 1].any(1); pred = np.full(len(X), -1)
        for k in range(5):
            tr = np.where(valid & (fo != k))[0]; te = np.where(valid & (fo == k))[0]
            if len(te): pred[te] = fit_pred(Xa, tr, te)
        v = np.where(valid)[0]; acc = float((pred[v] == y[v]).mean())
        per = {T[c]: dict(n=int((y[v] == c).sum()), acc=round(float((pred[v][y[v] == c] == c).mean()), 3)) for c in range(len(T)) if (y[v] == c).any()}
        conf = collections.Counter((T[a], T[b]) for a, b in zip(y[v], pred[v]) if a != b).most_common(12)
        res['%s@%d' % (mode, t)] = dict(n=len(v), acc=round(acc, 4), balanced_acc=round(float(np.mean([p['acc'] for p in per.values()])), 4), per_task=per, top_confusions=[[a, b, c] for (a, b), c in conf])
        print(mode, t, len(v), round(acc, 4), res['%s@%d' % (mode, t)]['balanced_acc'], conf[:5], flush=True)
json.dump(dict(grouping='v2: ctx if >=5 contexts, (ctx, layout) if >=2 layouts, else episode', within_context_only=WITHIN, tasks=T, n_episodes=len(X), counts={t: int((task == t).sum()) for t in T}, results=res), open('RESULT-taskid-dev-v2.json', 'w'), indent=1)
