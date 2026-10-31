"""Final task-ID head for deploy (M-R2): same MLP / recipe as eval_taskid.py (4608 -> 512 -> 42, 40 epochs, class-balanced), fit on ALL
archived episodes at t = 0 (dev acc 1.00 at t = 0, RESULT-taskid-dev-v2.json; fresh seeds RESULT-taskid-fresh.json).
Input = router_rgb.dino_features(...) of the first frame, flattened (3 x 1536).  -> models/taskid.pt (state_dict, mu, sd, tasks)"""
import glob, json
import numpy as np, torch, torch.nn as nn
X, OK, EP = [], [], []
for f in sorted(glob.glob('taskid/feat/shard-*.npz')):
    z = np.load(f); X.append(z['X'][:, 0].astype(np.float32)); OK.append(z['ok'][:, 0]); EP += list(z['episodes'])
X = np.concatenate(X).reshape(-1, 3 * 1536); OK = np.concatenate(OK); idx = {e['episode']: e for e in json.load(open('taskid/episode_index.json'))}
T = sorted({e['task'] for e in idx.values()}); ti = {t: i for i, t in enumerate(T)}; y = np.array([ti[idx[e]['task']] for e in EP])
X, y = X[OK], y[OK]; dev = torch.device('cuda:0')
mu, sd = X.mean(0), X.std(0) + 1e-6; A = torch.tensor((X - mu) / sd, device=dev); Y = torch.tensor(y, device=dev)
cnt = np.bincount(y, minlength=len(T)); w = torch.tensor(1.0 / np.maximum(cnt, 1), dtype=torch.float32, device=dev); w = w / w.mean()
torch.manual_seed(0); m = nn.Sequential(nn.Linear(X.shape[1], 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, len(T))).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
for ep in range(40):
    perm = torch.randperm(len(X), device=dev)
    for i0 in range(0, len(X), 256):
        b = perm[i0:i0 + 256]; opt.zero_grad(); l = nn.functional.cross_entropy(m(A[b]), Y[b], weight=w); l.backward(); opt.step()
m.eval()
with torch.no_grad(): tr_acc = float((m(A).argmax(1) == Y).float().mean())
torch.save(dict(state_dict=m.cpu().state_dict(), mu=mu, sd=sd, tasks=T), 'models/taskid.pt'); print('n', len(X), 'train acc', round(tr_acc, 4))
