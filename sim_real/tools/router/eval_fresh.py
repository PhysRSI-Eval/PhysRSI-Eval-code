"""Task-ID on fresh published seeds + leave-layouts-out (dev). Train: archived episodes' t=0 features (all 42 tasks). Tests:
 (1) fresh seeds: published layout ids never present in the archive (only build_tower and classify_objects_by_language have them),
     t=0 frames captured by fresh_t0.py (native reset, zero quota);
 (2) leave-layouts-out: for the 8 multi-layout tasks, layouts 3 / 8 / 13 held out of training (all other tasks fully in training).
 MLP as eval_taskid.py.  -> RESULT-taskid-fresh.json"""
import glob, json, sys, collections
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import router_rgb as RR   # noqa: E402
from transformers import AutoModel
X, OK, EP = [], [], []
for f in sorted(glob.glob('feat/shard-*.npz')):
    z = np.load(f); X.append(z['X'][:, 0].astype(np.float32)); OK.append(z['ok'][:, 0]); EP += list(z['episodes'])
X = np.concatenate(X).reshape(-1, 3 * 1536); OK = np.concatenate(OK); idx = {e['episode']: e for e in json.load(open('episode_index.json'))}
task = np.array([idx[e]['task'] for e in EP]); lay = np.array([idx[e]['layout'] for e in EP]); T = sorted(set(task)); ti = {t: i for i, t in enumerate(T)}; y = np.array([ti[t] for t in task])
dev = torch.device('cuda:0')
def fit(tr):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; A = torch.tensor(((X - mu) / sd).astype(np.float32), device=dev); Y = torch.tensor(y, device=dev)
    cnt = np.bincount(y[tr], minlength=len(T)); w = torch.tensor(1.0 / np.maximum(cnt, 1), dtype=torch.float32, device=dev); w = w / w.mean()
    torch.manual_seed(0); m = nn.Sequential(nn.Linear(X.shape[1], 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, len(T))).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
    trt = torch.tensor(tr, device=dev)
    for ep in range(40):
        perm = trt[torch.randperm(len(tr), device=dev)]
        for i0 in range(0, len(tr), 256):
            b = perm[i0:i0 + 256]; opt.zero_grad(); nn.functional.cross_entropy(m(A[b]), Y[b], weight=w).backward(); opt.step()
    m.eval(); return m, mu, sd
def pred(m, mu, sd, Z):
    with torch.no_grad(): return m(torch.tensor(((Z - mu) / sd).astype(np.float32), device=dev)).argmax(1).cpu().numpy()
R = {}
# (2) leave-layouts-out
multi = [t for t in T if len(set(lay[task == t])) > 1]; hold = np.isin(task, multi) & np.isin(lay, [3, 8, 13])
m, mu, sd = fit(np.where(OK & ~hold)[0]); te = np.where(OK & hold)[0]; p = pred(m, mu, sd, X[te])
R['leave_layouts_3_8_13'] = dict(n=len(te), acc=round(float((p == y[te]).mean()), 4), per_task={t: dict(n=int((task[te] == t).sum()), acc=round(float((p[task[te] == t] == ti[t]).mean()), 3)) for t in multi if (task[te] == t).any()},
                                 confusions=[[T[a], T[b], c] for (a, b), c in collections.Counter((a, b) for a, b in zip(y[te], p) if a != b).most_common(10)])
# (1) fresh seeds
m, mu, sd = fit(np.where(OK)[0]); vit = AutoModel.from_pretrained(RR.DINO).to(dev).eval(); nreg = vit.config.num_register_tokens; F, FT, FL = [], [], []
for mf in sorted(glob.glob('fresh/*/L*/meta.json')):
    md = json.load(open(mf))
    if not md.get('ok') or md['layout'] < 1 and md['task'] == 'sweep_blocks': continue
    dd = Path(mf).parent; ims = {c: np.asarray(Image.open(dd / ('%s.png' % c)).convert('RGB')) for c in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')}
    F.append(RR.dino_features(vit, nreg, ims, dev).reshape(-1)); FT.append(md['task']); FL.append(md['layout'])
if F:
    p = pred(m, mu, sd, np.asarray(F)); R['fresh_seeds'] = [dict(task=t, layout=l, pred=T[q], ok=bool(T[q] == t)) for t, l, q in zip(FT, FL, p)]
    R['fresh_seeds_acc'] = {t: round(float(np.mean([r['ok'] for r in R['fresh_seeds'] if r['task'] == t])), 3) for t in sorted(set(FT))}
json.dump(R, open('RESULT-taskid-fresh.json', 'w'), indent=1); print(json.dumps({k: (v if k != 'fresh_seeds' else len(v)) for k, v in R.items()}, indent=0)[:3000])
