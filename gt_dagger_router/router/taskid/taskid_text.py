"""Router task ID = instruction text + image (user 10-31 02:2xZ: the prompt is available at deploy).
Text encoder: no standalone local text encoder in the caches (SAM3's text tower lives inside svc/sam3-locate/ckpt/sam3.pt, not extracted)
-> learned token embedding: lowercase word tokens (vocab from the TRAIN rows only, unknown -> <unk>), EmbeddingBag mean (64-d).
Image: dino_features [CLS, mean patch] x 3 views of the first frame (4608), standardised on train.  Fusion: concat -> MLP 512 -> 42.
Variants: text_only, image_only, text+image.  Data / held-out:
  (A) 4688 archived Astra episodes (taskid/feat t = 0; instruction from <episode>/reset.json), eval_taskid v2 grouping, 5-fold CV;
  (B) 900 official demos (official/feat, first cached frame + 'instruction'), GT.md split 80 / 10 / 10 (val, test reported).
Per task accuracy and the language-variant pairs (tasks whose names differ only by '_by_language' / '_language').  -> RESULT-taskid-text.json"""
import glob, json, random, re, collections
from pathlib import Path
import numpy as np, torch, torch.nn as nn
H = Path(__file__).resolve().parent; RR = H.parent; NAT = Path('/lizekai/RSIws/artifacts/native'); dev = torch.device('cuda:0')
idx = {e['episode']: e for e in json.load(open(H / 'episode_index.json'))}; TASKS = sorted({e['task'] for e in idx.values()}); TI = {t: i for i, t in enumerate(TASKS)}
def toks(s): return re.findall(r"[a-z]+", s.lower())
# (A) archive
X, EP = [], []
for f in sorted(glob.glob(str(H / 'feat/shard-*.npz'))):
    z = np.load(f); ok = z['ok'][:, 0]; X.append(z['X'][ok, 0].astype(np.float32).reshape(ok.sum(), -1)); EP += [e for e, o in zip(z['episodes'], ok) if o]
XA = np.concatenate(X); TA = [json.load(open(NAT / e / 'reset.json')).get('instruction', '') for e in EP]; YA = np.array([TI[idx[e]['task']] for e in EP])
task = np.array([idx[e]['task'] for e in EP]); ctx = np.array(['-'.join(e.split('-')[:3]) for e in EP]); lay = np.array([idx[e]['layout'] for e in EP])
nctx = collections.Counter(t for t, c in {(t, c) for t, c in zip(task, ctx)}); nlay = {t: len(set(lay[task == t])) for t in set(task)}
grp = np.array([c if nctx[t] >= 5 else ('%s|L%d' % (c, l) if nlay[t] >= 2 else e) for t, c, l, e in zip(task, ctx, lay, EP)])
cs = sorted(set(grp)); random.Random(20261030).shuffle(cs); fold = {c: i % 5 for i, c in enumerate(cs)}; FO = np.array([fold[c] for c in grp])
# (B) official demos
XB, TB, YB, SB = [], [], [], []
for f in sorted(glob.glob(str(RR / 'official/feat/*/*.npz'))):
    z = np.load(f, allow_pickle=True)
    if not len(z['X']) or str(z['task']) not in TI: continue
    e = int(re.search(r'episode_(\d+)', f).group(1)); XB.append(z['X'][0].astype(np.float32).reshape(-1)); TB.append(str(z['instruction'])); YB.append(TI[str(z['task'])]); SB.append(0 if e < 80 else 1 if e < 90 else 2)
XB, YB, SB = np.array(XB), np.array(YB), np.array(SB)
class M(nn.Module):
    def __init__(s, nv, use_t, use_i):
        super().__init__(); s.use_t, s.use_i = use_t, use_i; s.emb = nn.EmbeddingBag(nv, 64, mode='mean')
        s.mlp = nn.Sequential(nn.Linear(64 * use_t + 4608 * use_i, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, len(TASKS)))
    def forward(s, xi, ti, off):
        h = ([s.emb(ti, off)] if s.use_t else []) + ([xi] if s.use_i else []); return s.mlp(torch.cat(h, 1))
def fit_pred(Xi, T, y, tr, te, use_t, use_i, T_te=None):
    """T_te: optional replacement texts for the test rows (paraphrase robustness; vocab still from the train rows)."""
    voc = {'<unk>': 0}
    for i in tr:
        for w in toks(T[i]): voc.setdefault(w, len(voc))
    ids = [[voc.get(w, 0) for w in toks(t)] or [0] for t in T]
    if T_te is not None:
        for r, t in zip(te, T_te): ids[r] = [voc.get(w, 0) for w in toks(t)] or [0]
    mu, sd = Xi[tr].mean(0), Xi[tr].std(0) + 1e-6; A = torch.tensor((Xi - mu) / sd, dtype=torch.float32, device=dev)
    def tb(rows):
        flat = [w for r in rows for w in ids[r]]; off = np.cumsum([0] + [len(ids[r]) for r in rows[:-1]])
        return torch.tensor(flat, device=dev), torch.tensor(off, device=dev)
    cnt = np.bincount(y[tr], minlength=len(TASKS)); w = torch.tensor(1.0 / np.maximum(cnt, 1), dtype=torch.float32, device=dev); w = w / w.mean()
    torch.manual_seed(0); m = M(len(voc), use_t, use_i).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05); Y = torch.tensor(y, device=dev)
    for ep in range(40):
        perm = np.random.default_rng(ep).permutation(tr)
        for i0 in range(0, len(perm), 256):
            b = perm[i0:i0 + 256]; ti, off = tb(list(b)); opt.zero_grad(); nn.functional.cross_entropy(m(A[b], ti, off), Y[b], weight=w).backward(); opt.step()
    m.eval(); ti, off = tb(list(te))
    with torch.no_grad(): return m(A[te], ti, off).argmax(1).cpu().numpy()
PAIRS = [(a, b) for a in TASKS for b in TASKS if a != b and re.sub(r'_?(by_)?language', '', b) == a]
def report(y, p):
    per = {TASKS[c]: dict(n=int((y == c).sum()), acc=round(float((p[y == c] == c).mean()), 3)) for c in range(len(TASKS)) if (y == c).any()}
    pairs = {'%s|%s' % (a, b): dict(confusions=int(sum(((y == TI[a]) & (p == TI[b])) | ((y == TI[b]) & (p == TI[a])))), n=int(sum((y == TI[a]) | (y == TI[b])))) for a, b in PAIRS}
    return dict(n=len(y), acc=round(float((y == p).mean()), 4), balanced_acc=round(float(np.mean([v['acc'] for v in per.values()])), 4), language_pairs=pairs, per_task=per,
                top_confusions=collections.Counter((TASKS[a], TASKS[b]) for a, b in zip(y, p) if a != b).most_common(8))
if __name__ == '__main__':
  res = dict(text_encoder='learned token EmbeddingBag (no standalone local text encoder; SAM3 text tower not extracted)', pairs=PAIRS,
             n_unique_instructions_archive={t: len({s for s, tt in zip(TA, task) if tt == t}) for t in TASKS}, archive={}, demos={})
  for name, ut, ui in (('text_only', 1, 0), ('image_only', 0, 1), ('text+image', 1, 1)):
      p = np.full(len(YA), -1)
      for k in range(5):
          tr, te = np.where(FO != k)[0], np.where(FO == k)[0]; p[te] = fit_pred(XA, TA, YA, tr, te, ut, ui)
      res['archive'][name] = report(YA, p); print('archive', name, res['archive'][name]['acc'], res['archive'][name]['top_confusions'][:3], flush=True)
      tr = np.where(SB == 0)[0]; res['demos'][name] = {}
      for sn, sv in (('val', 1), ('test', 2)):
          te = np.where(SB == sv)[0]; res['demos'][name][sn] = report(YB[te], fit_pred(XB, TB, YB, tr, te, ut, ui))
      print('demos', name, {s: res['demos'][name][s]['acc'] for s in ('val', 'test')}, flush=True)
  json.dump(res, open(H / 'RESULT-taskid-text.json', 'w'), indent=1, default=str)
