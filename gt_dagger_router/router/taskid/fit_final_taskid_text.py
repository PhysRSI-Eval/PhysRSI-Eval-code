"""Deploy task-ID head = instruction text + first-frame image (paraphrase test: text-only breaks on synonym / content rewordings,
text+image 1.00 on all kinds, RESULT-taskid-paraphrase.json). Same recipe as taskid_text.fit_pred, fit on ALL archive episodes + all
official demo rows.  -> ../models/taskid_text.pt (state_dict, vocab, mu, sd, tasks)"""
import numpy as np, torch, torch.nn as nn
import taskid_text as TT
X = np.concatenate([TT.XA, TT.XB]); T = list(TT.TA) + list(TT.TB); y = np.concatenate([TT.YA, TT.YB]); dev = TT.dev
voc = {'<unk>': 0}
for t in T:
    for w in TT.toks(t): voc.setdefault(w, len(voc))
ids = [[voc[w] for w in TT.toks(t)] or [0] for t in T]; mu, sd = X.mean(0), X.std(0) + 1e-6; A = torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev); Y = torch.tensor(y, device=dev)
def tb(rows):
    flat = [w for r in rows for w in ids[r]]; off = np.cumsum([0] + [len(ids[r]) for r in rows[:-1]]); return torch.tensor(flat, device=dev), torch.tensor(off, device=dev)
cnt = np.bincount(y, minlength=len(TT.TASKS)); w = torch.tensor(1.0 / np.maximum(cnt, 1), dtype=torch.float32, device=dev); w = w / w.mean()
torch.manual_seed(0); m = TT.M(len(voc), 1, 1).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05); allr = np.arange(len(y))
for ep in range(40):
    perm = np.random.default_rng(ep).permutation(allr)
    for i0 in range(0, len(perm), 256):
        b = perm[i0:i0 + 256]; ti, off = tb(list(b)); opt.zero_grad(); nn.functional.cross_entropy(m(A[b], ti, off), Y[b], weight=w).backward(); opt.step()
m.eval(); ti, off = tb(list(allr))
with torch.no_grad(): acc = float((m(A, ti, off).argmax(1) == Y).float().mean())
torch.save(dict(state_dict=m.cpu().state_dict(), vocab=voc, mu=torch.tensor(mu), sd=torch.tensor(sd), tasks=TT.TASKS), TT.RR / 'models/taskid_text.pt'); print('n', len(y), 'vocab', len(voc), 'train acc', round(acc, 4))
