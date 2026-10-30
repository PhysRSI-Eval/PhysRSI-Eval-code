"""Fit the deploy takeover head on all 67 deduplicated plug rows (no CV; for the hv2g slot). -> models/takeover.pt"""
import json, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, '/lizekai/RSIws/experiments/macro-tools-20261027/router-rgb'); import router_rgb as RR
G = '/lizekai/RSIws/experiments/macro-tools-20261027/gate'; D = json.load(open(G + '/ROUTER-DATA-v0.json'))['rows']; X = np.load(G + '/ROUTER-DATA-v0.npz')['dino']
seen = set(); idx = []
for i in sorted([i for i, r in enumerate(D) if r['kind'] == 'rg3f'], key=lambda i: (D[i]['kind'], D[i]['episode'])):
    k = (D[i]['ctx'], D[i]['t'])
    if k not in seen: seen.add(k); idx.append(i)
y = torch.tensor([float(D[i]['int_success'] and not D[i]['astra_success'] and not D[i]['abstained']) for i in idx]); xi = torch.tensor(X[idx]); xt = torch.tensor([RR.tab(D[i]['features']) for i in idx], dtype=torch.float32); xf = torch.zeros(len(idx), 1)
w = (len(idx) - y.sum()) / max(y.sum(), 1); sds = []
for seed in range(3):
    torch.manual_seed(seed); m = RR.Router(xt.shape[1], 1); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05); lf = nn.BCEWithLogitsLoss(pos_weight=w)
    for ep in range(200): m.train(); opt.zero_grad(); l = lf(m(xi, xt, xf), y); l.backward(); opt.step()
    sds.append(m.state_dict())
torch.save(dict(state_dicts=sds, d_tab=xt.shape[1], rows=len(idx), positives=int(y.sum()), recipe='ROUTER-v0 without facts, 200 ep full batch, 3 seeds'), '/lizekai/RSIws/experiments/macro-tools-20261027/router-rgb/models/takeover.pt')
print('saved', len(idx), int(y.sum()))
