"""ROUTER-RGB-v1 level (b) DEV (router-rgb/PREREG-ROUTER-RGB-v1.md + Amendment 1): = gate/train_router_v0.py with NO task id and NO tool facts
(RGB + proprio/progress only). Copy of the v0 script:
ROUTER-v0 (PREREG gate/PREREG-router-v0.md + AMENDMENT-A): grouped 5-fold CV by context, judged once.
  /opt/PhysRSI/ENV/robodojo/bin/python gate/train_router_v0.py --gpu G [--pw PATH.npz] -> gate/RESULT-router-v0[b].json"""
import argparse, json, random
import numpy as np
import torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
V = '/lizekai/RSIws/experiments/macro-tools-20261027/gate'
ap = argparse.ArgumentParser(); ap.add_argument('--gpu', type=int, default=0); ap.add_argument('--pw', default=None); ap.add_argument('--coin-kind', default='coin_cn2-comp'); ap.add_argument('--tasks', default='plug_in_charger'); a = ap.parse_args()
D = json.load(open(V + '/ROUTER-DATA-v0.json'))['rows']; X_img = np.load(V + '/ROUTER-DATA-v0.npz')['dino']
PW = np.load(a.pw)['pw'] if a.pw else None
TASKS_ALL = ['plug_in_charger', 'deposit_coin']; TASKS = a.tasks.split(','); PRIM = {'rg3f'} | ({a.coin_kind, 'coin_cn-comp'} if 'deposit_coin' in TASKS else set())   # Amendment C


def tab(r):
    f = r['features']
    return [f['budget_used'], f['grip_l'], f['grip_r'], f['cmd_l'], f['cmd_r'], f['holding_l'], f['holding_r'], f['closes_l'], f['closes_r'], f['travel_rad'] / 10.0] + [v / 3.0 for v in f['joints']]


def facts(r): return [0.0]   # level (b): no task id, no tool facts (constant token)


class Router(nn.Module):
    def __init__(self, d_tab, d_fact, d_pw=0, d=128):
        super().__init__()
        self.img = nn.Linear(1536, d); self.view = nn.Parameter(torch.zeros(3, d)); self.tab = nn.Linear(d_tab, d); self.fact = nn.Linear(d_fact, d)
        self.pw = nn.Linear(d_pw, d) if d_pw else None; self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 256, dropout=0.1, batch_first=True), 2); self.out = nn.Linear(d, 1)

    def forward(self, xi, xt, xf, xp=None):
        toks = [self.img(xi) + self.view, self.tab(xt)[:, None], self.fact(xf)[:, None]] + ([self.pw(xp)[:, None]] if self.pw is not None else [])
        h = torch.cat([self.cls.expand(xi.shape[0], -1, -1)] + toks, 1); return self.out(self.enc(h)[:, 0, :])[:, 0]


def fit_predict(tr, te, rows, dev):
    def T(idx):
        out = [torch.tensor(X_img[idx], dtype=torch.float32, device=dev), torch.tensor([tab(rows[i]) for i in idx], dtype=torch.float32, device=dev),
               torch.tensor([facts(rows[i]) for i in idx], dtype=torch.float32, device=dev)]
        if PW is not None: out.append(torch.tensor(PW[idx], dtype=torch.float32, device=dev))
        return out
    y = torch.tensor([float(rows[i]['y']) for i in tr], device=dev); pos = max(float(y.sum()), 1.0); w = (len(tr) - pos) / pos
    ps = []
    for seed in range(3):
        torch.manual_seed(seed); m = Router(len(tab(rows[0])), len(facts(rows[0])), PW.shape[1] if PW is not None else 0).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05); Xtr = T(tr); lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w, device=dev))
        for ep in range(200):
            m.train(); opt.zero_grad(); l = lossf(m(*Xtr), y); l.backward(); opt.step()
        m.eval()
        with torch.no_grad(): ps.append(torch.sigmoid(m(*T(te))).cpu().numpy())
    return np.mean(ps, 0)


def evaluate(idx, rows, p, name):
    R = [rows[i] for i in idx]; ab = np.array([r['abstained'] for r in R]); go = (p > 0.65) & ~ab
    sa = np.array([r['astra_success'] for r in R], float); si = np.where(ab, sa, np.array([r['int_success'] for r in R], float)); sg = np.where(go, si, sa)
    y = np.array([r['y'] for r in R]); harm = int(sum(g and r['astra_success'] and not r['int_success'] for g, r in zip(go, R)))
    per = {t: dict(n=int(sum(r['task'] == t for r in R)), positives=int(sum(y_ for y_, r in zip(y, R) if r['task'] == t)),
                   routed=round(float(np.mean([s for s, r in zip(sg, R) if r['task'] == t])), 3) if any(r['task'] == t for r in R) else None,
                   astra=round(float(np.mean([s for s, r in zip(sa, R) if r['task'] == t])), 3) if any(r['task'] == t for r in R) else None,
                   executor=round(float(np.mean([s for s, r in zip(si, R) if r['task'] == t])), 3) if any(r['task'] == t for r in R) else None,
                   takeovers=int(sum(g for g, r in zip(go, R) if r['task'] == t))) for t in TASKS}
    return dict(model=name, n=len(R), positives=int(y.sum()), success=dict(routed=round(float(sg.mean()), 3), always_astra=round(float(sa.mean()), 3),
                always_executor=round(float(si.mean()), 3), oracle=round(float(np.maximum(sa, si).mean()), 3)), takeovers=int(go.sum()), harmful=harm, per_task=per)


if __name__ == '__main__':
    dev = torch.device('cuda:%d' % a.gpu if a.gpu >= 0 else 'cpu')
    for r in D: r['y'] = int(bool(r['int_success']) and not r['astra_success'] and not r['abstained'])
    seen = set(); idx = []   # Amendment B: one row per (executor, unique state); the coin key includes the archived action-prefix hash
    for i in sorted([i for i, r in enumerate(D) if r['kind'] in PRIM], key=lambda i: (D[i]['kind'], D[i]['episode'])):
        k = (D[i]['kind'], D[i]['ctx'], D[i]['t'], D[i].get('prefix_hash') if D[i]['task'] == 'deposit_coin' else None)
        if k in seen: continue
        seen.add(k); idx.append(i)
    ctx = sorted({D[i]['ctx'] for i in idx}); random.Random(20261029).shuffle(ctx); fold = {c: k % 5 for k, c in enumerate(ctx)}
    p_r = np.zeros(len(idx)); p_l = np.zeros(len(idx)); pos = {i: j for j, i in enumerate(idx)}
    for k in range(5):
        tr = [i for i in idx if fold[D[i]['ctx']] != k]; te = [i for i in idx if fold[D[i]['ctx']] == k]
        if not te: continue
        pr = fit_predict(tr, te, D, dev)
        ytr = np.array([D[i]['y'] for i in tr]); Xl = lambda ii: np.array([tab(D[i]) + facts(D[i]) for i in ii])
        pl = LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000).fit(Xl(tr), ytr).predict_proba(Xl(te))[:, 1] if 0 < ytr.sum() < len(ytr) else np.zeros(len(te))
        for i, a_, b_ in zip(te, pr, pl): p_r[pos[i]] = a_; p_l[pos[i]] = b_
    res = dict(prereg=['gate/PREREG-router-v0.md', 'gate/PREREG-router-v0-AMENDMENT-A.md'], pw=a.pw, coin_kind=a.coin_kind,
               router=evaluate(idx, D, p_r, 'router-v0' + ('b' if PW is not None else '')), logistic=evaluate(idx, D, p_l, 'gate-v2 logistic baseline'))
    R_, L_ = res['router'], res['logistic']; few = min(R_['per_task'][t]['positives'] for t in TASKS)
    ok = (R_['success']['routed'] > R_['success']['always_astra'] and R_['success']['routed'] > R_['success']['always_executor'] and R_['harmful'] <= 0.10 * max(R_['takeovers'], 1)
          and R_['success']['routed'] > L_['success']['routed'])
    res['verdict'] = 'INCONCLUSIVE' if few < 10 else ('PASS' if ok else 'FAIL')
    json.dump(res, open('/lizekai/RSIws/experiments/macro-tools-20261027/router-rgb/DEV-b-%s.json' % __import__('time').strftime('%Y%m%dT%H%M%S', __import__('time').gmtime()), 'w'), indent=1); print(json.dumps(res, indent=1))
