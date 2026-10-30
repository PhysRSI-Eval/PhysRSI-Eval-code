"""Freeze ROUTER-v0 (gate/PREREG-router-v0.md + Amendments A-C; PASS single-task, plug_in_charger + frozen rg3f, gate/CONCLUSION-router-v0.md)
as the deployable L3 successor of gate v2. Same rows (Amendment B dedup, plug only), same model class, same features and training recipe as
gate/train_router_v0.py fit_predict; fitted once on ALL 67 deduplicated rows (the CV verdict is the evidence; this is only the deployment fit),
3 seeds kept (their mean probability is the decision, as in the CV). CPU.
  /opt/PhysRSI/ENV/robodojo/bin/python tools/fit_router_v0_final.py -> rsi_v2/policies/router_v0/{model_s0..2.pt, meta.json}"""
import hashlib, json, sys
import numpy as np, torch, torch.nn as nn
G = '/lizekai/RSIws/experiments/macro-tools-20261027/gate'; OUT = '/lizekai/RSIws/experiments/harness-v2-20261028/rsi_v2/policies/router_v0'
sys.argv = [sys.argv[0], '--gpu', '-1', '--tasks', 'plug_in_charger']
src = open(G + '/train_router_v0.py').read().split("if __name__ == '__main__':")[0]; ns = {}; exec(compile(src, G + '/train_router_v0.py', 'exec'), ns)
D, X_img, tab, facts, Router = ns['D'], ns['X_img'], ns['tab'], ns['facts'], ns['Router']
for r in D: r['y'] = int(bool(r['int_success']) and not r['astra_success'] and not r['abstained'])
seen = set(); idx = []
for i in sorted([i for i, r in enumerate(D) if r['kind'] in {'rg3f'}], key=lambda i: (D[i]['kind'], D[i]['episode'])):
    k = (D[i]['kind'], D[i]['ctx'], D[i]['t'], None)
    if k in seen: continue
    seen.add(k); idx.append(i)
dev = torch.device('cpu'); xi = torch.tensor(X_img[idx], dtype=torch.float32); xt = torch.tensor([tab(D[i]) for i in idx], dtype=torch.float32); xf = torch.tensor([facts(D[i]) for i in idx], dtype=torch.float32)
y = torch.tensor([float(D[i]['y']) for i in idx]); pos = max(float(y.sum()), 1.0); w = (len(idx) - pos) / pos; P = []
for seed in range(3):
    torch.manual_seed(seed); m = Router(xt.shape[1], xf.shape[1], 0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05); lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w))
    for ep in range(200):
        m.train(); opt.zero_grad(); l = lossf(m(xi, xt, xf), y); l.backward(); opt.step()
    m.eval(); torch.save(m.state_dict(), OUT + '/model_s%d.pt' % seed)
    with torch.no_grad(): P.append(torch.sigmoid(m(xi, xt, xf)).numpy())
p = np.mean(P, 0); res = json.load(open(G + '/RESULT-router-v0.json'))
meta = dict(policy='router_v0', verdict=res.get('verdict'), prereg=['gate/PREREG-router-v0.md', 'AMENDMENT-A', 'AMENDMENT-B', 'AMENDMENT-C'], conclusion='gate/CONCLUSION-router-v0.md',
            cv=dict(router=res['router']['success'], takeovers=res['router']['takeovers'], harmful=res['router']['harmful'], logistic=res['logistic']['success']['routed']),
            n_train=len(idx), positives=int(y.sum()), scope=['plug_in_charger'], executor='insert_charger', executor_kind='rg3f', dedup_refit_required=False,
            features=dict(images='DINOv3-B/16 (svc/dinov3-weights/dinov3-vitb16, frozen) CLS + mean patch token per view, raw 640x480, ImageNet mean/std; views [cam_high, cam_left_wrist, cam_right_wrist]',
                          tab='gate-v1 progress/proprio 10 (travel/10) + 12 arm joints /3', facts="[task=plug_in_charger, task=deposit_coin, chain_abstained, lift_inhand_abstain]"),
            deploy_note=("'chain_abstained' in training was the rg3f chain's own abstain, which includes no_outlets (known only after the observer moves); at a "
                         "decision point it is not available -> deploy uses chain_abstained := lift_inhand_abstain (the 0-step in-hand v4 pre-check). Evaluation "
                         "treated abstaining rows as no takeover, so this only affects the input distribution; log/offer modes only."),
            threshold=0.65, in_sample=dict(above_0_65=int((p > 0.65).sum()), positives_caught=int(((p > 0.65) & (y.numpy() > 0)).sum())),
            data_sha256={f: hashlib.sha256(open(G + '/' + f, 'rb').read()).hexdigest() for f in ('ROUTER-DATA-v0.json', 'ROUTER-DATA-v0.npz', 'train_router_v0.py')})
json.dump(meta, open(OUT + '/meta.json', 'w'), indent=1); print(json.dumps({k: meta[k] for k in ('verdict', 'cv', 'n_train', 'positives', 'in_sample')}))
