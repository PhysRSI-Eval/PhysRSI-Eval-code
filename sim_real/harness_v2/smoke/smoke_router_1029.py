"""ROUTER-v0 deploy wiring smoke (no simulator, no model calls): RouterV0.decide() from the ARCHIVED per-step history + archived frames of the 67
training rows reproduces the frozen models applied to the dataset features (images recomputed on CPU here vs GPU in the dataset);
with deploy facts (chain_abstained := lift_inhand_abstain) the decisions are also reported. Harness: router in 'auto' -> 'offer'.
  -> out/SMOKE-router-1029.json"""
import json, sys
from pathlib import Path
import numpy as np, torch
H = Path('/lizekai/RSIws/experiments/harness-v2-20261028'); sys.path.insert(0, str(H))
from rsi_v2 import policy as P, harness as HN
G = '/lizekai/RSIws/experiments/macro-tools-20261027/gate'; sys.path.insert(0, G); import build_dataset_v2 as BD
D = json.load(open(G + '/ROUTER-DATA-v0.json'))['rows']; X = np.load(G + '/ROUTER-DATA-v0.npz')['dino']
src = open(G + '/train_router_v0.py').read().split("if __name__ == '__main__':")[0]
sys.argv = [sys.argv[0], '--gpu', '-1']; ns = {}; exec(compile(src, G + '/train_router_v0.py', 'exec'), ns); tab, facts = ns['tab'], ns['facts']
Rt = P.RouterV0(); seen = set(); rows = []
for i in sorted([i for i, r in enumerate(D) if r['kind'] == 'rg3f'], key=lambda i: D[i]['episode']):
    k = (D[i]['ctx'], D[i]['t'])
    if k in seen: continue
    seen.add(k); rows.append(i)
out = []
for i in rows:
    r = D[i]; inner = [p for p in (BD.NAT / r['episode']).iterdir() if p.is_dir() and (p / 'action_000000.json').exists()][0]
    acts = [json.load(open(inner / ('action_%06d.json' % k))) for k in range(r['t'] + 1) if (inner / ('action_%06d.json' % k)).exists()]
    S = [a['obs']['states'] for a in acts if (a.get('obs') or {}).get('states')]; A = [a['executed_action'] for a in acts]
    ims = dict(zip(P.RouterV0.VIEWS, r['frames']))
    obs = dict(task='plug_in_charger', step=r['t'], step_limit=400, states_hist=S, cmd_hist=A, executor='insert_charger', images=ims,
               tool_facts=dict(lift_inhand_abstain=r['facts']['lift_inhand_abstain'], chain_abstained=r['facts']['abstained']))
    d_exact = Rt.decide(obs)
    with torch.no_grad():
        xi = torch.tensor(X[i][None]); xt = torch.tensor([tab(r)], dtype=torch.float32); xf = torch.tensor([facts(r)], dtype=torch.float32)
        p_ref = float(np.mean([torch.sigmoid(m(xi, xt, xf)).item() for m in Rt.models]))
    obs['tool_facts'] = dict(lift_inhand_abstain=r['facts']['lift_inhand_abstain']); d_dep = Rt.decide(obs)
    out.append(dict(ep=r['episode'], t=r['t'], p_exact=d_exact['p'], p_ref=round(p_ref, 4), p_deploy=d_dep['p'], dec_ref=bool(p_ref > 0.65), dec_exact=d_exact['decision'], dec_deploy=d_dep['decision'], y=int(bool(r['int_success']) and not r['astra_success'] and not r['abstained'])))
dif = [abs(o['p_exact'] - o['p_ref']) for o in out]
agree = sum((o['dec_exact'] == 'intervene') == (o['dec_ref'] and not D[rows[k]]['facts']['lift_inhand_abstain']) for k, o in enumerate(out))
h = HN.Harness(dict(task='plug_in_charger', l3_policy=dict(name='router_v0', mode='auto', auto_approved=True, executor='insert_charger')))
R = dict(n=len(out), max_abs_p_diff=round(max(dif), 4), median_abs_p_diff=round(float(np.median(dif)), 5), decision_agree=agree,
         deploy_intervene=sum(o['dec_deploy'] == 'intervene' for o in out), deploy_intervene_on_y1=sum(o['dec_deploy'] == 'intervene' and o['y'] for o in out), positives=sum(o['y'] for o in out),
         harness_mode_for_auto=h.pol_mode, rows=out)
json.dump(R, open(H / 'out/SMOKE-router-1029.json', 'w'), indent=1); print({k: v for k, v in R.items() if k != 'rows'})
