"""hv2f router wiring (no model calls, no simulator): combined_tools.after_act with l3_policy {name: router_v0, mode: log|offer|auto} on archived
plug rg3f handover states (per-step files + archived frames at the step) -> l3-gate.jsonl p equals RouterV0 of the harness RC on the same inputs;
offer gives one availability line only when the router says intervene; 'auto' is treated as offer; no l3_policy -> nothing. -> RESULT-hv2f-router.json"""
import json, sys, tempfile
from pathlib import Path
from rsi import combined_tools as CT, hv2e_ext as X, harness_l3 as HL, hv2f_l3 as F
sys.path.insert(0, '/lizekai/RSIws/experiments/harness-v2-20261028'); from rsi_v2 import policy as P
G = '/lizekai/RSIws/experiments/macro-tools-20261027/gate'; D = json.load(open(G + '/ROUTER-DATA-v0.json'))['rows']
rows = [r for r in D if r['kind'] == 'rg3f'][:6]; RC = P.RouterV0(); R = dict(rows=[])
NAT = Path('/lizekai/RSIws/artifacts/native')


class FakeAd:
    def __init__(self, out_dir, ev, T, st, imgs):
        self.packet = dict(step_id=T, current_state=st, current_eef={'left': dict(position=[0.3, 0.2, 0.9], quaternion_wxyz=[1, 0, 0, 0]), 'right': dict(position=[0.3, -0.2, 0.9], quaternion_wxyz=[1, 0, 0, 0])},
                           images=[dict(camera=c, path=str(p)) for c, p in imgs.items()], request_id='r0')
        self.evidence = ev; self.output = out_dir; self.terminal_reason = None; self.ticket = dict(task_id='plug_in_charger', episode_id='e'); self.contract = dict(native_horizon=400)
    def _invoke(self, *a): raise RuntimeError('no motion in this test')
    def _content(self, pk): return []


for r in rows:
    T = r['t']; inner = [p for p in (NAT / r['episode']).iterdir() if p.is_dir() and (p / 'action_000000.json').exists()][0]
    imgs = dict(zip(F.RouterV0.VIEWS, r['frames'])); st = json.load(open(inner / ('action_%06d.json' % T)))['obs']['states']
    res = {}
    for mode in ('log', 'offer', 'auto', None):
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as od:
            dst = Path(od) / inner.name; dst.mkdir()
            for k in range(T + 1): (dst / ('action_%06d.json' % k)).write_text((inner / ('action_%06d.json' % k)).read_text())
            ad = FakeAd(od, td, T, st, imgs); cfg = dict(hv2=True, insert_charger={}, infer_url='http://127.0.0.1:9')
            if mode: cfg['l3_policy'] = dict(name='router_v0', mode=mode)
            it = CT.after_act(ad, cfg, None); lg = [json.loads(l) for l in open(Path(td) / 'l3-gate.jsonl')] if (Path(td) / 'l3-gate.jsonl').exists() else []
            res[str(mode)] = dict(logged=len(lg), p=lg[0]['p'] if lg else None, decision=lg[0]['decision'] if lg else None, mode=lg[0]['mode'] if lg else None,
                                  offer=[i['text'] for i in it if 'insert_charger' in i.get('text', '')], err=(Path(td) / 'l3-gate-error.txt').read_text() if (Path(td) / 'l3-gate-error.txt').exists() else None)
    S, A = HL.StepLog(inner.parent)(); S, A = S[:T + 1], A[:T + 1]
    ref = RC.decide(dict(task='plug_in_charger', step=T, step_limit=400, states_hist=S, cmd_hist=A, executor='insert_charger', images=imgs, tool_facts={}))
    filt = res['log']['p'] is None and res['log']['decision'] == 'astra' and ref['decision'] == 'astra'   # release skips the image pass without a hold (training: 10 non-hold states, 0 routed, 0 positives)
    ok = ((res['log']['p'] == ref['p'] or filt) and not res['log']['offer'] and (res['offer']['p'] == ref['p'] or filt) and len(res['offer']['offer']) == (1 if ref['decision'] == 'intervene' else 0)
          and res['auto']['mode'] == 'offer' and res['None']['logged'] == 0 and not any(v['err'] for v in res.values()))
    R['rows'].append(dict(ep=r['episode'], t=T, ref=ref, res=res, ok=ok))
R['all_pass'] = all(x['ok'] for x in R['rows']); Path('/lizekai/RSIws/experiments/harness-v2-20261028/release/tests/RESULT-hv2f-router.json').write_text(json.dumps(R, indent=1, default=str))
print(R['all_pass'], [(x['ok'], x['ref']['decision'], x['ref']['p']) for x in R['rows']])
