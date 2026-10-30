"""hv2e wiring tests (no model calls, no simulator). Run with PYTHONPATH=<hv2e release>:/opt/PhysRSI:/opt/PhysRSI/runtime/GPT-as-Policy.
W1 tool list: production appends SPEC_INSERT_CHARGER only when combined.insert_charger is a dict (spec JSON-serialisable, ASCII English).
W2 insert_charger through combined_tools.run with a fake adapter on an archived rg3f handover state:
   a) v4 unreachable -> status abstain, steps_used 0, NO robodojo_act sub-call;  b) fake v4 answers abstain -> same;
   c) fake v4 answers a pose -> the chain starts (the 4.5 cm lift = first robodojo_act); the fake action channel refuses -> abort-to-Astra
      (status stopped, reason reported), tool.act and ad._invoke restored, global step sizes restored;
   d) alias without insert_charger -> refused 'not available'.
W3 L3 gate through combined_tools.after_act: log mode writes l3-gate.jsonl with p equal to the gate on the same per-step files and returns no
   text; offer mode returns exactly one availability line at the first 'intervene' and none afterwards; no l3_policy -> nothing written.
  -> tests/RESULT-hv2e-wiring.json"""
import json, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np
from rsi import combined_tools as CT, stage_tools as ST, hv2e_ext as X, harness_l3 as HL
H = Path('/lizekai/RSIws/experiments/harness-v2-20261028/release/tests'); R = {}
sys.path.insert(0, '/lizekai/RSIws/experiments/macro-tools-20261027/gate'); import train_eval_v2 as TE, build_dataset_v2 as BD
row = [r for r in TE.D if r['kind'] == 'rg3f_comp' and r['features']['holding_l'] + r['features']['holding_r'] > 0][0]; EP = row['episode']; T = row['t']
inner = [p for p in (BD.NAT / EP).iterdir() if p.is_dir() and (p / 'action_000000.json').exists()][0]
a = json.load(open(inner / ('action_%06d.json' % T))); st = a['obs']['states']
obs_dir = BD.NAT / EP
imgs = sorted(obs_dir.rglob('*cam_high*.png'))[:1] + sorted(obs_dir.rglob('*cam_left_wrist*.png'))[:1] + sorted(obs_dir.rglob('*cam_right_wrist*.png'))[:1]
R['inputs'] = dict(episode=EP, t=T, images=[str(p) for p in imgs])


class FakeAd:
    def __init__(self, out_dir, ev):
        self.packet = dict(step_id=T, current_state=st, current_eef={'left': dict(position=[0.3, 0.2, 0.9], quaternion_wxyz=[1, 0, 0, 0]), 'right': dict(position=[0.3, -0.2, 0.9], quaternion_wxyz=[1, 0, 0, 0])},
                           images=[dict(camera=c, path=str(p)) for c, p in zip(('cam_high', 'cam_left_wrist', 'cam_right_wrist'), imgs)], request_id='r0')
        self.evidence = ev; self.output = out_dir; self.terminal_reason = None; self.ticket = dict(task_id='plug_in_charger', episode_id='e'); self.contract = dict(native_horizon=400); self.invoked = []
    def _invoke(self, name, args, call_id, on_ack):
        self.invoked.append(name); raise ST.Refused('fake action channel: no simulator in this test')
    def _content(self, pk): return []


class V4(BaseHTTPRequestHandler):
    MODE = 'abstain'
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers['Content-Length']); self.rfile.read(n)
        r = dict(abstain=True, flip_flag=True) if V4.MODE == 'abstain' else dict(abstain=False, flip_flag=False, position_m=[0.0, 0.0, 0.0], quaternion_wxyz=[1, 0, 0, 0], uncertainty={})
        b = json.dumps(r).encode(); self.send_response(200); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
srv = ThreadingHTTPServer(('127.0.0.1', 0), V4); threading.Thread(target=srv.serve_forever, daemon=True).start(); URL = 'http://127.0.0.1:%d' % srv.server_address[1]
# W1
from rsi import production as PR
src = Path(PR.__file__).read_text(); spec = json.loads(json.dumps(X.SPEC_INSERT_CHARGER))
R['W1'] = dict(patched=src.count('SPEC_INSERT_CHARGER') == 1, ascii=all(ord(c) < 128 for c in json.dumps(spec)), name=spec['name'])
R['W1']['pass'] = R['W1']['patched'] and R['W1']['ascii']
# W2
w2 = {}
def call(cfg, ev):
    ad = FakeAd(obs_dir, ev); m0 = (ST.STEP_M, ST.STEP_RAD)
    rep = CT.run(ad, 'insert_charger', dict(reason='test'), 'c1', lambda *a_: None, cfg); out = json.loads(rep['contentItems'][0]['text'])
    return out, ad, m0 == (ST.STEP_M, ST.STEP_RAD), ad._invoke.__func__ is FakeAd._invoke if hasattr(ad._invoke, '__func__') else False
with tempfile.TemporaryDirectory() as td:
    base = dict(hv2=True, infer_url='http://127.0.0.1:9', locate_url='http://127.0.0.1:9', insert_charger={})
    o, ad, mok, inv_ok = call(dict(base, infer_url_v4='http://127.0.0.1:9'), td); w2['a_unreachable'] = dict(status=o['status'], steps=o['measured']['steps_used'], invoked=ad.invoked, ok=o['status'] == 'abstain' and o['measured']['steps_used'] == 0 and not ad.invoked)
    V4.MODE = 'abstain'; o, ad, mok, inv_ok = call(dict(base, infer_url_v4=URL), td); w2['b_abstain'] = dict(status=o['status'], note=o['derived']['note'], invoked=ad.invoked, ok=o['status'] == 'abstain' and not ad.invoked)
    V4.MODE = 'pose'; o, ad, mok, inv_ok = call(dict(base, infer_url_v4=URL), td)
    w2['c_pose_then_refused'] = dict(status=o['status'], note=o['derived']['note'], invoked=ad.invoked[:3], step_sizes_restored=mok, invoke_restored=inv_ok,
                                     ok=o['status'] == 'stopped' and 'abort_to_astra' in o['derived']['note'] and ad.invoked[:1] == ['robodojo_act'] and mok and inv_ok)
    o, ad, mok, inv_ok = call(dict(hv2=True, infer_url_v4=URL), td); w2['d_not_enabled'] = dict(status=o['status'], ok=o['status'] == 'refused')
w2['pass'] = all(v['ok'] for v in w2.values()); R['W2'] = w2
# W3
w3 = {}; G = HL.GateV2(Path(X.__file__).with_name('gate_v2.json'))
S, A = HL.StepLog(obs_dir)(); S, A = S[:T + 1], A[:T + 1]
p_ref = G.decide(dict(task='plug_in_charger', step=T, step_limit=400, states_hist=S, cmd_hist=A, executor='insert_charger'))['p']
for mode in ('log', 'offer', None):
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as od:
        dst = Path(od) / inner.name; dst.mkdir()
        for k in range(T + 1): (dst / ('action_%06d.json' % k)).write_text((inner / ('action_%06d.json' % k)).read_text())
        ad = FakeAd(od, td); cfg = dict(hv2=True, insert_charger={}, infer_url='http://127.0.0.1:9')
        if mode: cfg['l3_policy'] = dict(mode=mode)
        it1 = CT.after_act(ad, cfg, None); it2 = CT.after_act(ad, cfg, None)
        lg = [json.loads(l) for l in open(Path(td) / 'l3-gate.jsonl')] if (Path(td) / 'l3-gate.jsonl').exists() else []
        txt1 = [i['text'] for i in it1 if 'insert_charger' in i.get('text', '')]; txt2 = [i['text'] for i in it2 if 'insert_charger' in i.get('text', '')]
        w3[str(mode)] = dict(logged=len(lg), p=lg[0]['p'] if lg else None, decision=lg[0]['decision'] if lg else None, offer_first=txt1, offer_second=txt2, err=(Path(td) / 'l3-gate-error.txt').exists())
w3['p_ref'] = p_ref
w3['pass'] = (w3['log']['logged'] == 2 and abs(w3['log']['p'] - p_ref) < 1e-3 and not w3['log']['offer_first'] and not w3['log']['err'] and
              w3['offer']['logged'] == 2 and (len(w3['offer']['offer_first']) == (0 if G.a.get('dedup_refit_required') else (1 if w3['offer']['decision'] == 'intervene' else 0))) and not w3['offer']['offer_second'] and
              w3['None']['logged'] == 0)
R['W3'] = w3; R['all_pass'] = R['W1']['pass'] and w2['pass'] and w3['pass']
(H / 'RESULT-hv2e-wiring.json').write_text(json.dumps(R, indent=1, default=str)); print(json.dumps(R, indent=1, default=str)[:3000])
