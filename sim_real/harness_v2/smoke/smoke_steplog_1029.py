"""gap (b) smoke: StepLog on a directory that grows file by file (copies of an archived episode) gives the same gate v2 decision at every
takeover row as the archived features; partial (being-written) files are skipped until complete.
  /opt/PhysRSI/ENV/robodojo/bin/python smoke/smoke_steplog_1029.py -> out/SMOKE-steplog-1029.json"""
import json, shutil, sys, tempfile
from pathlib import Path
import numpy as np
H = Path('/lizekai/RSIws/experiments/harness-v2-20261028'); sys.path.insert(0, str(H)); sys.path.insert(0, '/lizekai/RSIws/experiments/macro-tools-20261027/gate')
from rsi_v2 import policy as P, steplog as SL
import build_dataset_v2 as BD, train_eval_v2 as TE
G = P.GateV2(); rows = [r for r in TE.D if r['kind'] == 'rg3f_comp'][:12]; res = []
for r in rows:
    src = [p for p in (BD.NAT / r['episode']).iterdir() if p.is_dir() and (p / 'action_000000.json').exists()][0]
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / src.name; dst.mkdir(); sl = SL.StepLog(td)
        for k in range(r['t'] + 1):
            shutil.copy(src / ('action_%06d.json' % k), dst / ('action_%06d.json' % k))
            if k == r['t'] // 2: (dst / ('action_%06d.json' % (k + 1))).write_text('{"executed_act')   # half-written next file
            if k % 7 == 0 or k == r['t']: sl()
        S, A = sl()
        d = G.decide(dict(task=r['task'], step=r['t'], step_limit=400, states_hist=S, cmd_hist=A, executor='insert_charger'))
        x = np.asarray(TE.x_of(r)); pref = float(1 / (1 + np.exp(-(G.w @ x + G.b))))
        res.append(dict(ep=r['episode'], t=r['t'], n=len(A), p=d['p'], p_ref=round(pref, 4), ok=abs(d['p'] - pref) < 1e-3))
out = dict(rows=res, all_ok=all(x['ok'] for x in res)); json.dump(out, open(H / 'out/SMOKE-steplog-1029.json', 'w'), indent=1); print(out['all_ok'], [x['ok'] for x in res])
