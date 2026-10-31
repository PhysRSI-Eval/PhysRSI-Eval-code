"""bc_start.py (C1, 10-31): BC policy rollout from the PUBLIC task start state of a plug layout (step 0, no Astra replay). loc1cov.py
rejects --t 0 starts as 'not holding', so this driver creates the simulator itself and reuses bc_rollout.run_bc with --start 1
(same loop, frames, DAgger expert; there is no grasp expert, so rows before the charger is held carry no expert action).
  python bc_start.py --policy P --ckpt C [--exec_n 8] [--budget 400] --start 1 --layout L --gpu G --out DIR"""
import json, shutil, sys, traceback, types
from pathlib import Path
import numpy as np
import bc_rollout as BR

rest = sys.argv[1:]
lay = int(rest[rest.index('--layout') + 1]); gpu = int(rest[rest.index('--gpu') + 1]); out = Path(rest[rest.index('--out') + 1]); out.mkdir(parents=True, exist_ok=True)
BR.OPT['start'] = 1
rec = dict(start=dict(kind='public_start', layout=lay, episode='layout%d-t0' % lay), restore=dict(step=0))
sim = None
try:
    sim = BR._pe_sim.RA.Sim('plug_in_charger', Path('/root/pe-1026/sim') / ('c1-BCST-L%d' % lay), gpu)
    rpc = sim.rpc.request
    def objs(): return rpc('objects')
    rec['reset'] = rpc('reset', layout_id=lay)
    o = rpc('observe'); ch = np.asarray(objs()[BR.LC.CH]['position'], float)
    arm = min(BR.CT.ARMS, key=lambda a: float(np.linalg.norm(np.asarray(o['eef_positions'][BR.CT.ARMS.index(a)], float) - ch)))   # the arm nearer the charger
    BR.run_bc(types.SimpleNamespace(out=str(out)), rec, None, None, None, rpc, objs, arm, None, None, None, None)
except Exception:  # noqa: BLE001
    rec['error'] = traceback.format_exc()[-1500:]
finally:
    if sim is not None:
        sim.close(); shutil.rmtree(str(sim.out), ignore_errors=True)   # scratch only; results are in out/
    (out / 'result.json').write_text(json.dumps(rec, default=str, indent=1))
