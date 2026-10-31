"""Fresh-seed t=0 frames for the task-ID test (zero quota: native reset only, no controller, no Astra). For each job (task, layout_id):
start the simulator, reset(layout_id) (the env seeds its randomisation with the layout id), observe, save head + wrist RGB (640x480 PNG),
finish, close, remove the scratch.  python -I fresh_t0.py JOBS.json WORKER NWORKERS GPU"""
import json, shutil, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, '/lizekai/RSIws/experiments/combined-online-20261026/plug-offline-eval/v14/src')
import v14_sim as S   # noqa: E402  (RA.Sim with /root/pe-1026 paths; pe_server_ext)
OUT = Path(__file__).resolve().parent / 'fresh'
jobs, w, n, g = json.load(open(sys.argv[1])), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
for j in jobs[w::n]:
    d = OUT / j['task'] / ('L%d' % j['layout'])
    if (d / 'meta.json').exists(): continue
    scratch = Path('/root/pe-1026/sim') / ('cp-tid0-%s-L%d' % (j['task'], j['layout'])); sim = None; t0 = time.time(); meta = dict(j)
    try:
        sim = S.RA.Sim(j['task'], scratch, g); r = sim.rpc.request('reset', layout_id=j['layout']); o = sim.rpc.request('observe')
        d.mkdir(parents=True, exist_ok=True)
        for cam in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'): Image.fromarray(np.asarray(o['images'][cam], np.uint8)).save(d / ('%s.png' % cam))
        meta.update(ok=True, instruction=r.get('instruction'), episode_id=r.get('episode_id'))
        try: sim.rpc.request('finish', reason='tid0')
        except Exception: pass  # noqa: BLE001
    except Exception as e:  # noqa: BLE001
        meta.update(ok=False, error='%s: %s' % (type(e).__name__, str(e)[:300])); d.mkdir(parents=True, exist_ok=True)
    finally:
        if sim is not None:
            try: sim.close()
            except Exception: pass  # noqa: BLE001
        shutil.rmtree(scratch, ignore_errors=True)
        meta['seconds'] = round(time.time() - t0); (d / 'meta.json').write_text(json.dumps(meta)); print(json.dumps(meta)[:200], flush=True)
