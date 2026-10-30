"""dev check: X-v1 head fit residual on stored Astra xylophone frames (step 0) per layout."""
import json, os, glob, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rsi_v2 import registry as L1
NAT = Path('/lizekai/RSIws/artifacts/native'); by = collections.defaultdict(list)
for d in sorted(os.listdir(NAT)):
    try: m = json.load(open(NAT / d / 'reset.json')).get('metadata') or {}
    except Exception: continue
    if m.get('task') != 'play_Xylophone': continue
    obs = glob.glob(str(NAT / d / '*' / 'observations' / '000000-cam_high.png'))
    if obs and len(by[m.get('layout_id')]) < 3: by[m.get('layout_id')].append(obs[0])
res = {}
for lay, fs in sorted(by.items(), key=lambda x: str(x[0])):
    for f in fs:
        ctx = L1.Ctx({'cam_high': f}, [0] * 14, {'left': ([0, 0, 0], [1, 0, 0, 0]), 'right': ([0, 0, 0], [1, 0, 0, 0])}, cfg=dict(device='cuda:0'))
        r = L1.T = None
        r = L1.t_locate_mm(ctx, 'xylophone_keys'); res.setdefault(str(lay), []).append((r['uncertainty'] or {}).get('fit_res_mm'))
print(json.dumps(res))
