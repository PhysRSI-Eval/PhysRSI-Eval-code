"""post-judgement FIT analysis (reported only; METHOD-FIT-20261030): KPNet vs DINOv3 head pixel error on TEST visible keypoints, per view
type x keypoint group (charger prongs = tips+roots, face, housing corners, hole entries, outlet centres). -> results/FIT-plug-v2.json"""
import glob, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
K = Path('/lizekai/RSIws/experiments/kpfactory-20261029'); sys.path.insert(0, str(K / 'src'))
import kp_api as A, dinokp  # noqa: E402
GR = {'prong': [0, 1, 2, 3], 'face': [4], 'hole_entry': [5, 6, 8, 9, 11, 12], 'outlet_centre': [7, 10, 13], 'housing_corner': list(range(14, 22))}
out = {}
for g in ('free', 'hold', 'head'):
    recs = [json.loads(l) for f in glob.glob(str(K / 'kp-charger-v2' / g / 'test' / 'L*.jsonl')) for l in open(f)]
    dn = dinokp.Detector(str(K / 'models' / ('kp2-dino-%s.pt' % g)), 0); E = {m: {k: [] for k in GR} for m in ('kpnet', 'dino')}
    for r in recs:
        im = np.asarray(Image.open(K / 'data-charger-hover' / r['image']).convert('RGB')); dk = A.detect(str(K / 'models' / ('kp2-kpnet-%s.pt' % g)), [im])[0]; dd = dn(im)
        for m, d in (('kpnet', dk), ('dino', dd)):
            for name, ks in GR.items():
                for k in ks:
                    if r['in_image'][k]: E[m][name].append(float(np.hypot(d[k][0] - r['outlets_uv'][k][0], d[k][1] - r['outlets_uv'][k][1])))
    out[g] = {m: {n: (round(float(np.median(v)), 2), len(v)) for n, v in E[m].items() if v} for m in E}
(K / 'results' / 'FIT-plug-v2.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out))
