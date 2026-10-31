"""C1 smoke (10-31): load the plug KPNet (kp_api) and the DINOv3 head (dinokp) and run each once on a logged idle-arm wrist image."""
import glob, json, sys
import numpy as np
from PIL import Image
sys.path.insert(0, '/lizekai/RSIws/experiments/kpfactory-20261029/src')
M = '/lizekai/RSIws/experiments/kpfactory-20261029/models/'
img = None
for f in sorted(glob.glob('/lizekai/RSIws/experiments/macro-tools-20261027/out/c1vh4h1/*/result.json')):
    for v in (json.load(open(f)).get('vs') or {}).get('views', []):
        if v.get('image') and v['tag'] == 'A': img = v['image']; break
    if img: break
print('image', img)
im = np.asarray(Image.open(img).convert('RGB'))
import kp_api, dinokp
a = kp_api.detect(M + 'kp2-kpnet-free.pt', [im], 0)[0]; print('kpnet', len(a), [round(x, 1) for x in a[5][:3]])
d = dinokp.Detector(M + 'kp2-dino-free.pt', 0)(im); print('dino', len(d), [round(x, 1) for x in d[5][:3]])
