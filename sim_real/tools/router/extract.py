"""Task-ID router (router v4 step 1, dev): DINOv3-B/16 [CLS, mean patch] (router_rgb.dino_features, same snapshot / 640x480 frames) of the
head + both wrist frames of archived Astra episodes (artifacts/native, all 42 tasks) at t in T_STEPS.  python -I extract.py SHARD NSHARD GPU"""
import json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import router_rgb as RR   # noqa: E402
from transformers import AutoModel
T_STEPS = (0, 10, 20, 40, 80); NAT = Path('/lizekai/RSIws/artifacts/native')
k, n, g = map(int, sys.argv[1:4]); E = json.load(open(Path(__file__).parent / 'episode_index.json'))[k::n]
dev = torch.device('cuda:%d' % g); vit = AutoModel.from_pretrained(RR.DINO).to(dev).eval(); nreg = vit.config.num_register_tokens
X = np.zeros((len(E), len(T_STEPS), 3, 1536), np.float16); ok = np.zeros((len(E), len(T_STEPS)), bool)
for i, e in enumerate(E):
    o = NAT / e['episode'] / e['inner'] / 'observations'
    for j, t in enumerate(T_STEPS):
        fs = [o / ('%06d-%s.png' % (t, c)) for c in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')]
        if not all(f.exists() for f in fs): continue
        ims = {c: np.asarray(Image.open(f).convert('RGB')) for c, f in zip(('cam_high', 'cam_left_wrist', 'cam_right_wrist'), fs)}
        X[i, j] = RR.dino_features(vit, nreg, ims, dev); ok[i, j] = True
    if i % 100 == 0: print(k, i, len(E), flush=True)
np.savez_compressed(Path(__file__).parent / 'feat' / ('shard-%02d.npz' % k), X=X, ok=ok, episodes=np.array([e['episode'] for e in E]))
print('done', k, flush=True)
