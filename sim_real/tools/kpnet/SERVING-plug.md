# Plug keypoint models: serving for the C1 executor (dev use; KPF-plug-v2 FAILED on view coverage. See CONCLUSION-KPF-plug-v2.md).
Models (kpnet3, K = 22): models/kp2-kpnet-{free,hold,head}.pt in /lizekai/RSIws/experiments/kpfactory-20261029/.
Keypoint order:
- 0 tip_l, 1 tip_r, 2 root_l, 3 root_r, 4 face_c;
- 5 + 3k: hole_a, 6 + 3k: hole_b (hole entries, ordered by strip x), 7 + 3k: outlet centre, for k = 0..2;
- 14..21: housing corners.
```python
import sys; sys.path.insert(0, '/lizekai/RSIws/experiments/kpfactory-20261029/src'); import kp_api as A, rung1_plug_v2 as V
M = '/lizekai/RSIws/experiments/kpfactory-20261029/models/kp2-kpnet-%s.pt'
dets = [A.detect(M % vt, [img])[0] for img, vt, Kc, Tc in views]   # vt in free/hold/head; wrist K = 397.04 / (320, 240); T = FK link6 @ cplabels.MOUNT
P, res = A.triangulate(dets, [(Kc, Tc) for _, _, Kc, Tc in views]); s = V.solve(P, target=k)   # None or dict(dxy, dz, tips)
```
Use two free-arm views (A + B) taken at the SAME static moment. Measured on TEST: 4 views give dxy 1.1 mm; holding wrist + head alone
give 14-22 mm (poor baseline).
