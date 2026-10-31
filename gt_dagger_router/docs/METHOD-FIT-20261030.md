# Method–task fit matrix (rsiws-32, 2026-10-30 08:3xZ; user directive)

User: we can no longer decide pass/fail purely from numbers. Each task type may suit a different method. A method that fails may simply have been tried on the wrong task. Look at where the new method improves things, not just at comparing numbers.

Rule from now on (all lines):
- A pre-registered aggregate verdict is still recorded as-is, for honesty. It is no longer a kill decision.
- Every result must also give a **fit analysis**:
  - (1) the subset of states/tasks/difficulties (D1–D8) where the method helps, and by how much;
  - (2) where it hurts or is neutral, and why;
  - (3) which other task or condition might fit it better.
- A method is set aside only when it shows no regime where it helps.
- The router's job is exactly to send each state to the method that fits it, so a fit map is router training data.

## Current evidence, read for fit
| Method | Where it helps | Where it doesn't | Why | Better-fitting use |
|---|---|---|---|---|
| Relative VS executor (C1) | Reachable plug states: 3/3 inserted within 175–184 steps (truth keypoints) | Astra-hold-unreachable states (D4, 6/11) | method is fine; the hold limits it | route only reachable states; pair with a compatible regrasp (o5) |
| KPNet keypoints | 4 simultaneous views: dxy 1.1 mm / dz 0.5 mm; 2× px, 3× tip vs DINOv3 head | hold-wrist + head only: 14–22 mm; objects moved between views | view geometry, not the detector | key/tubes from simultaneous Astra frames; plug after collision-free views |
| DINOv3 frozen head | small/thin-object masks (coin wrist 0.885 vs SAM3 0.215) | mm keypoints (tip 9.3 mm) | no fine localisation | segmentation, coarse localisation, router features |
| RGB router E1 (seen tasks) | the fixed benchmark task set: recall 0.885, false-route 0.020; plug takeover 0.254 = oracle | unseen C1 tasks (recall 0–0.54) | it learns the scene, not the category | deploy on the fixed task set; v2 tests category cues |
| Mallet COM forward-tilt grasp (C2) | play_Xylophone: 120/120 | hammer/wrench 0/55 | low clamp force on thin handles (physics) | tall/long objects with thick grip regions |
| rg3f executor + gate/router v0 | plug states with an early hold (Astra holds within ~95 steps) | late holds (steps 110–165 in hv2 arms) | timing window | route by hold time |

## How lines report from now on
For each round, give: overall number → helped subset (D-code / task / condition) → hurt subset → fit hypothesis → next step (improve the method in its regime, or try it on a better-fitting task).
