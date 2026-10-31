# GT P1: K1 insert-skill BC on official demos (tool line ip4, 2026-10-31 03:0xZ). Dev level, offline.
Recipe:
- Inputs: DINOv3 [CLS, mean patch] × 3 views from the router cache; state14; GT history of the last 8 steps (state, action) = teacher forcing; task token.
- Output: 16-step action chunk, masked at the episode end.
- Model: a 4-layer transformer, d = 256.
- Loss: L1, normalised with train-only stats. Arm targets are residuals to the current state; grippers are absolute.
- Split 80/10/10 by episode index. 40 epochs, DEV-best on val.
- Code: src/bcnet.py, src/train_bc.py.
- Policy for C1: src/policy.py (load / act). Handoff: ops/handoff/bc-insert-skill/READY.json -> models/bc-r2-k1.pt.

## Teacher-forced (TEST, rad, arm mean abs)
- Scale: one 25 Hz step of motion is about 0.011 rad, and the zero-motion chunk baseline is 0.115.

| model | first step | chunk (16) | grip chunk |
|---|---|---|---|
| per-task r1 (plug / key / tubes / coin) | 0.0054 / 0.0059 / 0.0032 / 0.0048 | 0.0185 / 0.0203 / 0.0103 / 0.0180 | 0.014 / 0.015 / 0.009 / 0.017 |
| **joint K1 r2-k1** (all) | **0.0032** | **0.0121** | 0.0090 |
| joint, no history | 0.0036 | 0.0139 | 0.0094 |
| joint, absolute targets | 0.0131 | 0.0165 | 0.0105 |

- The joint model beats the per-task models on every task; the shared data helps.
- Per stage (joint, chunk):
  - plug: reach 0.020, hold 0.011, release 0.013;
  - key: reach 0.008, hold 0.019, release 0.014;
  - tubes: about 0.009 in all stages;
  - coin: 0.011–0.014.
- The hardest stages are plug reach and key hold.
- The error grows linearly over the chunk: 0.003 at step 0 to 0.022 at step 15.

## Free-running offline (VAL)
- Setup: own actions fill the history; state[t+1] = own action[t] (the exact demo semantics); images stay GT; chunk executed for E steps; error vs the GT action at the same time index.

| model | free arm rad (all) | plug | key | tubes | coin |
|---|---|---|---|---|---|
| r2-k1 (E = 8) | 0.106 | 0.199 | 0.063 | 0.049 | 0.145 |
| + input noise 0.3 (DART) | 0.068 | 0.117 | 0.041 | 0.041 | 0.075 |
| + noise 0.1, E = 4 | 0.050 | 0.063 | 0.040 | 0.040 | 0.060 |

- Exposure-bias gap: teacher-forced 0.012 vs free-running 0.05–0.11. That is 4–9× from feeding the policy its own history and state.
- The gap is largest for plug and coin, at reach and hold, where the gripper timing errs (free grip error 0.13 / 0.09 vs 0.01 teacher-forced).
- History is NOT the main channel: the no-history and history-dropout ablations give the same free error. The cheap fixes are input noise and replanning more often.
- Caveat: the offline free run keeps GT images and compares by time index. A policy that is correct but slower or faster is penalised. The sim rollouts (P2) are the real measure.

## Harness checks (policy_smoke / policy_diag, sim driver python, host2)
- The features recomputed in the policy equal the router cache (max diff 0.002).
- policy.py path == training path (identical errors).
- Fixed: the harness queries every 8 steps, so policy.executed now fills the unobserved per-step states with the executed actions (state[t+1] = action[t]). Before the fix, the history was 8× too sparse and the chunk error was 0.09–0.41 rad.
- Known weak spot: the first chunk at t = 0 for plug / coin is 0.35 / 0.29 rad (start-time and arm ambiguity). Handover starts are mid-task and not affected.

Queued: r6/r7 (noise 0.3–0.5 × E 4 × H 8/16, 80 epochs), r8 per-task under the best recipe, r9 seed check. The READY checkpoint will move to the best free-running model once r6/r9 are in.

## Update 03:4xZ: v2 handoff (ops/handoff/bc-insert-skill-v2/READY.json; v1 is untouched)
Winner beyond seed noise: input noise 0.3 + 80 epochs + replanning every 4 steps (r7-k1-n03-e4-ep80).

| recipe | free-running VAL | teacher-forced TEST chunk | seed spread |
|---|---|---|---|
| v2 winner (3 data-order seeds) | 0.0447–0.0455 | 0.0114 | 0.001 |
| noise 0.3, exec 8 (3 seeds) | 0.068–0.073 | – | 0.005 |
| v1 r2-k1 | 0.106 | 0.0121 | – |

- Per task, free-running: plug 0.059, key 0.040, tubes 0.035, coin 0.052.
- 120 epochs gave 0.0428 (one seed; within about 2× the seed spread, so it was not adopted).
- Per-task models under the same recipe are worse: free 0.057–0.080, teacher-forced 0.013–0.021.
- No history: 0.079, worse. So under noise the history channel helps.
- policy_v2 fills a stationary history at mid-task starts. First chunk at a t = 40 start: 0.012–0.018 rad, vs 0.018–0.049 with the v1 wrapper.
