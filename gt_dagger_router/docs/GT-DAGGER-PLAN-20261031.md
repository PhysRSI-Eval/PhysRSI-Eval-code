# GT → DAgger plan (rsiws-32, 2026-10-31 02:1xZ)

User: read the internal GT data guide (GT.md, not included in this repo) and start using the official RoboDojo GT across the board to train tools and optimise the harness:
- first learn what is correct (the official demos);
- then roll out online and recover from failures, DAgger-style.

Data facts (GT.md):
- 9 tasks × 100 successful demos, 25 Hz, in /lizekai/RSIws/data/official/sim_train. K1 tasks: plug, key, tubes, coin.
- Each frame has head / left / right / third RGB (JPEG), depth, intrinsics + per-step extrinsics, 14-d state and 14-d action (left arm 6 + gripper 1, right arm 6 + gripper 1), and an instruction.
- No object poses, layout/seed, or success/stage labels.
- Fixed split: ep 0–79 train, 80–89 val, 90–99 test.
- Do not shift RGB/action indices. Control semantics, quaternion order, depth scale and gripper units must be verified, not assumed.
- DINOv3 features of all 900 demos already exist (router-rgb/official/feat, every 2nd step, plus state).

Deployment stays RGB + proprio only. Sim truth is used only for labels and as the privileged expert.

## Loop
| Phase | What | Owner | Queue | Due (Z) |
|---|---|---|---|---|
| P0 data | Add actions to the feature cache (every step for action chunks), the 80/10/10 split, normalisation from train only. Verify action semantics offline: action[t] vs state[t+k] → absolute target vs delta, the lag k, gripper units | KPNet/tool line | h1-kp / h2-kp | 04:00 |
| P1 learn correct (BC) | (a) tool: a K1 "insert skill" policy = action-chunk head (H steps) on DINOv3 + state, first per task then all 4 K1 tasks together; offline action error on val/test. (b) harness: ROUTER-BIG heads on GT demos (stage/progress/success-likelihood), already started → retrain with the split | tool line; router line | h*-kp; main-rt + H200 | 08:00 |
| P2 online rollout | Run the BC policy in sim (zero quota, no Astra) from (i) the 83 unique plug handover states and (ii) the public task start states; log every failure state with frames | C1 line (sim harness owner) | c1 queues | 11:00 |
| P3 recover (DAgger) | At the states the policy visits, the privileged expert labels the corrective action chunk: the truth-keypoint VS executor / cuRobo truth planner (C1 code). Aggregate (demos + relabelled visited states, including failed Astra states from the 4688 archive) → retrain → roll out again. Each iteration is a queue item; ≥2 iterations always queued | C1 + tool line | c1 / kp queues | iter 1 15:00, then every ~4 h |
| Router side | Train the router on the same rollouts: success-likelihood / failure detection per stage, and "which recovery" (policy vs VS executor vs Astra) from the aggregated outcomes | router line | main-rt / H200 | with each iteration |

Rules:
- Experiments run only from jobq tmux scripts (ops/jobq/README.md); agents only watch and queue.
- Results are reported fit-style (METHOD-FIT-20261030.md).
- Online Astra episodes (quota) still need first-hand user approval. All sim rollouts here are zero-quota.

## Revision 02:2xZ (user: use teacher forcing; first train tools and optimise the harness on correct trajectories; the router is still important; the router's task classification uses prompt + image directly)
- Teacher forcing everywhere in P1. Sequence models are conditioned on the GT history from the official demos, not on their own predictions:
  - the policy gets GT past actions/state;
  - the router's stage/progress model gets the GT previous stage.
- Exposure bias is handled afterwards by the DAgger iterations (P2/P3).
- Router task ID = instruction text (prompt, available at deploy) + image. The text resolves the language-variant pairs (classify vs classify_by_language, stack vs stack_by_language). Inputs: a local text encoder if one exists in the caches, else a learned embedding of the instruction tokens, fused with the DINOv3 image features. Reported per task on held-out demos/episodes.
