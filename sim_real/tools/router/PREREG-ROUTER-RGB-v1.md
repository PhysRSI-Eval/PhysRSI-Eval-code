# PREREG ROUTER-RGB-v1 (capability line, Track D, 2026-10-29 16:5xZ; user directive 16:4xZ; zero quota, offline; judged once)

Question: from head + wrist RGB and proprio/progress ONLY (no task id, no task text, no truth-derived or tool facts), can a router
(a) recognise "C1 precision-insertion state: object held, pre-insertion" across all tasks, and (b) make the plug takeover decision as well
as ROUTER-v0 did with facts?

## Inputs (deploy-legal; feature function router-rgb/features_rgb.py, delivered with the model)
- RGB: frozen DINOv3-B/16 (snapshot /lizekai/RSIws/svc/dinov3-weights/dinov3-vitb16, the same as ROUTER-v0), 640x480 frames, per view
  [CLS, mean patch token] -> 3 x 1536 (head, left wrist, right wrist). Identical to gate/build_router_v0.py.
- Proprio / progress: both arms' 6 joints, gripper readings and last commands, eef z, step / 600 (no task-specific step limit).

## Data
- (a) Archived Astra episodes in mva-20261029/astra_v1 (301 valid episodes, 15 tasks). Frames come from artifacts/native/<ep>/<inner>/observations
  (640x480 PNG, same as v0); proprio comes from astra_v1/arrays. One sample every 8 steps.
- C1 tasks present: plug_in_charger, insert_key, insert_tubes, deposit_coin, fill_pen_holder (store_tools is not in astra_v1).
- Label POSITIVE iff all hold:
  - the task is C1;
  - an arm is "holding and lifted": its last gripper command is closed, the reading is in (0.02, 0.9), and the eef is >= 2 cm above
    its z at that close;
  - for deposit_coin, the reading test is dropped (a 1.9 mm coin reads like an empty grip; label noise accepted and stated).
  Everything else is NEGATIVE: all non-C1 tasks at every phase, and C1 tasks before grasping or after release.
  Labels use proprio only (no sim truth). The rule is fixed now.
- (b) The ROUTER-v0 plug rows: 67 deduplicated rows, rg3f executor, gate/ROUTER-DATA-v0.*, same folds as v0. The task id and the tool facts
  are removed; RGB + proprio/progress only.

## Models
- (a) Category head: the ROUTER-v0 transformer (2 layers, d 128, 4 heads), tokens [3 views, proprio], binary C1-held-pre-insertion.
  Class-balanced BCE, AdamW 1e-3, wd 0.05, 60 epochs, minibatch 256, 3 seeds averaged.
- (b) Takeover head: the same architecture and recipe as ROUTER-v0, without the facts token; 200 epochs full batch, 3 seeds.
- Baselines: proprio-only (same head, no image tokens) and ROUTER-v0's numbers.

## Evaluation (judged once)
- (a1) Grouped 5-fold CV by context over all samples. Report AUROC and balanced accuracy at p > 0.5, per task.
- (a2) Leave-one-task-out:
  - for each C1 task T, train on all other tasks' episodes;
  - test on T's samples plus the negatives of a held-out context fold of the other tasks;
  - the gate is on insert_key; the others are reported.
- (b) The routed / always-Astra / always-executor / oracle success and the harmful count, as ROUTER-v0.

## Gates (calibrated on the ceilings / baselines, frozen now)
- (a1) PASS iff AUROC >= 0.90 AND balanced accuracy >= 0.85 AND AUROC >= proprio-only AUROC + 0.10 (RGB must carry the category).
  The ceiling is 1.0. The proprio-only baseline is the "no vision" floor.
- (a2) PASS iff insert_key held-out AUROC >= 0.80.
- (b) PASS iff routed success >= 0.224 AND routed > always-executor (0.209) AND harmful <= 10 % of takeovers.
  - Ceiling = oracle 0.254 (v0 reached it with facts).
  - 0.224 = 2 states of 67 below the ceiling.
- Each of (a1), (a2), (b) gets its own verdict. Fewer than 10 positives in a test group -> that group is INCONCLUSIVE.

## Delivery (integration target hv2f L3 router_v0 slot; the release is not edited)
router-rgb/features_rgb.py (feature function on a deploy packet), router-rgb/models/{category,takeover}.pt, README for the bottleneck line.
STATE: router-rgb/STATE.json, written at each step.

## AMENDMENT 1 (17:1xZ, dispatcher split; before any training)
- Level (a) is dropped from this PREREG. rsiws-dc's E1 (ee-physword train_router.py; its labels / folds / backbone) owns category recognition.
- Level (b) runs at DEV level on the v0 plug rows now. It is retrained as C1-executor outcome labels arrive (c1/STATE.json rounds).
- The FINAL level-(b) judgement is pre-registered separately once the C1 line's 83-state truth-ceiling labels exist.
- samples.json (built 16:5xZ for the dropped level (a)) is kept for the end-to-end routing check and not used for training.
