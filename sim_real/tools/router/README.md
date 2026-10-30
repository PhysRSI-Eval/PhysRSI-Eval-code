# ROUTER-RGB-v1: deploy interface for the harness L3 router slot (hv2g)

**Inputs.** The deploy packet only: head, left-wrist and right-wrist RGB, plus proprio/progress. No task id, no task text, no tool facts.

**Decision.** Route to the C1 executor iff BOTH hold:
1. **E1 says this is a C1 scene and an object is held** (rsiws-dc, ee-physword-20261029/src/train_router.py).
   - Checkpoints: out/router-dev-rgb-f{0..4}.pt.
   - Inputs: pooled DINOv3 tokens. The feature function is rsiws-dc's (train_pw.pool_cam).
   - Dev, 5 folds, held-out C1 tasks: AUROC c1 ~ 1.00, route ~ 0.997, false_route 0.6-2.9 %.
2. **The takeover head says p > 0.65** (router_rgb.RouterRGB.takeover; weights in models/takeover.pt).
   - Trained on the 67 deduplicated plug rows (frozen rg3f executor, 12 positives).
   - Dev CV, no task id and no facts (DEV-b-*.json): routed 0.254 = oracle; always-Astra 0.075; always-rg3f 0.209; logistic 0.239; harmful 0.

**Feature function.** router_rgb.dino_features computes [CLS, mean patch] of the frozen DINOv3-B/16 (svc/dinov3-weights/dinov3-vitb16)
on the 640x480 frames, plus router_rgb.tab(gate-v1 proprio features).

**Status.** Dev level. The final level-(b) judgement is pre-registered once the C1 line's 83-state truth-ceiling labels exist
(PREREG-ROUTER-RGB-v1.md, Amendment 1). Retrain with fit_final.py whenever new executor-outcome rows arrive.

**Do not edit the hv2f release.** The bottleneck line wires this into hv2g.
