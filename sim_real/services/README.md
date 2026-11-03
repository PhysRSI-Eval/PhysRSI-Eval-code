# Frozen plug-in-charger inference services

Small service/model-definition code is included; large weights and calibration data are referenced in `docs/plug_in_charger/assets.json`. Addresses use `h1005090:/lizekai/...` (SSH access required), not public download URLs. Fetch each asset to the same `/lizekai` location or use the environment overrides below. No weights or private authentication are committed.

Use the site's CUDA Python with numpy, scipy, pillow, torch, torchvision and transformers compatible with the frozen DINOv3 snapshot. This package's host-only tests do not download/run those models. These commands are explicit examples; the experiment launcher never starts GPU services automatically.

```bash
# in-hand v4, data root contains model/, cad/, results/
export PHYSRSI_INHAND_V4=/lizekai/RSIws/experiments/inhand-pose-20261026/v4
export PHYSRSI_DINO=/lizekai/RSIws/experiments/inhand-pose-20261026/model_v1/pkg/v1pkg_1026/dinov3-vitb16
export PHYSRSI_WRIST_CALIB=/lizekai/RSIws/experiments/inhand-pose-20261026/train_v2/objects_v2.json
CUDA_VISIBLE_DEVICES=0 python sim_real/services/inhand_v4/svc_inhand_v4.py --port 18808
# In another terminal; group 3 only uses /outlets here, not the old v1 /inhand endpoint.
CO_KP_CKPT=/lizekai/RSIws/experiments/loc1-20261026/pkg/kp-v1.pt \
python sim_real/services/outlets/co_infer_service.py --port 18807 --gpu 0
```

Also supply the site's existing SAM3/coarse localization endpoint as `plug_tools.locate_url`. Its POST contract receives `{images_b64, states, eef_positions, eef_quaternions_wxyz, object:"power strip", camera:"cam_head"}` and returns `{found, center_xyz}` (plus optional uncertainty); coordinates must match the episode frame. `camera:cam_head` is the source service's selector name; RGB dictionary uses `cam_high`. This external coarse service is a required deployment dependency, not implemented by the in-hand server. A compatible real RGB/depth localization service can implement the same contract.

**Real integration:** current frozen weights and default camera/CAD files are from simulation. Replace/validate the geometric preprocessing (head pose, wrist mount, common rectified intrinsics, distortion, crop, TCP, CAD, per-arm camera mapping) for actual ARX X5 2023 hardware, and evaluate domain shift. `PHYSRSI_WRIST_CALIB` changes wrist geometry but does not by itself port all camera assumptions or learned weights. Record `inference_calibration_id`; leave the real tool profile unconfigured until this work is complete. The two wrist camera views must be rectified into the common calibrated image model expected by this executor, or the estimator must be extended per-arm.

`SOURCES.json` records original and packaged code hashes. Source accuracy claims in vendored comments refer to the original simulation evaluation, not new robot validation.
