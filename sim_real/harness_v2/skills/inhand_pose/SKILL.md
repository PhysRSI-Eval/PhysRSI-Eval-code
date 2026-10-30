# inhand_pose (harness v2, v4 charger)

Pose of the charger in the gripper (position and orientation in the TCP frame: link6 axes, origin 0.14 m along link6 x) from the head and
holding-arm wrist cameras plus the arm's pose. No motion, 0 control steps. Reports facts only when called.

**Inputs.** arm; category (only "charger" is validated).
**Returns.** {value: position_in_tcp_frame_m, quaternion_in_tcp_frame_wxyz; uncertainty: keypoint residual (px), member spread (deg), flip margin,
flip_flag; abstain; steps_used 0; note}.
- abstain = true when the estimate is outside its calibrated range.
- With a possible 180 deg flip (flip_flag), no value is returned and the note says "unsure: possible 180 deg flip".
- Otherwise only the position is returned.
**Evidence (PREREG-IP-v4, judged once).** Charger on held-out layouts:
- scripted physical holds: 0.50 mm / 0.60 deg, answered on 97 % of frames, 0 % unflagged flips;
- Astra's own holds on held-out layouts: 0.36 mm / 0.45 deg, 99 % answered, 1.7 % unflagged flips;
- fresh Astra episodes (4 episodes only): 1.2 mm / 1.3 deg, 88 % answered.
Tool parity with the judged numbers: tool/PARITY-v4-charger.json (same flags on 474/474 frames).
**Known limits.** Charger only. A confident 180 deg flip on unusual holds (prongs toward the camera) can still pass, at about 2 % of
answered frames. Not tested in a closed loop online.
**Cost.** 0 steps. About 2-3 s on one GPU.
