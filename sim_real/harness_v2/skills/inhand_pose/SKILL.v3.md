# inhand_pose (harness v2 RC, v3 charger)

Pose of the charger in the gripper (position and orientation in the TCP frame) from the head and wrist cameras, with member agreement as uncertainty. No motion, 0 control steps.

**Inputs.** arm; category (only "charger" is validated).
**Returns.** {value: position_in_tcp_frame_m, quaternion_in_tcp_frame_wxyz; uncertainty: member spread mm / deg, flip caveat; abstain; steps_used 0; note}. If the members disagree about the orientation (spread above the frozen gate 22.8 deg), only the position is returned and abstain is true.
**Evidence (rung 1).** Scripted and regrasped charger holds: median 0.53 mm / 0.48 deg; on Astra-like holds about 0.5 mm.
**Known limits.** About 1 frame in 8 can be a whole 180 deg turn of the charger that the gate does not flag. Other objects: not available.
**Cost.** 0 steps.
