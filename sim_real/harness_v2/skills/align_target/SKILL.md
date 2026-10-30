# align_target (harness v2 RC, v1, charger)

Gripper (end-effector, link6) poses that put the held charger on an outlet's axis: one with the prongs 15 mm above the seated depth ("hover") and one at the seated depth ("seated"), for each of the three outlets, computed from the outlet positions (locate_mm) and the charger pose in the gripper (inhand_pose), with the predicted error. No motion, 0 control steps.

**Inputs.** strip (outlet strip position, top height, yaw from locate_mm); inhand (charger pose in the gripper from inhand_pose, with its uncertainty); hover height (default 15 mm).
**Returns.** {value: targets [{outlet, hover_link6, seated_link6, wrist_turn_from_now_deg, hand_travel_m}], uncertainty: predicted 1-sigma at the charger (mm, deg); abstain; steps_used 0; note}. No poses are returned when an input is missing or the predicted 1-sigma error exceeds 2 mm or 2 deg.
**Evidence (pre-registered, deploy inputs, Astra's own plug holds).** 48 of 76 states answered; in 44 of 48 the computed hover hand pose would put the charger within 2 mm and 2 deg of the true outlet axis (median 0.61 mm, 0.61 deg).
**Known limits.** Charger only. Reaching the hover pose with the current hold was possible in 12 of 42 tested states, and a scripted straight descent and release from these poses did not seat the charger even with exact targets (0 of 12); the pose is accurate, inserting is a separate problem.
**Cost.** 0 steps.
