# check_reach (harness v2 RC, v2)

Whether an end-effector (link6) pose can be reached by an arm from its current joints with the native controller, whether the arm or gripper touches the table on the way, and, if not reachable, the nearest reachable shift of that pose. No motion, 0 control steps.

**Inputs.** arm; target_pose {position, quaternion_wxyz} (world frame, the frame of your actions); optional boxes (known obstacles).
**Returns.** {value: reachable_from_current, controller_converges, residual_mm / residual_deg, min_clearance_mm, first_contact, collision_free_config_exists, nearest_reachable_offset; abstain; steps_used 0; note}.
**Evidence.** Pre-registered on 55 of Astra's plug states (deploy inputs): "hover not reachable with the current hold" recall 20/20, precision 20/22 (12/12, 12/14 after removing 8 rows with wrong outlet inputs). On 46 new states: every "not reachable" from the controller term was correct (26/26); 12 of 20 "reachable" were really reached.
**Known limits.** Only the table and given boxes are modelled; the held object and other objects are not. The table-collision term produced false "not reachable" answers in tests (clearance -0.7 to -54 mm, the arm still reached); controller_converges is reported separately.
**Cost.** 0 steps, about 0.35 s.
