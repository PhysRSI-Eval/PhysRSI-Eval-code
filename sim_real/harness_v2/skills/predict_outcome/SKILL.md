# predict_outcome (harness v2 RC, v3 model P)

For the arm that holds an object and its planned motion over the next 8 control steps, the predicted consequence for the held object: probability that it slips (>= 3 mm or >= 5 deg) or drops (>= 30 mm) while the gripper stays closed, and the expected in-hand displacement. No images, no motion, 0 control steps; it never proposes, ranks or changes a motion.

**Inputs.** arm; link6_traj (9 poses: now and the next 8 steps); gripper_cmds (next 8 steps); object_class (its CAD size is looked up).
**Returns.** {value: p_slip, p_drop, inhand_disp_mm, inhand_disp_deg; uncertainty: member spread, features outside the training range; abstain ("unsure"); steps_used 0; note}.
**Evidence (PREREG-W7-v3, judged once on fresh Astra episodes).** Slip events: balanced AUROC 0.79 (kinematic rule 0.50) on 1520 fresh samples, 195 positives.
**Known limits.** Per task the AUROC is lower for plug_in_charger and play_Xylophone (about 0.6, few positives).
**Cost.** 0 steps.
