# locate_mm (harness v2 RC, v2)

Millimetre position (world frame, metres) and yaw of one task target from RGB: outlets, keyhole, toolbox_slots, tube_holes, mahjong_tile, xylophone_keys. Head-camera targets (toolbox_slots, tube_holes, mahjong_tile, xylophone_keys) cost 0 control steps; outlets and keyhole move the arm that holds nothing to an observation pose and back, and are refused before any motion if the estimated cost exceeds budget_steps (default 0).

**Inputs.** target; budget_steps (wrist targets only); arms_homed (mahjong_tile: validated with both arms at home).
**Returns.** {value, uncertainty, abstain, steps_used, note}. value: outlets_xyz / keyhole_xyz + slot axis / slots_xyz / holes_xyz / tile_xyz / keys_xyz + yaw (lines are mod 180 deg). uncertainty: fit residual, keypoints used, the rung-1 statistics of that target.
**Evidence (rung 1, pre-registered, held-out data).** Median errors: outlets 0.15 mm, keyhole 0.55 mm, toolbox slots 0.21 mm, tube holes 0.11 mm, mahjong tile 0.19 mm; xylophone keys (all-layout model) 0.84 mm along the key row on table layouts never used in training.
**Known limits.**
- Outlets on Astra's own holds (offline, never-used states): 46 of 47 answers within 2 mm, one answer 392 mm off; and once an open gripper resting next to the strip dragged it 138 mm when it moved to observe (rung-4 non-interference check failed). Outlet results are therefore listed as development-level.
- The tile estimate is validated with both arms at home; with arms over the wall it abstains (10 of 49 Astra states).
- keyhole from the head camera alone is not available (failed its test).
**Table layouts checked.** outlets: answers accurate in 14 of 18 layouts, no answer in 4; xylophone keys: all 18 layouts (3 never used in training); keyhole, toolbox slots, tube holes, mahjong tile: their tasks have one layout; checked on 8-9 generated layouts each (objects moved and turned inside and beyond the official ranges): all within their accuracy gates, except the keyhole axis beyond about 30 deg of slot rotation (15 deg error at 35 deg, no answer at 48 deg).
**Cost.** 0 steps (head targets); outlets about 100 steps (view + return), keyhole about 130 steps.
