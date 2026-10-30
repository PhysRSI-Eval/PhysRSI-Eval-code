# check_held_small (harness v2 RC, coin)

Whether a small object (the coin) is still in the gripper after you lifted it: the coin is segmented in this arm's wrist camera at the close and now; a held coin moves with the camera, a coin left behind shifts in the image. No motion, 0 control steps.

**Inputs.** arm; object: coin; close_step (optional; default = the step right after your last close of this arm).
**Returns.** {value: {held: yes / no / unsure, shift_px, close_step, coin_at_close, coin_now}; uncertainty: the pixel threshold; abstain; steps_used 0; note}. unsure when the coin was not visible in the wrist view at the close.
**Evidence (pre-registered batch, deploy inputs).** After a lift, the answer agreed with whether the coin was really lifted in 29 of 29 cases (21 held, 8 not held); 20 of 20 in development.
**Known limits.** Coin only. Validated after a lift of at least 3 cm; right after the close (no lift yet) the answer is not meaningful.
**Cost.** 0 steps.
