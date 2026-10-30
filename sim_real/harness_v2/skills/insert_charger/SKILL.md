# insert_charger (harness v2 RC, executor rg3f, plug_in_charger only)

Executor for plug_in_charger: with the charger already held and lifted, the robot puts it down, locates the outlets with the free arm, re-grasps the charger with a grasp compatible with insertion, carries it through a safe height to the outlet, re-measures the in-hand pose, aligns and inserts it. It runs only when you call it, and it moves the robot.

**Inputs.** arm (optional; default = the arm that is holding).
**Returns.** {value: {inserted, chain_end_step, phase_steps, failure}; uncertainty: the in-hand pose reading used; abstain; steps_used; note}. First, the charger pose in the gripper is measured with no motion. If that measurement abstains (for example a possible 180 deg turn of the charger in the hand), nothing moves: abstain = true, steps_used 0. Otherwise the charger is lifted 4.5 cm and measured again; if that second measurement abstains, only the lift was made. The chain stops at a step cap or on any error and returns control at the current state.
**Evidence (pre-registered, offline counterfactual from your own earlier charger holds, deploy estimates only).** Inserted within the 400-step limit in 24 of 60 states; the same 24 with exact targets. On 52 states with a recorded outcome of your own continuation, the executor did better in 20 and your continuation in 1.
**Known limits.** plug_in_charger only. All successes started by about step 95; the chain uses about 300 control steps (median end step 363). It stops without insertion when the hover pose is not reachable, no outlet position is found, or the re-grasp does not hold.
**Cost.** About 240-300 control steps when it runs; 0 steps when it abstains.
