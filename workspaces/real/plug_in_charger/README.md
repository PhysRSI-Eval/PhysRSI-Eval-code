# Real workspace scaffold

This is an empty environment scaffold, **not a successful workspace**. The launcher generates this structure independently for every Harness round in all three conditions. The actor records the robot's own observations, source/reset identity, object-relative frames and causal experience in its live workspace. No simulated scene coordinates are supplied as live commands.

`memory/environment/` is replaced at each reset. Long-lived methods go in `skills/`, learned programs in `tools/`, reasoning in `memory/`, and policy in `policy/`. Successful committed snapshots can be exported with `python -m physrsi_eval.workspace export`; all originals are preserved and each file must be classified before transfer. `memory/library/` is historical source evidence, not live robot state.
