# grasp_compatible (harness v2 RC, v1.1, scope-limited)

Two-finger grasp candidates for a named object, generated from its CAD model and its estimated pose: grasp pose, approach direction, closing axis, opening before the grasp, and whether each is reachable from the current joints, collision-free (table, the object's support, the other arm), its margin and whether the next stage stays reachable. Info mode: no motion, 0 control steps. An execute mode exists and runs only when you call it explicitly, with a step cap of 150.

**Inputs.** object (class name); object pose (from locate_mm); arm; optional next-stage object poses; mode "info" (default) or "execute".
**Returns.** {value: candidates [...], n_candidates, n_reachable_collision_free; abstain; steps_used; note}. Execute returns held yes / no / unknown with its source (the finger reading cannot tell objects thinner than 5 mm from an empty grasp: then "unknown").
**Evidence and scope.** make_kong replacement tile from Astra's states with the arms idle at the end of phase 1: lifted and held 24 of 26 with the tile pose from the head camera (about 100 steps). Over all tested states (including states where Astra's arm was busy at the wall) 29 of 45, which failed the pre-registered 0.75 bar: deployable in the idle-arm scope only, not general. A gripper that holds something is never used.
**Cost.** Info 0 steps (several seconds to minutes of computation); execute <= 150 steps.

**Next-stage rule (folded in from rsiws-ee, deposit_coin OPPOSITE-v1 and G1; development level).** The insertion stage is checked before any grasp is chosen. For each arm (receptacle side first) and grasp depth (deepest first):
(a) the pre-grasp, the grasp and the lift are reachable;
(b) the insertion pose (receptacle functional frame x inverse of the object's functional frame) is reachable, collision-aware, together with its hover, from the lift joints. The pose may be spun about the object's symmetry axis and leaned up to the measured acceptance of the opening (coin slot: 20 deg).
Candidates are ranked by lean, then by joint travel. If no arm passes, the tool reports "no single-arm grasp keeps the insertion reachable (a hand-over would be needed)". It never starts a single-arm attempt in that case. Evidence: deposit_coin layout 0 coin-side single-arm insertion exists at lean 0 (Astra states 16/43 within 300 steps, versus 0/43 with a hand-over); in 7 other opposite-side layouts no reachable insertion exists at any lean.
