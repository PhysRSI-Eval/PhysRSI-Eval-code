# Router v4 tool table v0: (task, stage) → method (capability line, 2026-10-30; from FITMAP + METHOD-FIT-20261030; dev, evidence-based)

Routing chain = **task ID** (taskid/, RGB at the episode start, acc 1.00 on seen scenes)
→ **stage** (stage/, K1 temporal model, in progress)
→ the method for that cell, otherwise Astra.
The bracket says where the evidence comes from; "Astra" = no tool helps there yet (leave it alone).

| task | approach / locate | grasp | lift / transport | align + insert | conditions (fit) |
|---|---|---|---|---|---|
| plug_in_charger | Astra (+ locate_mm info) | Astra | rg3f executor if the hold is early (t ≤ ~95) and v4 not abstained [ROUTER-v0 0.254 = oracle] | relative VS executor when the hover is reachable [C1 o1–o2: 3/3 inserted in 175–184 steps, truth kp] | v4 lift abstain → no tool (D2 gap, KPNet held-object line); late hold → Astra |
| insert_key | Astra | Astra | Astra | finepose relative VS (dev 7/7 from a scripted hold; step budget open) | route only after a held key over the lock |
| deposit_coin | Astra | Astra; SSEG post-lift check = deploy 'held?' fact (29/29 = truth) | partial slow lift [coin v2 dev] | slot alignment → C1 executor (pending coin + slot KPNet) | coin chain paused; hand-over layouts out of scope |
| insert_tubes | Astra | Astra | Astra | C1 executor instance (pending KPNet; truth stage planner 79/80) | — |
| fill_pen_holder | Astra | Astra | Astra | (none yet) | no router cue generalises (v2/v3) |
| play_Xylophone | Astra | mallet COM forward-tilt grasp [C2 120/120] | Astra | Astra (strike) | thick-grip long objects |
| store_tools_in_toolbox | Astra | (none: hammer / wrench 0/55, thin handles) | — | — | physics limit of the clamp |
| other 35 tasks | Astra | Astra | Astra | Astra | add a cell only when a method shows a helped regime |

Next: fill the K1 stage cells from the stage model's predictions (stage/RESULT-stage-*.json), and rebuild FITMAP when the C1 line's
83-state truth-ceiling labels land.
