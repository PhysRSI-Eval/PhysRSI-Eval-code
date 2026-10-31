# TRACK-A: plug_in_charger end to end (capability line; operating mode v6, 2026-10-29)

## Where it sits in the system
Main-task layer, platform task. Tools go to harness L1 (locate_mm, inhand_pose v4, align_target, grasp_compatible, rg3f executor; next:
the KPNet relative visual servo). The executor's "use / do not use" decision belongs to the router (TRACK-D, L3).

## Current state (evidence)
- **rg3f (frozen, deployable executor)**: rung-4 PASS, comp 24/60 within 400 vs Astra 1/55 on the same states. Files: rg3f/CONCLUSION-rg3-fast.md.
- **rg3f-v2**: FAIL 11/50 on a fresh pool. Attribution: **measurement**. The gate (0.40) was set from the old pool; the truth ceiling
  of the new pool is 0.20 (truth: 31/49 hover unreachable). v2 >= rg3f on the same states (+2 / -0). rg3f iteration stopped (dispatcher 11:2xZ).
- **The two limiters left** (both appear in comp and truth):
  - L-a: Astra's hold (after our re-grasp) cannot reach any hover pose. Truth ceiling 0.20 on the fresh pool, 0.45 on the rg3 pool.
  - L-b: v4 in-hand abstains at the lift (flip flag): 21/60 comp rows. Extra views 0/8, flagged estimates 22-54 mm off.

## Hypothesis for the new sub-approach (finepose KPNet relative visual servo, README-KPNet-visual-servo.md)
Triangulating charger keypoints (prong tips / base) and outlet holes in the SAME pair of idle-arm wrist images gives the relative error
directly, with no in-hand pose. That removes L-b, and calibration errors mostly cancel. It also lets us servo from whatever hold we have,
with a free approach direction, which may help L-a.

## Plan (each step pre-registered where it is judged)
1. **Offline, 0 sim — size of L-a.** For the rg3f / rg3f-v2 truth rows that failed with hover unreachable: does ANY of the following
   reach a seated pose with the actual hold (controller model + cuRobo)? Outlet x {3}, arm x {2}, the charger's 180 deg symmetry (if the
   prongs allow), alternative re-grasp families. Gives the ceiling a VS approach could raise.
2. **Keypoints.** Charger CAD keypoints (v4 lib_v4 charger_0 kp: prong tips and base) plus the 3 outlets x 2 holes from the strip CAD
   (LOC1 layout). One KPNet per wrist camera (kpnet3 recipe, copied read-only from finepose with provenance).
3. **Data (automatic, CAD projection, as finepose §3-4).** Start states = Astra's real holds from the rg3/rg3f/rg3f-v2 pools,
   including the lift-abstain rows. Replay, lift, put the charger at truth-aimed hover / pre-hover poses with perturbations
   (xy +-10 mm, yaw +-20 deg, z 0-40 mm). The idle arm parks at 4-6 side views around the strip. Labels = CAD keypoints projected with
   truth and z-buffer visibility. Train / dev split by episode. The post-training line can train it (coordinate); otherwise aux GPU (~40 min).
4. **VS executor, step budget from the start.**
   - one-shot first move: rg3f transport targets, or a coarse target from the KPNet triangulation if the in-hand pose abstains;
   - FIXED observer: the idle arm parks once at 2 views reachable from its home; no return trips between iterations;
   - <= 2 servo corrections, a single re-check before the guarded insert (R1);
   - target: chain <= 250 steps after the handover.
5. **Rungs.**
   - Rung 1: keypoint accuracy (px) and triangulated relative error vs truth on held-out episodes. Gate calibrated against the truth
     relative-error needed for insertion (rg3 dev: <= 1 mm xy, <= 2 deg).
   - Rung 3: closed loop on fresh Astra holds, comp vs truth, native <= 400. The PASS bar is set from the same pool's truth ceiling (lesson of rg3f-v2).

## First launches
- Step 1 (offline reachability of the hover-unreachable rows).
- Step 3 data generator (biggest lead time), after a 2-state smoke run.

## DEV GATE for the 83-state truth-ceiling run (C1 line, written 2026-10-29T14:47Z, BEFORE any vs6dev result was read)
Dispatcher condition 14:5xZ. Applies to vs6dev = out/c1vs6dev (11 dev starts c1/jobs-vs6-dev.json; code sha256 vs_plug 7398b19a10afbc7c,
vs_exec 830b656b0b205df9; truth keypoints). Judged once when all 11 rows are finished; engineering errors (simulator start) are re-run once.
- G1 no_measurement <= 2/11;
- G2 inserted (R1 checks all pass) >= 4/11;
- G3 median steps after the handover <= 300 (over all rows that reached the insert, i.e. outcome inserted or insert_failed).
All three hold -> launch the 83-unique-start truth run with the SAME code (no edits), reported separately as native_success (within the
native step_lim 400) vs timeout_success (inserted only after 400 under --extend 400), plus within-250 count.
Any miss -> attribute (view / reach / alignment / budget / infra), fix on the dev starts only, re-gate with the same thresholds; no 83-state run
on a known-broken executor. Known fix queued for the next dev iteration (not applied to vs6dev, to keep it single-version): the hold-arm
Pin can command a pose > 5 cm from the measured hand -> 'position bound' assertion (7/83 in vs5) -> clamp the pin target to the measured hand.

## vs6dev judgement against the dev gate (C1, judged once, 2026-10-29 after restart)
- Why every row says failure=probe_done / insertion_ok=False: those are loc1cov.py's OWN fields (its legacy plug path ends with
  end('probe_done') whenever a probe library replaces LC.run; insertion_ok belongs to that unused path). The VS executor's outcome is
  rec['vs'] (outcome, steps, insert{descent, checks, inserted}). vs6 was NOT a measurement-only probe: the R1 insert was called in the
  1 row that aligned (05e90e18: inserted, 466 steps); in the other 10 the executor stopped before the insert.
- vs6dev outcomes (out/c1vs6dev): inserted 1, no_measurement 5, not_aligned 2, lost_measurement 1, no_reachable_target 1, error 1.
- **Gate MISSED**: G1 no_measurement 5/11 (> 2), G2 inserted 1/11 (< 4), G3 median steps 466 (> 300). The 83-state run is not launched.
- Attribution:
  - view coverage 6/11: the idle arm did not reach a planned view (park error 51-162 mm) although the controller model said reachable
    (method / infra: the sim line path differs from the model's fixed-target prediction);
  - alignment 2/11: residual stays 2-7 mm over 2 corrections (method: the remaining idle-arm transit disturbance / pinch swing);
  - infra 1/11: 'position bound' (hold-arm pin > 5 cm from the measured hand);
  - reach 1/11: no outlet/pairing reachable with Astra's hold.
- Next (overfit mode, user 16:4xZ): iterate on the 11 dev starts, ONE change per round, until inserted >= 8/11; round 1 isolates the
  motion part with truth 3-D keypoints (no idle-arm views), view coverage comes back in a later round.
- o1 (truth 3-D keypoints, 11 dev): inserted 3/11 in 175-184 steps (motion works when the target is reachable); D4 reach 6/11
  (direct-jump reach model from the lift pose), D3 2/11 (first move 7 mm / 9 deg short, corrections diverge; one ends in position bound).
  o2 = path reach model (aux); view half vh1 = sim-verified parks, probe on 83 starts (host1). STATE: c1/STATE.json.
- o3 (re-grasp fallback): inserted 3/11; the re-grasp itself held in 5/6 D4 rows, but every one then crashed in fine() (position bound:
  first-move lines did not reach the hover after the re-grasp) -> o4 = clamp fine() targets + log the legs, rerun only those 6 rows.
- vh1 (sim-verified parks, 83 starts, probe): held + target triangulated 52/83, target unseen 18, position bound 12; per-row 3-D error median
  14 mm, p90 464 mm -> consistent with the KPNet line's finding that the parked idle arm disturbs the scene (D1 = view-pose collision).
  vh2 = cuRobo collision-checked views + per-observation displacement (host1).
- Per-difficulty denominators (rsiws-dc census v1, 878 C1 Astra episodes): transport drops 19-45 %; about half lost at reaching hover/align;
  failure to insert after alignment. Our executor rows map: transport drop -> D5, hover/align -> D4 + D3, insert -> D6.
- Fit reading (METHOD-FIT 10-30): overall o1/o2 3/11 -> helped: model-reachable states 3/5 inserted in 175-184 steps (truth 3-D keypoints); all 3 states
  whose first move reached the hover in the sim inserted (3/3), the other 2 stopped 7 mm / 9 deg short (D3) -> hurt: D4 unreachable holds 6/11, D3 first move short in sim 1-2/11 -> fit: route only reachable states to the
  executor (per-state labels c1/out/REACH-LABELS.json for the router); o5 tests whether the re-grasp moves D4 states into that regime.
- o5 (all targets clamped): 0 crashes; overall 0/6. Re-grasp held 5/5 but 4/5 stop 117 mm above the pre-hover (oblique family).
- **Finding (10-30 08:2xZ, offline, c1/tools/self_coll_stalls.py)**: the "controller model says reachable, sim arm stops" class is arm
  SELF-collision: 241/297 census-v2 stalls self-penetrate in the cuRobo sphere model (link3-link5 229, link4-link6 52; median -6.8 mm) vs
  30/300 random joints from the same ranges (median +3.4 mm). Not joint limits (+-10 rad in URDF and USD). reach_model v15 has no
  self-collision -> D4 counts so far partly measured the model, not the hold. o6 = self-collision-aware reach (c1/src/self_coll.py).
- vh2 (82 rows): collision-checked views no longer move the strip (p90 0 mm); held-tip error > 1 mm in 33/56, 32 of them during the
  background staging transport with the charger slipping 15 mm (median) in the hand -> D5 slip, not views. vh4 adds self-collision to views.
- (dispatcher 10-30) **D4 counts before o6 are contaminated by the arm self-collision model defect** (o1-o5 'no_reachable_target' and
  the post-re-grasp height floors); re-count D4 from o6 on.
- D5 slip vs relative VS: the executor ALREADY re-measures the held plug after the transport: every check (check0..2, the last one =
  the pre-insert re-check) is a fresh triangulation of the tips AND the holes from the same two socket views (A then B), and each
  correction is computed from that measurement; the grasp-time (H1/H2) measurement is used only for the one-shot first move. So the
  15 mm slip costs one-shot accuracy (larger first residual), not the final alignment; no extra change needed after o6/vh4 unless the
  checks show held points unseen at the hover (then: aim the socket views at the tips, not the strip centre).

## M2 — truth ceiling, 83 unique plug starts (in-sample; truth 3-D keypoints; o6 code = re-grasp + self-collision reach), out/c1o6t-*
- Overall inserted 18/83 (22 %). Native vs timeout: see c1/STATE.json M2 (handover step + executor steps vs the native 400).
- Helped: the 37 states with a reachable target -> 18 inserted (49 %), executor 178-194 steps (one 363).
- Hurt, by D-code: D4 46 (41 = re-grasp held but the target is still unreachable once self-collision is checked: grasp_compatible's
  next-stage check has no self-collision; 5 = re-grasp not held); D6 10 (R1 descent stalled, depth ok but not 'in'; 5 entered with a
  1.9 mm residual = between the 0.5 mm tol and the 2 mm abort); D3 9 (residual stuck at 8-25 mm, or a lost charger).
- Fit: the VS executor is the right method for reachable states; D4 is a re-grasp CHOICE problem -> o7 (queued); D6 next (tighter
  abort / one more correction before the descent).
- Per-difficulty denominators from rsiws-dc's 878-episode census: transport drop 19-45 % (D5), ~half lost at hover/align (D4+D3),
  insert after alignment (D6).

## M2b — truth ceiling after o7 (self-collision-aware re-grasp) + o7r: 40/83 inserted (48 %, from 22 %); c1/out/TRUTH-CEILING-combined-o7.json
- Helped: D4 46 -> 15 (the re-grasp choice with self-collision moved 31 states into the executor regime; inserted +22).
- Left, by D-code: D3 not_aligned 18 (10 hand stops with the tips 20-80 mm ABOVE the hover = hand/fingers on the strip/table; 4 charger
  lost/rotated in hand = D5; 3 corrections not reached + diverging after big yaw; 1 tilt oscillation); D4 15 (6 no self-collision-free
  candidate in the top 20, 6 re-grasp not held, 3 held but unreachable); D6 insert_failed 10.
- o8 (3 corrections) 0/19: more iterations do not fix D3/D6 (fit: not an iteration-count problem).
- Queued: o9 (D3: hover/lower legs must clear table + strip, closed-jaw hand model) on the 18; o10 (D4 rest: + candidate search 80) on the 15.
- o9 (scene clearance) 0/18 and o10 1/15 -> over-restricted; reverted: best code = o7 (40/83). Queued 10-31: 70- M3 KPNet swap (83 states,
  o7 code, real views), 80- the same with the DINOv3 head, 90- o11 D6 spiral retry (10 o7 insert_failed: 7 entered with the tips 4-27 mm
  above the hover = height floor, 3 aligned to 0.01 mm but stalled 15-22 mm above the seat).

## GT -> DAgger P2 (BC insert skill, zero-quota rollouts from the 83 handover starts), 10-31
- v1 bc-r2-k1 0/88: 81 drops, 75 of them the POLICY opening the holding gripper (gripper 0.36 -> > 0.5 before the loss; action[t] =
  state[t+1]), 6 lost with the gripper closed; median drop at step 24 after the handover. v2 (r7, exec 4) 0/88: 56 drops (48 policy-open),
  27 kept the charger to the budget without inserting.
- The 5 public start rows never ran (loc1cov rejects --t 0 as not holding) -> needs a separate start driver.
- Fit: the BC skill's dominant failure is an out-of-context gripper release, not the motion. Queued: d1 rollout (waits for the d1 READY)
  and a DIAGNOSTIC gripper gate (holding gripper may not open beyond its handover value) on the same 88 starts.
