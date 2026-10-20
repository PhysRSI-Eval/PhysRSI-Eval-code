# Protocol

This page describes what the agent sees and what the host enforces. Agent-facing strings are in
`physrsi_eval/protocol.py`. Tool schemas are in `physrsi_eval/tool_specs.json`.

## Rounds

| | K0 (Context) | K1–K17 (Harness) |
|---|---|---|
| user prompt | `evaluate task` | `optimize task` |
| system prompt | none | none |
| tools | `robodojo_contract`, `robodojo_start`, `robodojo_act`, `r1_notes` | Context tools, `m1_files`, `m1_program`, `m1_development` |
| workspace | none | persistent per chain: `skills/ tools/ policy/ memory/` |
| repetitions | 3 independent episodes, mean score | one attempt per round, stop at first native success, at most 17 rounds |
| after the attempt | turn ends | revision with robot actions closed, then commit |

Every round starts a fresh Codex thread on a fresh app-server process. The only state carried between rounds is
the committed workspace and the chain's own evidence. The agent can read that evidence through `m1_development`.

## The physical episode

- One simulator process per round. The task, its reward and score code, and its step limit (`step_lim`) are
  RoboDojo's own at the pinned commit. Layout group (`eval_seed`) 0, layout 0.
- `robodojo_start` resets once. It returns the observation packet and three camera images (`cam_high`,
  `cam_left_wrist`, `cam_right_wrist`; 640×480, sent as JPEG at quality 90).
- `robodojo_act` takes `{request_id, mode: "eef", reason, steps: 1..5, target: {left, right}}`, where each
  arm has `{position[3], quaternion_wxyz[4], gripper_closed}`. The host rejects the action without executing
  anything if any of these hold:
  - `request_id` is not the current one.
  - `reason` is empty or not in English.
  - A target is more than 5 cm or 0.35 rad from the current EEF pose.
  - A quaternion is not a unit quaternion.
  - `gripper_closed` is not a boolean.
- The simulator runs the action as `steps` control ticks at 25 Hz. On each tick it moves the joints one
  damped-least-squares step toward the fixed target, with the task-space error capped at 2 cm / 0.1 rad and the
  per-joint change capped at 0.05 rad. Only the robot's URDF is used (no scene knowledge, no motion planner).
  The episode ends early at the native terminal.
- The observation packet holds these fields: `episode_id`, `step_id`, `task`, `max_episode_steps`,
  `instruction`, `remaining_steps`, `current_state` (14D: six joints and the commanded opening per arm),
  `current_eef` (position, quaternion, rpy in degrees, commanded opening), `task_context.task_definition`,
  `request_id`, `rollout_finished`, `result` and `previous_result`. In Harness rounds the start packet also holds
  `prior_harness_outcomes` (the outcomes of the last three rounds) and `execution_budget.phase`.
- `r1_notes` reads or replaces a note of at most 4096 bytes for this episode only.

Physical actions close in four cases:

| `termination_reason` | when | grade |
|---|---|---|
| `native_terminal` | RoboDojo's own end: success, or step limit reached | native success / score |
| `actor_completed` | the agent ended its turn first | native state graded as it is, no extra steps |
| `effective_reasoning_time_limit` | 4200 s of measured reasoning | native state graded as it is |

The score is 1.0 on success. Otherwise it is the task's own partial score (`reward_manager.get_score()/100`) if
the task has one, and 0 if it does not.

## Harness tools

- `m1_files {operation: list|read|write, path, text}` works on the own workspace only.
  - Paths start with `skills/`, `tools/`, `policy/` or `memory/`. Files are at most 1 MiB.
  - `tools/mcp_client.py` is provided by the host and read-only. It is rewritten every round.
  - `memory/library/` (transfer only) is read-only.
- `m1_program {path}` runs an own Python file with the workspace root as working directory.
  - The program calls the four physical tools as newline-delimited JSON on stdout/stdin (`tools/mcp_client.py`).
    Each action is validated exactly like a direct call.
  - Any other method stops the program. So does stdout that is not JSON, or 700 s without output or reply.
    The episode continues after the stop.
  - The workspace is read-only while the program runs.
  - The reply holds `{exit_code, native_terminal}` and the current observation.
- `m1_development {path}`: `path=""` lists the evidence; a listed path reads one file. The evidence is
  the same chain's earlier rounds, named `H<k>/...`:
  - `result.json`, `failure-feedback.json`, `reasoning-budget.json`, `history.json`
  - `request_<n>.json`, the observation packets
  - `observations/<n>/<cam>.png`, the camera images
  - `NOTES.md`
  - `workspace/...`, the committed workspace of round k

  During the revision of round k, round k's own evidence is listed too (no `workspace/` yet).
  Context rounds and other tasks are never visible.

## Terminal feedback and revision

When physical actions close in a Harness round, the reply that closed them carries `terminal_feedback`. So does
every later tool reply.

```json
{"condition": "Harness", "logical_k": 3, "rollout_finished": true,
 "termination_reason": "native_terminal", "native_result": {"success": false, "score": 0.15},
 "failure_reason": "NATIVE_GOAL_NOT_MET",
 "failure_mechanism": "Use this episode RGB and acknowledged actions; hidden predicate diagnostics are not supplied.",
 "revision": "Physical actions are closed. In this same CLI, revise your own skills/tools/policy/memory using the observed outcome before finishing. Revisions apply only to the next ticket. File writes: ... Before finishing, write this round's summary to memory/improvements/K3.json ..."}
```

- **Revision in the same turn.** If the episode ended during the agent's turn, the agent revises in that turn.
- **Revision in a new turn.** If the agent ended its turn first (`actor_completed`), the host starts one new turn
  in the same thread. The turn's input is the feedback JSON, with the note that `m1_development` lists this
  attempt's evidence.
- **Revision window.** The window is 1800 s (GPT systems) from the moment actions close. A turn still running
  at the deadline is interrupted.
- **Per-round summary.** The revision must write `memory/improvements/K<k>.json` =
  `{"k": k, "summary": "..."}` and keep earlier `K<j>.json` byte-identical. If they are missing or changed, the
  host starts up to three continuation turns with the validation message, inside the same window.
- **Commit.** The workspace minus the host helper becomes `K<k>/snapshot/`, which is round k+1's starting
  workspace.

## Reasoning meter

The host measures model time from the app-server event stream. A turn's wall time counts, minus the time spent
inside host tool calls: simulator steps, the agent's programs and file I/O are excluded. This covers model
inference and streaming.

Counting only open `reasoning` items is not usable with current Codex clients. They emit a reasoning item only
after the thinking has happened, so its open interval is about 0 s.

At 4200 s the episode is graded at once. The next tool reply returns
`{"rollout_finished": true, "reason": "effective_reasoning_time_limit"}` with the terminal feedback.

## Cross-task transfer

The target chain's K1 workspace holds `memory/library/<source>/` for each source task. Each source folder
contains:

- the final committed workspace of the source chain,
- `k0.json` (the K0 mean score and success),
- `history.json` (per-round scores and successes).

A `README.md` and an `index.json` describe the library, and the target task is never in it. Everything else
about the target chain is identical to a normal chain.
