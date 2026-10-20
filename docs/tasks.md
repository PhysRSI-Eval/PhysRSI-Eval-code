# Tasks

The paper evaluates 18 RoboDojo tasks. This repository includes the four used to illustrate cross-task transfer.
Each task is defined by RoboDojo's task file, task config and evaluation layout at the pinned commit and asset
revision. Their sha256 are pinned in `tasks/lock.json` and checked by `python -m physrsi_eval check`.

## Difficulty strata

Strata are computed from the official RoboDojo leaderboard (50 evaluations per task), not from any of our runs.
The 18 tasks are ordered by official success rate, with ties broken by official score, and split into six strata
of three tasks each.

| Stratum | Official success | Tasks |
|---|---|---|
| D1 | 60–62% | match_and_pick_from_conveyor, arrange_largest_number, push_T |
| D2 | 30–46% | **cover_blocks**, **stack_blocks_by_language**, classify_objects_by_language |
| D3 | 4–10% | put_bottles_into_dustbin, play_Xylophone, pour_liquid_into_cup |
| D4 | 0–2% | deposit_coin, organize_table, **insert_key** |
| D5 | 0% (by score) | pack_objects_into_box, fill_pen_holder, store_tools_in_toolbox |
| D6 | 0% (by score) | pour_by_language, **plug_in_charger**, swap_blocks |

## The four tasks

| Task | Category | Steps | Scoring | RoboDojo instruction | Clarification shown to the agent |
|---|---|---|---|---|---|
| `stack_blocks_by_language` | Open | 400 | graded [20, 100] | stack the blocks from bottom to top in the order of blue, yellow, and orange, then reset the robot arm. | none |
| `cover_blocks` | Memory | 800 | graded [0, 0, 5, 15, 30, 100] | Cover the blocks from left to right, remember their colors, then uncover them in the order: red, green, and blue. | Leave all cups upside down after each covering or uncovering step, and avoid shifting the blocks from their original positions. Finally, reset the robot arm. |
| `insert_key` | Precision | 300 | graded [15, 100] | Pick up the key, hand it over to the other hand, insert it into the keyhole, then turn it. | The key should be properly seated in the keyhole and turned to complete the operation, not left tilted at the opening. |
| `plug_in_charger` | Precision | 400 | binary | Plug the charger into the power strip. | Align it with the socket and leave it fully seated rather than tilted at the opening. Then reset the robot arm. |

Each clarification makes explicit a condition that RoboDojo's grader already checks (for example, the final
"reset the robot arm" check or the seating depth). The grader itself is unchanged. At every reset the host
checks the simulator's instruction against `native_instruction` and the step limit against
`native_horizon_steps`. It then shows the agent `public_instruction`.

`task.json` also holds `public_task`. This is the task definition returned in `task_context` and by
`robodojo_contract`. It is kept as in the paper's runs, except that the model and effort fields are filled from
the selected system.
