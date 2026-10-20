"""Fixed agent-facing texts of the PhysRSI-Eval protocol.

These strings are part of the experimental condition: the agent receives no
system prompt, only the user prompt ("evaluate task" / "optimize task"), the
tool descriptions below and the tool replies. Change them only on purpose.
"""
import copy
import json
from pathlib import Path

WORKSPACE_DIRS = ('skills', 'tools', 'policy', 'memory')
CLIENT_HELPER_PATH = 'tools/mcp_client.py'
LIBRARY_PREFIX = 'memory/library/'  # cross-task transfer: read-only experience of other tasks
SUMMARY_PATH = 'memory/improvements/K{k}.json'
READ_LIMIT_BYTES = 1024 * 1024
NOTES_LIMIT_BYTES = 4096
PRIOR_OUTCOMES = 3
REASONING_LIMIT_S = 4200
PROGRAM_SILENCE_S = 700

# Host-provided, read-only bridge for m1_program (programs run with the workspace root as cwd).
CLIENT_HELPER = '''"""Generic supervised MCP bridge; all calls consume the host ticket budget.
Import this file via sys.path.insert(0, 'tools').
call('robodojo_contract', {}) returns coordinate/budget definitions.
call('robodojo_start', {}) begins the one assigned episode if not yet started.
call('robodojo_act', args) submits the same bounded action as the direct tool.
Results are ordered text/image content; imageUrl values contain actual PNGs.
Standard output is reserved for protocol messages; diagnostic logs use stderr.
"""
import json,sys,uuid
def call(method,params):
 ident=uuid.uuid4().hex
 print(json.dumps({'id':ident,'method':method,'params':params}),flush=True)
 response=json.loads(sys.stdin.readline())
 if response.get('id')!=ident:raise RuntimeError('MCP reply identity mismatch')
 return response['result']
'''

F3_RULE = 'Harness revises its own files using returned outcome facts in the same CLI after terminal.'

FILE_WRITE_NOTE = (' File writes: the read-only sandbox notice applies only to the disabled shell. '
                   'Create and edit workspace files (the report and skill files) with the m1_files tool: '
                   '{"operation": "write", "path": "<workspace-relative path>", "text": "<full file content>"}. '
                   'These writes are permitted now.')

SUMMARY_CONTINUATION = (' Continue this already graded workspace revision only; write the required summary file with '
                        'm1_files before finishing. Physical actions remain closed.')

DEVELOPMENT_HINT = ' m1_development lists this attempt\'s evidence, including camera images.'

POST_TERMINAL_REFUSAL = 'Physical episode has ended. Only workspace file and revision-check tools remain available.'
CONTEXT_POST_TERMINAL = 'Physical episode has ended. This run has no revision step; finish this turn now.'


def summary_note(k):
    return (' Before finishing, write this round\'s summary to ' + SUMMARY_PATH.format(k=k) + ' with m1_files as a JSON '
            'object {"k": %d, "summary": "<what was tried, what the outcome showed, what you changed for the next '
            'ticket>"}. Earlier memory/improvements/K<j>.json files are history: keep them unchanged. The revision is '
            'not committed without this summary.' % k)


def summary_errors(workspace, previous_snapshot, k):
    """[] when K<k>.json is present and every earlier K<j>.json is byte-identical to the last commit."""
    errors = []
    path = Path(workspace) / SUMMARY_PATH.format(k=k)
    try:
        value = json.loads(path.read_text()) if path.is_file() and not path.is_symlink() else None
    except (OSError, ValueError, UnicodeDecodeError):
        value = None
    if not (isinstance(value, dict) and value.get('k') == k and isinstance(value.get('summary'), str)
            and value['summary'].strip()):
        errors.append('per-K summary required: write %s as {"k": %d, "summary": "<non-empty text>"}'
                      % (SUMMARY_PATH.format(k=k), k))
    if previous_snapshot is not None:
        for old in sorted(Path(previous_snapshot).glob('memory/improvements/K*.json')):
            new = Path(workspace) / old.relative_to(previous_snapshot)
            if not new.is_file() or new.read_bytes() != old.read_bytes():
                errors.append('earlier summary %s must stay unchanged' % old.relative_to(previous_snapshot))
    return errors


def terminal_feedback(receipt, live=True):
    """Outcome facts returned to a Harness agent once physical actions are closed.

    live=True is the copy delivered in the current round (with the file-write and summary notes);
    live=False is the copy shown in later rounds (prior_harness_outcomes, H<k>/failure-feedback.json).
    """
    reason = receipt['termination_reason']
    success = bool(receipt['native_success'])
    failure = 'NATIVE_GOAL_NOT_MET' if reason == 'native_terminal' and not success else None if success else reason
    note = FILE_WRITE_NOTE + summary_note(receipt['k']) if live else ''
    return dict(condition='Harness', logical_k=receipt['k'], rollout_finished=True, termination_reason=reason,
                native_result=dict(success=success, score=receipt['score']), failure_reason=failure,
                failure_mechanism='Use this episode RGB and acknowledged actions; hidden predicate diagnostics are not supplied.',
                revision='Physical actions are closed. In this same CLI, revise your own skills/tools/policy/memory '
                         'using the observed outcome before finishing. Revisions apply only to the next ticket.' + note)


def tool_specs(condition):
    specs = json.loads((Path(__file__).with_name('tool_specs.json')).read_text())[condition]
    tools = copy.deepcopy(specs['dynamic_tools'])
    if condition == 'Harness':
        tools = [dict(t, description=t['description'] + ' ' + F3_RULE) for t in tools]
    return specs['user_prompt'], tools


def public_contract(task, condition, *, phase='PHYSICAL_EPISODE', prior_outcomes=None):
    """Reply of robodojo_contract: robot, frame, action semantics and budget; no task strategy."""
    value = dict(robot='RoboDojo ARX X5 dual arms', frame='shared environment_origin', eef_link='link6',
                 position_unit='metre', orientation='unit quaternion wxyz', gripper_closed='boolean', mode='eef',
                 proprioception='14D: left six joint radians and commanded opening, then right six joint radians and commanded opening',
                 opening='0 closed, 1 open; commanded opening is not a measured gap or grasp confirmation',
                 action_steps=[1, 5], control_hz=25, max_translation_m=.05, max_rotation_rad=.35,
                 action_submissions=None, tool_calls=None, wall_timeout_s=None,
                 native_horizon=task['native_horizon_steps'], task_definition=task['public_task'],
                 effective_reasoning_limit_seconds=REASONING_LIMIT_S, reset='one per assigned ticket; no retry',
                 terminal='physical actions close at native terminal or measured reasoning limit')
    if condition == 'Harness':
        value.update(execution_budget=dict(phase=phase), prior_harness_outcomes=prior_outcomes or [],
                     budget_terminal_rule=F3_RULE,
                     program_stdio={'request': 'one JSON line: {id, method, params}; flush stdout and read one reply line',
                                    'methods': ['robodojo_contract', 'robodojo_start', 'robodojo_act', 'r1_notes'],
                                    'reply': '{id, result: public contentItems}; log diagnostics to stderr',
                                    'workspace': '/workspace read-only during program; m1_files writes between programs',
                                    'limits': {'wall_seconds': None, 'max_calls': None,
                                               'no_output_or_reply_stop_seconds': PROGRAM_SILENCE_S}})
    else:
        value['program_stdio'] = None
    return value
