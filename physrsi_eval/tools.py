"""Dispatch of the agent's tools for one round (Context = K0, Harness = K1..K17).

Context tools : robodojo_contract, robodojo_start, robodojo_act, r1_notes
Harness adds  : m1_files (own workspace), m1_program (run own code), m1_development (earlier rounds)

Physical actions close at the native terminal, at the reasoning limit, or when
the agent ends its turn. In Harness rounds the grade is then attached to the
reply (terminal_feedback) and only m1_files / m1_development stay usable.
"""
import json
import os
import time

from . import protocol as P
from .evidence import ChainEvidence
from .program import ProgramError, run_program
from .sim import ActionRejected
from .util import write_json


def text(value):
    return dict(type='inputText', text=value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))


class ToolHost:
    def __init__(self, condition, task, episode, k, *, workspace=None, chain=None, program_cfg=None):
        assert condition in ('Context', 'Harness')
        self.condition, self.task, self.episode, self.k = condition, task, episode, k
        self.workspace, self.chain = workspace, chain
        self.program_cfg = program_cfg or {}
        self.notes = ''
        self.receipt = None          # set once the episode is graded
        self.closed_at = None
        self.timed_out = False
        self.limit_notice = False
        self.program_active = False
        self.evidence = ChainEvidence(chain) if chain is not None else None

    # ---- grading ----
    @property
    def closed(self):
        return self.receipt is not None

    def close(self, reason):
        """Close physical actions and grade the native state as it is."""
        if self.receipt is None:
            outcome = self.episode.finish(reason)
            self.receipt = dict(task_id=self.task['task_id'], condition=self.condition, k=self.k,
                                termination_reason=reason, native_success=bool(outcome.get('native_success')),
                                score=float(outcome.get('native_score') or 0.0), timed_out=self.timed_out,
                                native_control_steps=outcome.get('native_control_steps'),
                                decisions=outcome.get('decisions'))
            write_json(self.episode.client / 'receipt.json', self.receipt)
            self.closed_at = time.monotonic()
        return self.receipt

    def close_at_limit(self):
        """Measured reasoning reached the limit: grade now; the next tool reply reports it."""
        self.timed_out, self.limit_notice = True, True
        self.close('effective_reasoning_time_limit')

    def feedback(self, live=True):
        return P.terminal_feedback(self.receipt, live=live)

    def _with_feedback(self, result):
        if self.condition == 'Harness' and self.closed:
            result = dict(result, contentItems=list(result['contentItems']) + [text({'terminal_feedback': self.feedback()})])
        return result

    # ---- entry point for one item/tool/call ----
    def call(self, name, args):
        if self.limit_notice:
            self.limit_notice = False
            return self._with_feedback(dict(success=True, contentItems=[text(dict(
                rollout_finished=True, reason='effective_reasoning_time_limit'))]))
        if self.closed:
            if self.condition == 'Context':
                return dict(success=False, contentItems=[text(P.CONTEXT_POST_TERMINAL)])
            if name == 'm1_development':
                return self._with_feedback(self.development(args, include_current=True))
            if name != 'm1_files':
                return self._with_feedback(dict(success=False, contentItems=[text(P.POST_TERMINAL_REFUSAL)]))
            return self._with_feedback(self._safe(self.files, args))
        handlers = dict(robodojo_contract=self.contract, robodojo_start=self.start, robodojo_act=self.act,
                        r1_notes=self.r1_notes)
        if self.condition == 'Harness':
            handlers.update(m1_files=self.files, m1_program=self.program, m1_development=self.development)
        if name not in handlers:
            return dict(success=False, contentItems=[text('tool not connected for condition')])
        result = self._safe(handlers[name], args)
        return self._with_feedback(result)

    def _safe(self, handler, args):
        try:
            value = handler(args if isinstance(args, dict) else {})
        except (ActionRejected, ValueError, KeyError, TypeError) as error:
            return dict(success=False, contentItems=[text(str(error))])
        return value if isinstance(value, dict) and 'contentItems' in value else dict(success=True, contentItems=value)

    # ---- physical tools (also reachable from m1_program) ----
    def physical(self, name, args):
        if self.closed:
            raise ActionRejected('native episode is closed')
        if name == 'robodojo_contract':
            return self.contract(args)
        if name == 'robodojo_start':
            return self.start(args)
        if name == 'robodojo_act':
            return self.act(args)
        if name == 'r1_notes':
            return self.r1_notes(args)
        raise ActionRejected('unsupervised method')

    def contract(self, args):
        if args != {}:
            raise ActionRejected('contract takes no arguments')
        prior = self.evidence.prior_outcomes(self.k) if self.evidence else None
        return [text(P.public_contract(self.task, self.condition, prior_outcomes=prior,
                                       phase='POST_TERMINAL' if self.closed else 'PHYSICAL_EPISODE'))]

    def start(self, args):
        if args != {}:
            raise ActionRejected('tool takes no actor reset/task/path override')
        content = self.episode.start()
        if self.condition == 'Harness':
            packet = json.loads(content[0]['text'])
            packet['execution_budget'] = dict(phase='PHYSICAL_EPISODE')
            packet['prior_harness_outcomes'] = self.evidence.prior_outcomes(self.k)
            content[0] = text(packet)
        return content

    def act(self, args):
        content = self.episode.act(args)
        if self.episode.closed:
            self.close('native_terminal')
        return content

    def r1_notes(self, args):
        if set(args) != {'text'} or not (args['text'] is None or isinstance(args['text'], str)):
            raise ActionRejected('notes schema violation')
        if args['text'] is not None:
            if len(args['text'].encode()) > P.NOTES_LIMIT_BYTES:
                raise ActionRejected('notes exceed 4096 bytes')
            self.notes = args['text']
            write_json(self.episode.client / 'episode-notes.json', {'text': self.notes})
        return [text(self.notes)] + self.episode.last_observation()

    # ---- workspace tools ----
    def _path(self, relative):
        parts = relative.split('/') if isinstance(relative, str) else []
        if not parts or parts[0] not in P.WORKSPACE_DIRS or '..' in parts or '' in parts:
            raise ActionRejected('outside own Harness workspace')
        path = self.workspace
        for part in parts:
            path = path / part
            if path.is_symlink():
                raise ActionRejected('workspace symlink forbidden')
        return path

    def files(self, args):
        if self.program_active:
            raise ActionRejected('Harness workspace busy')
        if set(args) != {'operation', 'path', 'text'} or args['operation'] not in ('list', 'read', 'write'):
            raise ActionRejected('invalid file operation')
        if args['operation'] == 'list':
            from .util import tree
            return [text(tree(self.workspace))]
        path = self._path(args['path'])
        if args['operation'] == 'read':
            if not path.is_file():
                return dict(success=False, contentItems=[text(dict(error='WORKSPACE_FILE_NOT_FOUND_OR_NOT_FILE',
                                                                  path=args['path']))])
            if path.stat().st_size > P.READ_LIMIT_BYTES:
                return dict(success=False, contentItems=[text(dict(error='WORKSPACE_FILE_TOO_LARGE', path=args['path'],
                                                                  size_bytes=path.stat().st_size,
                                                                  limit_bytes=P.READ_LIMIT_BYTES))])
            return [text(path.read_text(errors='replace'))]
        if args['path'] == P.CLIENT_HELPER_PATH:
            raise ActionRejected('tools/mcp_client.py is host-provided and read-only')
        if args['path'].startswith(P.LIBRARY_PREFIX):
            raise ActionRejected('memory/library/ is a read-only transferred library')
        if not isinstance(args['text'], str) or len(args['text'].encode()) > P.READ_LIMIT_BYTES:
            raise ActionRejected('file text too large')
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name('.' + path.name + '.tmp')
        tmp.write_text(args['text'])
        os.replace(tmp, path)
        return [text({'written': args['path']})]

    def program(self, args):
        if set(args) != {'path'} or not isinstance(args['path'], str):
            raise ActionRejected('invalid workspace operation')
        path = self._path(args['path'])
        if path.suffix != '.py' or not path.is_file():
            raise ActionRejected('own Python program required')
        self.program_active = True
        try:
            exit_code, terminal = run_program(self, args['path'], self.program_cfg)
        except ProgramError as error:
            return dict(success=False, contentItems=[text(str(error))])
        finally:
            self.program_active = False
        content = [text({'exit_code': exit_code, 'native_terminal': terminal})]
        if self.episode.started and not terminal:
            content += self.episode.last_observation()
        return content

    def development(self, args, include_current=False):
        if set(args) != {'path'} or not isinstance(args['path'], str):
            raise ActionRejected('invalid workspace operation')
        below = self.k + 1 if include_current else self.k
        return self.evidence.tool(args['path'], below, current=(self.k, self) if include_current else None)
