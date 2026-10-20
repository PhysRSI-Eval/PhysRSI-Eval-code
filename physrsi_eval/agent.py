"""Codex app-server client (JSON-RPC over stdio) that serves the agent's tool calls.

The agent gets no system prompt and no built-in tools: shell, web search, apps,
plugins, image tools etc. are disabled, the sandbox is read-only, and the only
tools are the host's dynamic tools (tools.ToolHost). Reasoning time is measured
from the client side: wall time while a turn runs, minus time spent inside host tools.
"""
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

from .util import write_json

DISABLED_FEATURES = ('apps', 'plugins', 'remote_plugin', 'memories', 'hooks', 'shell_tool', 'unified_exec',
                     'shell_snapshot', 'multi_agent', 'multi_agent_v2', 'browser_use', 'browser_use_external',
                     'computer_use', 'in_app_browser', 'image_generation', 'view_image', 'skill_search',
                     'skill_mcp_dependency_install', 'tool_suggest', 'workspace_dependencies', 'goals', 'sleep_tool',
                     'request_permissions_tool', 'code_mode')


class ReasoningMeter:
    """Model time of a round: wall time while a turn is running, minus time spent inside host tools
    (simulator steps, programs, file I/O). It includes model inference and streaming; it excludes
    everything the host does. (Counting only open reasoning items undercounts badly: current Codex
    clients emit a reasoning item only after the thinking has happened.)"""

    def __init__(self, limit):
        self.limit, self.active, self.excluded = limit, 0.0, 0.0
        self.tool_depth, self.in_turn = 0, False
        self.last = time.monotonic()

    def advance(self):
        now = time.monotonic()
        elapsed, self.last = now - self.last, now
        if self.in_turn and not self.tool_depth:
            self.active += elapsed
        else:
            self.excluded += elapsed

    def turn(self, running):
        self.advance()
        self.in_turn = running

    def notice(self, message):
        self.advance()
        if message.get('method') == 'turn/completed':
            self.in_turn = False

    def tool(self, begin):
        self.advance()
        self.tool_depth += 1 if begin else -1

    @property
    def at_limit(self):
        self.advance()
        return self.active >= self.limit

    def snapshot(self):
        self.advance()
        return dict(effective_reasoning_seconds=self.active, max_effective_reasoning_seconds=self.limit,
                    excluded_seconds=self.excluded, clock_source='turn wall time minus host tool time')


class CodexAgent:
    """One fresh app-server process and one thread per round; revision turns reuse the thread."""

    def __init__(self, system, folder):
        self.system, self.folder = system, Path(folder)
        self.home = self.folder / 'codex-home'
        self.home.mkdir(parents=True, exist_ok=True)
        auth = Path(os.path.expanduser(system.get('codex_auth', '~/.codex/auth.json')))
        if not auth.is_file():
            raise FileNotFoundError(f'Codex login not found at {auth}; run `codex login` or set codex_auth')
        link = self.home / 'auth.json'
        if not link.exists():
            link.symlink_to(auth)
        config = (f'model = "{system["model"]}"\nmodel_provider = "openai"\n'
                  f'model_reasoning_effort = "{system["effort"]}"\nweb_search = "disabled"\nproject_doc_max_bytes = 0\n'
                  '[features]\n' + ''.join(f'{name} = false\n' for name in DISABLED_FEATURES) +
                  '[analytics]\nenabled = false\n')
        (self.home / 'config.toml').write_text(config)
        self.cwd = self.folder / 'cwd'  # empty: the agent reaches its workspace only through m1_files
        self.cwd.mkdir(exist_ok=True)
        self.log = (self.folder / 'events.jsonl').open('a')
        env = dict(os.environ, CODEX_HOME=str(self.home), RUST_LOG='error')
        self.proc = subprocess.Popen([system.get('codex_binary', 'codex'), 'app-server', '--stdio'], cwd=self.cwd,
                                     env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=(self.folder / 'codex-stderr.log').open('a'), text=True, bufsize=1)
        self.inbox, self.replies, self.serial = queue.Queue(), {}, 0
        threading.Thread(target=self._read, daemon=True).start()
        self.request('initialize', dict(clientInfo=dict(name='physrsi-eval', version='1.0'),
                                        capabilities=dict(experimentalApi=True)))
        self._send(dict(method='initialized'))
        self.thread_id = None

    def _read(self):
        for line in self.proc.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.log.write(json.dumps(_redact(message), ensure_ascii=False) + '\n')
            self.log.flush()
            if 'id' in message and 'method' not in message:
                self.replies[message['id']] = message
            else:
                self.inbox.put(message)
        self.inbox.put(dict(method='transport/closed'))

    def _send(self, message):
        self.proc.stdin.write(json.dumps(message) + '\n')
        self.proc.stdin.flush()

    def request(self, method, params, timeout=600):
        self.serial += 1
        ident = f'host-{self.serial}'
        self._send(dict(id=ident, method=method, params=params))
        deadline = time.monotonic() + timeout
        while ident not in self.replies:
            if time.monotonic() > deadline or self.proc.poll() is not None:
                raise RuntimeError(f'no reply to {method}')
            time.sleep(.01)
        reply = self.replies.pop(ident)
        if 'error' in reply:
            raise RuntimeError(f'{method} failed: {reply["error"]}')
        return reply['result']

    def start_thread(self, tools):
        result = self.request('thread/start', dict(
            model=self.system['model'], modelProvider='openai', cwd=str(self.cwd), approvalPolicy='never',
            sandbox='read-only', baseInstructions='', developerInstructions='', dynamicTools=tools,
            config=dict(model_reasoning_effort=self.system['effort'], web_search='disabled', project_doc_max_bytes=0)))
        if (result.get('model'), result.get('reasoningEffort')) != (self.system['model'], self.system['effort']):
            raise RuntimeError('model/effort differs from the requested system')
        self.thread_id = result['thread']['id']
        write_json(self.folder / 'thread-start.json', result)
        return self.thread_id

    def run_turn(self, text, host, meter, revision_window):
        """One turn, serving item/tool/call with host.call. Returns 'completed' | 'interrupted' | 'failed'.

        At the reasoning limit the episode is graded at once and the next tool reply says so.
        Once physical actions are closed, the turn may continue revising for `revision_window`
        seconds; after that it is interrupted.
        """
        turn = self.request('turn/start', dict(threadId=self.thread_id, model=self.system['model'],
                                               effort=self.system['effort'], input=[dict(type='text', text=text)]))
        turn_id = turn['turn']['id']
        meter.turn(True)
        interrupted = False
        while True:
            try:
                message = self.inbox.get(timeout=5)
            except queue.Empty:
                message = None
            if not host.closed and meter.at_limit:
                host.close_at_limit()
            if host.closed and not interrupted and time.monotonic() > host.closed_at + revision_window:
                self.request('turn/interrupt', dict(threadId=self.thread_id, turnId=turn_id))
                interrupted = True
            if message is None:
                continue
            meter.notice(message)
            method, params = message.get('method'), message.get('params', {})
            if method == 'transport/closed':
                return 'failed'
            if method == 'item/tool/call' and 'id' in message:
                args = params.get('arguments')
                args = json.loads(args) if isinstance(args, str) else args
                meter.tool(True)
                try:
                    result = host.call(params.get('tool'), args)
                finally:
                    meter.tool(False)
                self._send(dict(id=message['id'], result=result))
            elif 'id' in message and method:  # any other server request (approvals): refuse
                self._send(dict(id=message['id'], error=dict(code=-32601, message='not available')))
            elif method == 'turn/completed' and params.get('turn', {}).get('id') == turn_id:
                status = params['turn'].get('status')
                return 'interrupted' if interrupted else ('completed' if status == 'completed' else 'failed')

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()


def _redact(message):
    """Keep event logs small: drop inline image payloads."""
    text = json.dumps(message)
    if 'data:image' not in text:
        return message
    def strip(value):
        if isinstance(value, dict):
            return {k: ('<image>' if isinstance(v, str) and v.startswith('data:image') else strip(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value
    return strip(message)
