"""m1_program: run one of the agent's own Python programs against the supervised robot tools.

The program talks newline-delimited JSON on stdin/stdout (see tools/mcp_client.py):
  request  {"id": ..., "method": "robodojo_act", "params": {...}}
  reply    {"id": ..., "result": [contentItems]}
Every physical action passes the same validation as a direct tool call. The workspace is
read-only while the program runs: with bubblewrap (used when unprivileged namespaces work) the
program also has no network and sees only the workspace (at /workspace) and the system Python;
without it, the program runs in the workspace directory and any file change fails the call.
"""
import json
import os
import select
import shutil
import signal
import subprocess
import time

from . import protocol as P
from .sim import ActionRejected
from .util import tree

ALLOWED = ('robodojo_contract', 'robodojo_start', 'robodojo_act', 'r1_notes')


class ProgramError(RuntimeError):
    """Program stopped by the host; the episode continues unless it reached the native terminal."""


_BWRAP_USABLE = None


def bwrap_usable():
    """bwrap needs unprivileged user namespaces; many containers do not allow them."""
    global _BWRAP_USABLE
    if _BWRAP_USABLE is None:
        _BWRAP_USABLE = bool(shutil.which('bwrap')) and subprocess.run(
            ['bwrap', '--unshare-all', '--ro-bind', '/usr', '/usr', '--symlink', 'usr/bin', '/bin', '/usr/bin/true'],
            capture_output=True).returncode == 0
    return _BWRAP_USABLE


def command(workspace, relative, cfg):
    python = cfg.get('python', '/usr/bin/python3')
    sandbox = cfg.get('sandbox', 'auto')
    if sandbox == 'auto':
        sandbox = 'bwrap' if bwrap_usable() else 'none'
    if sandbox == 'none':
        return [python, '-B', relative], str(workspace)
    argv = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session', '--clearenv',
            '--ro-bind', '/usr', '/usr', '--symlink', 'usr/bin', '/bin', '--symlink', 'usr/lib', '/lib',
            '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp',
            '--ro-bind', str(workspace), '/workspace', '--chdir', '/workspace',
            '--setenv', 'PATH', '/usr/bin', '--setenv', 'HOME', '/tmp']
    if os.path.isdir('/lib64'):
        argv[argv.index('--proc'):argv.index('--proc')] = ['--symlink', 'usr/lib64', '/lib64']
    return argv + ['--', python, '-B', '/workspace/' + relative], '/'


def run_program(host, relative, cfg):
    """Returns (exit_code, native_terminal)."""
    argv, cwd = command(host.workspace, relative, cfg)
    before = tree(host.workspace)
    child = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             start_new_session=True)
    stderr = (host.episode.client / 'program-stderr.log').open('ab')
    buffer, terminal, last = b'', False, time.monotonic()
    try:
        streams = [child.stdout, child.stderr]
        while streams and not terminal:
            if time.monotonic() - last >= P.PROGRAM_SILENCE_S:
                raise ProgramError(json.dumps(dict(error='PROGRAM_STOPPED_NO_PROGRESS', seconds=P.PROGRAM_SILENCE_S,
                                                   episode_continues=True, original_requests_not_replayed=True)))
            for stream in select.select(streams, [], [], .05)[0]:
                block = os.read(stream.fileno(), 65536)
                if not block:
                    streams.remove(stream)
                    continue
                last = time.monotonic()
                if stream is child.stderr:
                    stderr.write(block[:P.READ_LIMIT_BYTES])
                    continue
                buffer += block
                while b'\n' in buffer and not terminal:
                    line, buffer = buffer.split(b'\n', 1)
                    try:
                        request = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        raise ProgramError(json.dumps(dict(
                            error='PROGRAM_STDOUT_INVALID_JSON', stdout=line[:2048].decode('utf-8', 'replace'),
                            instruction='Program stdout must contain only newline-delimited JSON tool requests.')))
                    method = request.get('method') if isinstance(request, dict) else None
                    if method not in ALLOWED:
                        raise ProgramError(json.dumps(dict(error='PROGRAM_METHOD_NOT_SUPERVISED', method=method,
                                                           allowed=list(ALLOWED), original_requests_not_replayed=True)))
                    try:
                        result = host.physical(method, request.get('params', {}))
                    except (ActionRejected, ValueError, KeyError, TypeError) as error:
                        result = [dict(type='inputText', text=str(error))]
                    try:
                        child.stdin.write((json.dumps({'id': request.get('id'), 'result': result}) + '\n').encode())
                        child.stdin.flush()
                    except BrokenPipeError:
                        pass  # reader gone: the reply is lost, the action is never replayed
                    last = time.monotonic()
                    terminal = host.closed
        if not terminal:
            child.wait(timeout=5)
        return child.poll(), terminal
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=2)
        stderr.close()
        if tree(host.workspace) != before:  # only reachable without bwrap
            raise ProgramError('workspace changed during program; use m1_files between programs')
