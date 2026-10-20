"""The PhysRSI-Eval loop.

K0      Context: no workspace, prompt "evaluate task", 3 independent episodes, scores averaged.
K1..K17 Harness: one persistent workspace per chain, prompt "optimize task". Each round is one
        judged physical attempt followed by a revision (robot actions closed) and a commit.
        The chain stops at the first native success (H = 17 rounds at most).

Run layout (one folder per round):
  <runs>/<system>/<task>/K0/r<i>/            sim/ client/ agent/
  <runs>/<system>/<task>/<chain>/K<kk>/      workspace/ (live)  snapshot/ (committed)  sim/ client/ agent/
"""
import json
import shutil
import socket
import time
from pathlib import Path

from . import protocol as P
from .agent import CodexAgent, ReasoningMeter
from .sim import Episode, launch, stop
from .tools import ToolHost
from .util import read_json, write_json

REPO = Path(__file__).resolve().parents[1]
MAX_SUMMARY_TURNS = 3


def load_task(task_id):
    return json.loads((REPO / 'tasks' / task_id / 'task.json').read_text())


def load_json(path):
    return json.loads(Path(path).read_text())


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def run_round(cfg, system, task, folder, condition, k, *, chain=None, previous_snapshot=None):
    """One physical attempt (+ revision for Harness). Returns the graded receipt."""
    folder = Path(folder)
    port = free_port()
    process, _ = launch(cfg, task['task_id'], folder / 'sim', port, gpu=cfg.get('gpu', 0),
                        vulkan_index=cfg.get('vulkan_index'))
    agent = None
    try:
        episode = Episode(task, port, folder / 'sim', folder / 'client', layout_id=cfg.get('layout_id', 0))
        workspace = folder / 'workspace' if condition == 'Harness' else None
        host = ToolHost(condition, task, episode, k, workspace=workspace, chain=chain, program_cfg=cfg.get('program'))
        prompt, tools = P.tool_specs(condition)
        agent = CodexAgent(system, folder / 'agent')
        agent.start_thread(tools)
        meter = ReasoningMeter(P.REASONING_LIMIT_S)
        window = system['revision_window_s']
        status = agent.run_turn(prompt, host, meter, window)
        if status == 'failed':  # client/service failure, not an agent decision: the round is void and rerun
            raise RuntimeError('agent turn failed; see agent/events.jsonl')
        revised_in_turn = host.closed and status == 'completed'
        if not host.closed:  # the agent ended its turn before the native end: grade the state as it is
            host.close('actor_completed')
        write_json(folder / 'agent' / 'reasoning-budget.json', meter.snapshot())
        if condition == 'Harness':
            revise(agent, host, meter, window, k, previous_snapshot, revised_in_turn, folder)
        return host.receipt
    finally:
        if agent is not None:
            agent.close()
        stop(process)


def revise(agent, host, meter, window, k, previous_snapshot, revised_in_turn, folder):
    """Revision with robot actions closed, in the same thread, until the per-K summary is valid."""
    turns = []
    if not revised_in_turn and time.monotonic() < host.closed_at + window:
        feedback = host.feedback()
        feedback['revision'] += P.DEVELOPMENT_HINT
        turns.append(agent.run_turn(json.dumps(feedback, ensure_ascii=False), host, meter, window))
    for _ in range(MAX_SUMMARY_TURNS):
        errors = P.summary_errors(host.workspace, previous_snapshot, k)
        if not errors or time.monotonic() >= host.closed_at + window:
            break
        text = 'Revision validation: ' + '; '.join(errors) + P.SUMMARY_CONTINUATION + P.FILE_WRITE_NOTE
        turns.append(agent.run_turn(text, host, meter, window))
    write_json(folder / 'revision.json', dict(revised_in_first_turn=revised_in_turn, extra_turns=turns,
                                              summary_errors=P.summary_errors(host.workspace, previous_snapshot, k)))


def prepare_workspace(workspace, source=None):
    if source is not None:
        shutil.copytree(source, workspace, symlinks=False)
    for name in P.WORKSPACE_DIRS:
        (workspace / name).mkdir(parents=True, exist_ok=True)
    helper = workspace / P.CLIENT_HELPER_PATH
    if helper.exists():
        helper.chmod(0o644)
        helper.unlink()
    helper.write_text(P.CLIENT_HELPER)
    helper.chmod(0o444)


def commit(folder):
    """The revised workspace becomes the next round's starting point (host helper excluded)."""
    snapshot = folder / 'snapshot'
    shutil.copytree(folder / 'workspace', snapshot, symlinks=False,
                    ignore=lambda d, names: ['mcp_client.py'] if Path(d).name == 'tools' and
                    Path(d).parent == folder / 'workspace' else [])
    return snapshot


def run_k0(cfg, system, task, root, repeats=3):
    rows = []
    for i in range(1, repeats + 1):
        folder = Path(root) / 'K0' / f'r{i}'
        receipt = read_json(folder / 'client' / 'receipt.json')
        if receipt is None:
            if folder.exists():
                folder.rename(folder.with_name(f'{folder.name}.void-{int(time.time())}'))
            receipt = run_round(cfg, system, task, folder, 'Context', 0)
        rows.append(receipt)
    summary = dict(task_id=task['task_id'], repeats=rows,
                   mean_score=sum(r['score'] for r in rows) / len(rows),
                   successes=sum(bool(r['native_success']) for r in rows))
    write_json(Path(root) / 'K0' / 'summary.json', summary)
    return summary


def run_chain(cfg, system, task, chain, horizon=17, seed=None):
    """K1..K<horizon>, stopping at the first native success. Resumable: committed rounds are kept,
    an interrupted round is moved aside (it counts as an extra attempt) and rerun."""
    chain = Path(chain)
    chain.mkdir(parents=True, exist_ok=True)
    write_json(chain / 'chain.json', dict(task_id=task['task_id'], system=system['name'], horizon=horizon,
                                          seeded=seed is not None, layout_id=cfg.get('layout_id', 0)))
    rows = []
    for k in range(1, horizon + 1):
        folder = chain / f'K{k:02d}'
        receipt = read_json(folder / 'client' / 'receipt.json')
        if receipt is None or not (folder / 'snapshot').is_dir():
            if folder.exists():
                folder.rename(folder.with_name(f'{folder.name}.void-{int(time.time())}'))
            previous = chain / f'K{k - 1:02d}' / 'snapshot' if k > 1 else None
            prepare_workspace(folder / 'workspace', previous if k > 1 else seed)
            receipt = run_round(cfg, system, task, folder, 'Harness', k, chain=chain,
                                previous_snapshot=previous)
            commit(folder)
        rows.append(dict(k=k, score=receipt['score'], success=receipt['native_success'],
                         reason=receipt['termination_reason'], steps=receipt['native_control_steps']))
        write_json(chain / 'curve.json', rows)
        if receipt['native_success']:
            break
    first = next((r['k'] for r in rows if r['success']), None)
    write_json(chain / 'curve.json', rows)
    write_json(chain / 'summary.json', dict(task_id=task['task_id'], first_success_k=first, rounds=len(rows),
                                            best_score=max(r['score'] for r in rows)))
    return rows
