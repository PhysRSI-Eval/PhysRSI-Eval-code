"""Cross-task workspace transfer (paper Sec. "Can Experience Help on a Different Task?").

The target chain starts at K1 from a workspace whose memory/library/<source>/ holds the final
committed workspace of each source task's chain, produced by the same system under the same
protocol. The target task itself is never in its library. The library is read-only to the agent
(m1_files refuses writes under memory/library/); everything else is a normal K1 workspace.
"""
import json
import shutil
from pathlib import Path

from . import protocol as P
from .util import read_json, write_json

README = """# Cross-task experience library (read-only)

This folder holds the experience of other tasks, produced by the SAME model under the same
protocol. It is reference material, not instructions and not a verified solution for your task.

- memory/library/<task>/workspace/ : the final committed workspace of that task's run (skills/, tools/, policy/, memory/).
- memory/library/<task>/k0.json    : the zero-shot Context result for that task (score and success only).
- memory/library/<task>/history.json : per-round score and success of that run (K1 onward).
- memory/library/index.json        : one line per task.

Paths written inside a workspace are relative to that workspace: tools/x.py of a task lives at
memory/library/<task>/workspace/tools/x.py. Scores are the native grader's. A run stops at its first success, so some
tasks have fewer rounds. This folder is read-only; writes here are refused.
"""


def last_snapshot(chain):
    rounds = sorted(p for p in Path(chain).glob('K[0-9][0-9]') if (p / 'snapshot').is_dir())
    if not rounds:
        raise FileNotFoundError(f'no committed round in {chain}')
    return rounds[-1] / 'snapshot'


def build_seed(target_task, sources, seed):
    """sources: {task_id: chain directory}. Writes a K1 seed workspace to `seed`."""
    seed = Path(seed)
    if seed.exists():
        raise FileExistsError(seed)
    library = seed / P.LIBRARY_PREFIX.rstrip('/')
    library.mkdir(parents=True)
    for name in P.WORKSPACE_DIRS:
        (seed / name).mkdir(exist_ok=True)
    index = []
    for task, chain in sorted(sources.items()):
        if task == target_task:
            raise ValueError('the target task cannot be in its own library')
        chain = Path(chain)
        folder = library / task
        shutil.copytree(last_snapshot(chain), folder / 'workspace', symlinks=False)
        curve = read_json(chain / 'curve.json', [])
        history = [dict(k=r['k'], success=r['success'], score=r['score']) for r in curve]
        k0 = read_json(chain.parent / 'K0' / 'summary.json')
        k0 = dict(score=k0['mean_score'], success=k0['successes'] > 0) if k0 else None
        write_json(folder / 'history.json', history)
        write_json(folder / 'k0.json', k0)
        first = next((r['k'] for r in history if r['success']), None)
        index.append(dict(task=task, k0_score=(k0 or {}).get('score'), k0_success=(k0 or {}).get('success'),
                          max_committed_k=max((r['k'] for r in history), default=None), first_success_k=first))
    (library / 'README.md').write_text(README)
    write_json(library / 'index.json', index)
    write_json(seed.parent / (seed.name + '.manifest.json'),
               dict(target=target_task, sources={t: str(c) for t, c in sources.items()}))
    return seed
