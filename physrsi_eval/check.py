"""Setup verification: pinned versions, pinned task files, Python stack, Codex CLI, GPU.

`python -m physrsi_eval check` runs the static checks; `--smoke <task>` additionally launches the
simulator once, resets the pinned layout, executes one zero-motion action and grades it (no model).
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

from .util import sha256

REPO = Path(__file__).resolve().parents[1]
TASKS = ('stack_blocks_by_language', 'cover_blocks', 'insert_key', 'plug_in_charger')


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True).stdout.strip()


def run_checks(cfg):
    lock = json.loads((REPO / 'tasks' / 'lock.json').read_text())
    root = Path(cfg['robodojo_root'])
    results = []

    def check(name, ok, detail=''):
        results.append((name, bool(ok), detail))

    check('RoboDojo checkout', (root / '.git').exists(), str(root))
    if (root / '.git').exists():
        head = git(root, 'rev-parse', 'HEAD')
        check('RoboDojo commit', head == lock['robodojo']['commit'], head)
        for path, commit in lock['robodojo']['submodules'].items():
            actual = git(root / path, 'rev-parse', 'HEAD')
            check(f'submodule {path}', actual == commit, actual)
    for relative, digest in lock['files_sha256'].items():
        path = root / relative
        check(f'sha256 {relative}', path.is_file() and sha256(path) == digest, 'missing' if not path.is_file() else '')
    for task in TASKS:
        spec = json.loads((REPO / 'tasks' / task / 'task.json').read_text())
        source = (root / spec['native_task_file']['path']).read_text() if (root / spec['native_task_file']['path']).is_file() else ''
        # The instruction itself is checked at every reset (it can be generated per layout).
        check(f'{task} step limit {spec["native_horizon_steps"]}', f'step_lim = {spec["native_horizon_steps"]}' in source)
    for required in ('Robots', 'Object', 'Material', 'Eval_Layout'):
        check(f'assets {required}/', (root / 'Assets' / required).is_dir())
    probe = subprocess.run([cfg['robodojo_python'], '-c',
                            'import isaacsim, isaaclab, torch, msgpack, scipy, PIL, imageio_ffmpeg;'
                            'print(torch.cuda.is_available(), torch.cuda.device_count())'],
                           capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=900,
                           env=dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', ACCEPT_EULA='Y', PRIVACY_CONSENT='Y'))
    check('RoboDojo Python stack (isaacsim, isaaclab, torch, msgpack, ...)', probe.returncode == 0,
          (probe.stdout or probe.stderr).strip().splitlines()[-1:] if (probe.stdout or probe.stderr) else '')
    check('CUDA visible', probe.returncode == 0 and probe.stdout.strip().startswith('True'), probe.stdout.strip())
    codex = shutil.which('codex')
    version = subprocess.run([codex, '--version'], capture_output=True, text=True).stdout.strip() if codex else ''
    check('Codex CLI', bool(codex), version + (f' (paper: {lock["codex"]["version_in_paper"]})' if codex else ''))
    from .program import bwrap_usable
    sandbox = (cfg.get('program') or {}).get('sandbox', 'auto')
    usable = bwrap_usable()
    check('program sandbox', sandbox != 'bwrap' or usable,
          'bwrap' if usable else 'bwrap unavailable here: programs run unsandboxed in the workspace directory')
    return results


def smoke(cfg, task_id, folder):
    """Launch → reset → one zero-motion action → grade. Exercises the full simulator path without a model."""
    from .loop import free_port, load_task
    from .sim import Episode, launch, stop
    import time
    task = load_task(task_id)
    folder = Path(folder) / f'smoke-{task_id}-{int(time.time())}'
    port = free_port()
    process, meta = launch(cfg, task_id, folder / 'sim', port, gpu=cfg.get('gpu', 0), vulkan_index=cfg.get('vulkan_index'))
    try:
        episode = Episode(task, port, folder / 'sim', folder / 'client', layout_id=0)
        packet = json.loads(episode.start()[0]['text'])
        hold = {arm: dict(position=packet['current_eef'][arm]['position'],
                          quaternion_wxyz=packet['current_eef'][arm]['quaternion_wxyz'],
                          gripper_closed=packet['current_eef'][arm]['gripper_opening_command'] < .5) for arm in ('left', 'right')}
        episode.act(dict(request_id=packet['request_id'], mode='eef', reason='smoke test: hold pose', steps=1, target=hold))
        outcome = episode.finish('actor_completed')
    finally:
        stop(process)
    frames = len(list((folder / 'sim' / 'observations').glob('*-cam_high.png')))
    return dict(folder=str(folder), metadata=meta, outcome=outcome, recorded_frames=frames)
