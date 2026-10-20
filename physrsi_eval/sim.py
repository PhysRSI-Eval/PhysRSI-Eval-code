"""Host side of one physical episode: launch the simulator, validate actions, build observations.

The agent sees exactly one observation packet per tool reply: a JSON state packet
followed by the three camera images (JPEG, quality 90). Every action is checked
against the shared action bounds before it reaches the simulator.
"""
import base64
import io
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

from .rpc import RPCClient
from .util import write_json

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
ARMS = ('left', 'right')
MAX_TRANSLATION_M = 0.05
MAX_ROTATION_RAD = 0.35
REPO = Path(__file__).resolve().parents[1]


class ActionRejected(ValueError):
    """Invalid proposal; nothing was executed."""


def launch(cfg, task, output, port, gpu=0, vulkan_index=None, startup_timeout=1800):
    """Start one Isaac simulator process for `task`; return (process, metadata) once it is ready."""
    output = Path(output)
    cache = output / 'cache'
    for name in ('home', 'tmp', 'cache', 'data', 'config', 'run'):
        (cache / name).mkdir(parents=True, exist_ok=True)
    robodojo = Path(cfg['robodojo_root']).resolve()
    # Kit pulls a few extensions from the online registry on first start; keep them across runs.
    registry = Path(cfg.get('kit_cache_dir') or Path(cfg.get('runs_dir', 'runs')) / '.kit-cache') / 'registry'
    registry.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES=str(gpu),
               PYTHONPATH=os.pathsep.join(map(str, (REPO, robodojo, robodojo / 'XPolicyLab'))),
               PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               OMNI_KIT_ACCEPT_EULA='YES', ACCEPT_EULA='Y', PRIVACY_CONSENT='Y',
               HOME=str(cache / 'home'), TMPDIR=str(cache / 'tmp'), XDG_CACHE_HOME=str(cache / 'cache'),
               XDG_DATA_HOME=str(cache / 'data'), XDG_CONFIG_HOME=str(cache / 'config'),
               XDG_RUNTIME_DIR=str(cache / 'run'))
    if cfg.get('extra_ld_library_path'):
        env['LD_LIBRARY_PATH'] = cfg['extra_ld_library_path'] + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
    kit = (f'--portable-root {cache} --/renderer/multiGpu/enabled=false --/renderer/multiGpu/maxGpuCount=1 '
           f'--/renderer/activeGpu={gpu if vulkan_index is None else vulkan_index} --/physics/cudaDevice=0 '
           '--/renderer/asyncInit=false --/app/updateOrder/checkForHydraRenderComplete=1000 '
           '--/app/renderer/waitIdle=true --/app/hydraEngine/waitIdle=true '
           f'--/exts/omni.kit.registry.nucleus/cachePath={registry.resolve()} '
           f'--/log/file={output}/kit.log --/app/hangDetector/enabled=false')
    for folder in cfg.get('ext_folders') or ():  # optional offline copies of registry extensions
        kit += f' --ext-folder {Path(folder).resolve()}'
    argv = [cfg['robodojo_python'], '-B', '-u', '-m', 'physrsi_eval.sim_server', '--task', task,
            '--output', str(output), '--port', str(port), '--eval-seed', '0',
            '--headless', '--enable_cameras', '--kit_args', kit]
    log = (output / 'sim.log').open('ab')
    process = subprocess.Popen(argv, cwd=robodojo, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'simulator exited during startup (code {process.returncode}); see {output}/sim.log')
        for line in (output / 'sim.log').read_text(errors='replace').splitlines():
            if line.startswith('{"event": "ready"'):
                return process, json.loads(line)['metadata']
        time.sleep(2)
    stop(process)
    raise TimeoutError(f'simulator not ready after {startup_timeout}s; see {output}/sim.log')


def stop(process, timeout=60):
    if process is None or process.poll() is not None:
        return
    import signal
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def jpeg_url(pixels):
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(pixels, np.uint8)).convert('RGB').save(buffer, format='JPEG', quality=90)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()


def _rotation():
    from scipy.spatial.transform import Rotation
    return Rotation


def _has_cjk(text):
    return any('㐀' <= c <= '鿿' or '\U00020000' <= c <= '\U000323af' for c in text)


def validate_action(args, request, current_eef):
    """The shared action bounds: one EEF target per arm within 5 cm / 0.35 rad of the current pose."""
    Rotation = _rotation()
    if not isinstance(args, dict) or args.get('request_id') != request['request_id']:
        raise ActionRejected('Response must match the current observation request_id')
    if not isinstance(args.get('reason'), str) or not args['reason'].strip():
        raise ActionRejected('Brief visible evidence and action purpose are required')
    if _has_cjk(args['reason']):
        raise ActionRejected('Public decision text must be written in English')
    if args.get('mode') != 'eef':
        raise ActionRejected('Only bounded EEF actions are available')
    if type(args.get('steps')) is not int or not 1 <= args['steps'] <= 5:
        raise ActionRejected('Expected 1..5 execution steps')
    target = args.get('target')
    if not isinstance(target, dict) or set(target) != set(ARMS):
        raise ActionRejected('EEF action requires explicit left and right targets')
    for arm in ARMS:
        t = target[arm]
        if not isinstance(t, dict) or set(t) != {'position', 'quaternion_wxyz', 'gripper_closed'}:
            raise ActionRejected('unexpected arm target fields')
        p, q = np.asarray(t['position'], float), np.asarray(t['quaternion_wxyz'], float)
        if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all():
            raise ActionRejected('Finite EEF position/quaternion required')
        if abs(np.linalg.norm(q) - 1) > 1e-4 or type(t['gripper_closed']) is not bool:
            raise ActionRejected('Unit quaternion and explicit boolean gripper required')
        if np.linalg.norm(p - np.asarray(current_eef[arm]['position'])) > MAX_TRANSLATION_M + 1e-9:
            raise ActionRejected('EEF target exceeds 5 cm')
        old = np.asarray(current_eef[arm]['quaternion_wxyz'])
        delta = Rotation.from_quat(q[[1, 2, 3, 0]]) * Rotation.from_quat(old[[1, 2, 3, 0]]).inv()
        if delta.magnitude() > MAX_ROTATION_RAD + 1e-9:
            raise ActionRejected('EEF target exceeds 0.35 rad')


class Episode:
    """One assigned episode: start once, act until the native end, then finish (grade)."""

    def __init__(self, task, port, sim_output, client_dir, layout_id=0):
        self.task, self.port, self.layout_id = task, port, layout_id
        self.sim_output, self.client = Path(sim_output), Path(client_dir)
        self.client.mkdir(parents=True, exist_ok=True)
        self.rpc = None
        self.history = []
        self.request = None
        self.outcome = None
        self.started = False

    @property
    def closed(self):
        return self.outcome is not None

    def start(self):
        if self.started:
            raise ActionRejected('assigned episode already started')
        self.rpc = RPCClient(self.port)
        meta = self.rpc.request('metadata')
        if meta['task'] != self.task['task_id']:
            raise RuntimeError('launched task differs from the assigned task')
        reset = self.rpc.request('reset', layout_id=self.layout_id)
        if reset['instruction'] != self.task['native_instruction']:
            raise RuntimeError('native instruction differs from the pinned task definition; check the RoboDojo version')
        if reset['max_episode_steps'] != self.task['native_horizon_steps']:
            raise RuntimeError('native step limit differs from the pinned task definition')
        self.reset, self.started = reset, True
        write_json(self.client / 'reset.json', reset)
        return self._observation()

    def _current_eef(self, obs):
        Rotation = _rotation()
        eef = {}
        for i, arm in enumerate(ARMS):
            q = np.asarray(obs['eef_quaternions_wxyz'][i], float)
            eef[arm] = dict(position=np.asarray(obs['eef_positions'][i]).tolist(), quaternion_wxyz=q.tolist(),
                            rpy_deg=Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler('xyz', degrees=True).tolist(),
                            frame='environment_origin', link='link6',
                            gripper_opening_command=float(obs['states'][i * 7 + 6]))
        return eef

    def _observation(self, result=None, obs=None):
        """Public packet + images for the agent; also archived as request_<n>.json for later rounds."""
        from PIL import Image
        obs = obs if obs is not None else self.rpc.request('observe')
        decision = len(self.history)
        self.current_eef = self._current_eef(obs)
        finished = result is not None
        packet = dict(episode_id=self.reset['episode_id'], step_id=obs['step_id'], task=self.task['task_id'],
                      max_episode_steps=self.reset['max_episode_steps'],
                      instruction=self.task['public_instruction'], remaining_steps=obs['remaining_steps'],
                      current_state=np.asarray(obs['states']).tolist(), current_eef=self.current_eef,
                      task_context=dict(task_definition=self.task['public_task']), robot_profile='robodojo',
                      request_id=None if finished else uuid.uuid4().hex, rollout_finished=finished,
                      result=result, previous_result=self.history[-1] if self.history else None)
        self.request = packet
        write_json(self.client / f'request_{decision:03d}.json', packet)
        folder = self.client / 'observations' / f'{decision:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        content = [dict(type='inputText', text=json.dumps(packet, ensure_ascii=False))]
        for camera in CAMERAS:
            pixels = obs['images'][camera]
            Image.fromarray(np.asarray(pixels, np.uint8)).save(folder / f'{camera}.png')
            content += [dict(type='inputText', text='camera_id=' + camera),
                        dict(type='inputImage', imageUrl=jpeg_url(pixels))]
        return content

    def last_observation(self):
        """Re-deliver the current observation without stepping (used after programs and notes)."""
        if not self.started:
            return []
        content = [dict(type='inputText', text=json.dumps(self.request, ensure_ascii=False))]
        folder = self.client / 'observations' / f'{len(self.history):03d}'
        from PIL import Image
        for camera in CAMERAS:
            content += [dict(type='inputText', text='camera_id=' + camera),
                        dict(type='inputImage', imageUrl=jpeg_url(np.asarray(Image.open(folder / f'{camera}.png'))))]
        return content

    def act(self, args):
        if not self.started:
            raise ActionRejected('native action before robodojo_start')
        if self.closed:
            raise ActionRejected('native episode is closed')
        validate_action(args, self.request, self.current_eef)
        decision = len(self.history)
        write_json(self.client / f'response_{decision:03d}.json', args)
        start = self.request['step_id']
        ack = self.rpc.request('act', targets=args['target'], steps=args['steps'])
        done = ack['terminated'] or ack['truncated']
        self.history.append(dict(decision=decision, start_tick=start, end_tick=ack['step_id'], response=args,
                                 executed_steps=ack['executed_steps'], terminal=done, native_success=ack['success']))
        write_json(self.client / 'history.json', self.history)
        if not done:
            return self._observation()
        obs = self.rpc.request('observe')  # the final frame, read before the simulator closes
        return self._observation(result=self.finish('native_terminal'), obs=obs)

    def finish(self, reason):
        """Close physical actions and grade. reason: native_terminal | actor_completed | effective_reasoning_time_limit."""
        if self.outcome is None:
            if not self.started:
                self.outcome = dict(reason=reason, valid=False, native_success=False, native_score=0.0,
                                    native_control_steps=0, note='episode never started')
            else:
                self.outcome = self.rpc.request('finish', reason=reason)
                self.rpc.close()
            self.outcome['decisions'] = len(self.history)
            write_json(self.client / 'result.json', self.outcome)
        return self.outcome
