"""Simulator process: one native RoboDojo episode behind a loopback RPC.

Runs inside RoboDojo's Isaac Python (launched by physrsi_eval.sim). The task,
its reward/score and its step limit are RoboDojo's own and are never modified.
Every control step records the three 640x480 camera images under
<output>/observations/ so that episode videos can be rendered afterwards.

Ops: metadata, reset, observe, act, finish.
"""
import argparse
import importlib
import json
import socket
import traceback
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
CAMERA_ALIASES = dict(cam_high=('cam_high', 'cam_head', 'head_camera', 'top_camera'),
                      cam_left_wrist=('cam_left_wrist', 'left_camera'),
                      cam_right_wrist=('cam_right_wrist', 'right_camera'))
ARMS = ('left', 'right')


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)))
    tmp.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


# --- robot-only kinematics: URDF FK and one bounded damped-least-squares step per control tick ---

def _rotation():
    from scipy.spatial.transform import Rotation
    return Rotation


def transform(position, quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, float)
    t = np.eye(4)
    t[:3, :3] = _rotation().from_quat(q[[1, 2, 3, 0]]).as_matrix()
    t[:3, 3] = position
    return t


class ArmFK:
    def __init__(self, urdf, joint_names, base='base_link', tip='link6'):
        Rotation = _rotation()
        self.names = list(joint_names)
        edges = {j.find('child').get('link'): j for j in ET.parse(urdf).getroot().findall('joint')}
        self.chain = []
        while tip != base:
            j = edges[tip]
            origin = j.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            t = np.eye(4)
            t[:3, :3], t[:3, 3] = Rotation.from_euler('xyz', rpy).as_matrix(), xyz
            axis = j.find('axis')
            axis = np.fromstring(axis.get('xyz'), sep=' ') if axis is not None else np.array([1., 0, 0])
            self.chain.insert(0, (j.get('name'), j.get('type'), t, axis))
            tip = j.find('parent').get('link')
        moving = [n for n, kind, _, _ in self.chain if kind != 'fixed']
        require(set(moving) == set(self.names), f'URDF chain/arm joint mismatch: {moving} vs {self.names}')

    def matrix(self, q):
        Rotation = _rotation()
        values, t = dict(zip(self.names, np.asarray(q, float))), np.eye(4)
        for name, kind, origin, axis in self.chain:
            motion = np.eye(4)
            if kind in ('revolute', 'continuous'):
                motion[:3, :3] = Rotation.from_rotvec(axis * values[name]).as_matrix()
            t = t @ origin @ motion
        return t

    def bounded_step(self, q, limits, root, target):
        """One DLS step toward the target: task error capped at 2 cm / 0.1 rad, joints at 0.05 rad."""
        Rotation = _rotation()
        q = np.asarray(q, float)
        now = root @ self.matrix(q)
        goal = transform(target['position'], target['quaternion_wxyz'])
        bounded = lambda x, cap: x * min(1., cap / max(np.linalg.norm(x), 1e-12))
        error = np.r_[bounded(goal[:3, 3] - now[:3, 3], .02),
                      bounded(Rotation.from_matrix(goal[:3, :3] @ now[:3, :3].T).as_rotvec(), .1)]
        jacobian = np.zeros((6, len(q)))
        for i in range(len(q)):
            shifted = q.copy()
            shifted[i] += 1e-5
            nxt = root @ self.matrix(shifted)
            jacobian[:3, i] = (nxt[:3, 3] - now[:3, 3]) / 1e-5
            jacobian[3:, i] = Rotation.from_matrix(nxt[:3, :3] @ now[:3, :3].T).as_rotvec() / 1e-5
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .05 ** 2 * np.eye(6), error)
        limits = np.asarray(limits, float)
        low, high = np.maximum(limits[:, 0], q - .05), np.minimum(limits[:, 1], q + .05)
        require(np.all(low <= high), 'joint outside its valid interval')
        return np.clip(q + delta, low, high).astype(np.float32)


class DualKinematics:
    def __init__(self, env):
        self.manager = env.robot_manager
        self.robots = {r.arm_name.split('_')[0]: r for r in self.manager.robot_list if r.type == 'target'}
        require(set(self.robots) == set(ARMS), 'expected left/right target ARX X5 arms')
        self.fk = {arm: ArmFK(r.urdf_path, r.arm_joints_name, r.base_link, r.ee_link_name)
                   for arm, r in self.robots.items()}

    def root(self, arm):
        robot = self.robots[arm]
        value = self.manager.get_link_pose(robot, robot.base_link, is_relative=True)[0]
        return transform(value[:3], value[3:])

    def check(self):
        """FK must reproduce the measured link6 pose (2 mm / 0.01 rad), else the frame is wrong."""
        Rotation = _rotation()
        for arm, robot in self.robots.items():
            predicted = self.root(arm) @ self.fk[arm].matrix(self.manager.get_joint(robot)[0])
            measured = self.manager.get_real_endpose(robot)[0]
            error = np.linalg.norm(predicted[:3, 3] - measured[:3])
            angle = Rotation.from_matrix(predicted[:3, :3] @ transform(measured[:3], measured[3:])[:3, :3].T).magnitude()
            require(error < .002 and angle < .01, f'robot FK does not match measured link6 ({arm})')

    def action(self, targets):
        """14D absolute joint command for one control step toward the left/right EEF targets."""
        action = []
        for arm in ARMS:
            robot, target = self.robots[arm], targets[arm]
            key = self.manager.robot_key[self.manager.robot_list.index(robot)]
            limits = key.data.soft_joint_pos_limits[0, robot.arm_joint_indices].cpu().numpy()
            joints = self.fk[arm].bounded_step(self.manager.get_joint(robot)[0], limits, self.root(arm), target)
            action.extend([*joints.tolist(), float(not target['gripper_closed'])])
        return np.asarray(action, np.float32)


# --- one native episode ---

class Session:
    def __init__(self, env, output, task):
        self.env, self.output, self.task = env, Path(output), task
        self.episode_id, self.step_id = None, 0
        self.success = self.terminated = self.truncated = False
        self.finished, self.poisoned = False, True
        self.obs = None
        self.metadata = dict(task=task, control_dt=1 / env.obs_manager.collect_freq,
                             max_episode_steps=int(env.step_lim), cameras=list(CAMERAS),
                             frame='environment_origin', eef_link='link6')
        env._stream_vision = lambda *args, **kwargs: None  # observations are recorded here, once per step

    def _observe(self):
        from PIL import Image
        raw = self.env.get_obs()
        state = raw['state']
        states = np.concatenate([np.r_[state[f'{arm}_arm_joint_state'], state[f'{arm}_ee_joint_state']]
                                 for arm in ARMS]).astype(np.float32)
        require(states.shape == (14,) and np.isfinite(states).all(), 'invalid dual-arm state')
        images = {}
        for key, names in CAMERA_ALIASES.items():
            source = next((n for n in names if n in raw['vision']), None)
            require(source is not None, f'missing camera {key}')
            images[key] = np.asarray(raw['vision'][source]['color'], np.uint8)[..., :3]
            require(images[key].shape == (480, 640, 3), 'camera resolution changed')
        poses = np.asarray([state[f'{arm}_ee_pose'] for arm in ARMS], np.float32)
        self.obs = dict(images=images, states=states, eef_positions=poses[:, :3], eef_quaternions_wxyz=poses[:, 3:],
                        instruction=raw['instruction'], step_id=self.step_id,
                        remaining_steps=max(0, self.metadata['max_episode_steps'] - self.step_id))
        folder = self.output / 'observations'
        folder.mkdir(exist_ok=True)
        for camera in CAMERAS:
            Image.fromarray(images[camera]).save(folder / f'{self.step_id:06d}-{camera}.png')
        with (self.output / 'states.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step_id=self.step_id, states=states.tolist(),
                                         eef_positions=poses[:, :3].tolist(),
                                         eef_quaternions_wxyz=poses[:, 3:].tolist())) + '\n')
        return self.obs

    def reset(self, layout_id):
        require(self.episode_id is None, 'one episode per simulator process; no reset retry')
        self.env.reset(seed=[layout_id])
        # Same preamble as RoboDojo's EvalEnv.run_eval: register the task's completion checks.
        self.env.run_reward()
        if hasattr(self.env, 'get_score'):
            self.env.get_score()
        groups = ('check_list', 'final_check_list', 'trigger_check_list')
        require(sum(len(getattr(self.env.reward_manager, g)[0]) for g in groups) > 0,
                'native completion conditions are empty; refusing vacuous success')
        if getattr(self.env, 'interact', False) and hasattr(self.env, 'query_support_arm_traj'):
            for env_idx in self.env.get_running_env_idx_list():
                self.env.query_support_arm_traj(env_idx=env_idx)
        self.kinematics = DualKinematics(self.env)
        self.kinematics.check()
        self.episode_id, self.poisoned = uuid.uuid4().hex, False
        self._observe()
        result = dict(episode_id=self.episode_id, step_id=0, layout_id=layout_id, instruction=self.obs['instruction'],
                      **self.metadata)
        write_json(self.output / 'reset.json', result)
        return result

    def act(self, targets, steps):
        """Up to `steps` control ticks toward fixed EEF targets; stops early at the native end."""
        require(not (self.finished or self.terminated or self.truncated), 'episode finished')
        require(type(steps) is int and 1 <= steps <= 5, 'steps must be 1..5')
        executed = 0
        for _ in range(steps):
            action = self.kinematics.action(targets)
            command = {}
            for arm, offset in (('left', 0), ('right', 7)):
                command[f'{arm}_arm_joint_state'] = action[offset:offset + 6]
                command[f'{arm}_ee_joint_state'] = action[offset + 6:offset + 7]
            before = int(self.env.take_action_cnt[0])
            self.env.take_action(command)
            require(int(self.env.take_action_cnt[0]) == before + 1, 'native action not executed exactly once')
            self.step_id += 1
            executed += 1
            ended = bool(self.env.end_flag[0])
            self.success = bool(ended and self.env.success[0])
            self.truncated = bool(ended and not self.success and self.step_id >= self.env.step_lim)
            self.terminated = bool(ended and not self.truncated)
            self._observe()
            if ended:
                self.finish('native_terminal')
                break
        return dict(step_id=self.step_id, executed_steps=executed, terminated=self.terminated,
                    truncated=self.truncated, success=self.success)

    def observe(self):
        return dict(self.obs, images={k: v for k, v in self.obs['images'].items()})

    def finish(self, reason):
        """Grade the current native state without stepping: success, else the task's own partial score."""
        if self.finished:
            return json.loads((self.output / 'evaluation_outcome.json').read_text())
        self.finished = True
        valid = self.episode_id is not None and not self.poisoned and 0 not in getattr(self.env, 'unstable_envs', set())
        has_score = hasattr(self.env, 'get_score')
        score = None
        if valid:
            score = 1.0 if self.success else (float(self.env.reward_manager.get_score()[0]) / 100 if has_score else 0.0)
        outcome = dict(task=self.task, episode_id=self.episode_id, reason=reason, valid=valid,
                       native_success=bool(self.success) if valid else None, native_score=score,
                       has_native_partial_score=has_score, native_control_steps=self.step_id,
                       native_step_limit=self.metadata['max_episode_steps'],
                       native_horizon_reached=bool(self.terminated or self.truncated))
        write_json(self.output / 'evaluation_outcome.json', outcome)
        return outcome

    def dispatch(self, op, args):
        if op == 'metadata':
            return self.metadata
        if op == 'reset':
            return self.reset(**args)
        require(self.episode_id is not None and not self.poisoned, 'no valid episode')
        if op == 'observe':
            return self.observe()
        if op == 'act':
            return self.act(**args)
        if op == 'finish':
            return self.finish(**args)
        raise ValueError('unsupported op: ' + op)


def serve(session, port):
    from physrsi_eval.rpc import VERSION, receive_packet, send_packet
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', port))
        listener.listen(1)
        print(json.dumps(dict(event='ready', port=port, metadata=session.metadata)), flush=True)
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(None)  # model thinking time is not a simulator deadline
            try:
                while True:
                    request = receive_packet(connection)
                    reply = dict(version=VERSION, request_id=request.get('request_id'))
                    try:
                        reply.update(ok=True, result=session.dispatch(request['op'], request.get('args', {})))
                    except Exception as error:
                        if request.get('op') in ('reset', 'act'):
                            session.poisoned = True  # unknown physical state: never continue this episode
                        traceback.print_exc()
                        reply.update(ok=False, error=f'{type(error).__name__}: {error}')
                    send_packet(connection, reply)
                    if session.finished and request.get('op') == 'finish':
                        return
            except (EOFError, ConnectionError):
                session.finish('host_disconnect')


class _NoPolicy:
    """RoboDojo's eval env expects a policy websocket; the agent acts through RPC instead."""
    def __init__(self, **kwargs):
        pass

    def call(self, func_name, **kwargs):
        if func_name != 'reset':
            raise RuntimeError('native autonomous policy loop is disabled')

    def close(self):
        pass


def main():
    import cv2  # noqa: F401  (load before Isaac's extension paths, as RoboDojo's main does)
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--eval-seed', type=int, default=0, help='published layout group (Eval_Layout/<arm>/<eval_seed>)')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless, args.enable_cameras = True, True
    args.output.mkdir(parents=True, exist_ok=True)

    import yaml
    from env.global_configs import ROOT_DIR, BENCHMARK, ENV_CONFIG_PATH
    registry = importlib.import_module(f'task.{BENCHMARK}.task_registry')
    config_dir = str(Path(ROOT_DIR) / 'task' / BENCHMARK / 'config')
    task_values = yaml.safe_load(open(registry.task_config_path(config_dir, args.task))) or {}
    monitor = bool(task_values.get('Articulation'))
    if monitor:  # same as RoboDojo main.py: articulation tasks start the PhysX warning monitor first
        from src.eval_client.physx_warning_monitor import get_monitor
        get_monitor().start(enabled=True)
    app = AppLauncher(args).app
    env = session = None
    try:
        from omegaconf import OmegaConf
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization
        from src.eval_client import eval_env
        root = Path(ENV_CONFIG_PATH)
        evaluation = load_yaml(str(root / 'arx_x5.yml'))
        evaluation.update(task_name=args.task, num_envs=1, device_id=0, eval_batch=False, policy_name='Pi_05',
                          additional_info='physrsi', seed=args.eval_seed, physx_monitor_enabled=monitor)
        values = {key: load_yaml(str(root / key / (evaluation['config'][key] + '.yml')))
                  for key in ('sim', 'scene', 'camera', 'robot')}
        values.update(eval_cfg=evaluation, deploy_cfg=dict(port=1, policy_name='Pi_05'),
                      task_env=load_yaml(registry.task_config_path(config_dir, args.task)))
        cfg = OmegaConf.create(values)
        cfg.sim.scene.num_envs = 1
        cfg = process_randomization(cfg)
        cfg, _ = process_config(cfg, task_name=args.task)
        cfg.eval_cfg.eval_num = 1
        cfg.camera.default_frequency = cfg.eval_cfg.observation.collect_freq
        cfg.sim.seed = [0]  # startup only; reset() selects the layout
        for robot in cfg.robot.robots:
            robot.need_planner = False  # EEF actions use the robot-only DLS above, never cuRobo
        original = eval_env.WsModelClient
        try:
            eval_env.WsModelClient = _NoPolicy
            env = eval_env.create_eval_env(cfg, app)
        finally:
            eval_env.WsModelClient = original
        session = Session(env, args.output, args.task)
        serve(session, args.port)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if session is not None and session.episode_id is not None:
            session.finish('server_close')
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
