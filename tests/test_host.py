"""Host-side checks without Isaac or a model: a fake simulator behind the real RPC server.

    python -m pytest tests/        (or: python tests/test_host.py)
"""
import json
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physrsi_eval import protocol as P  # noqa: E402
from physrsi_eval.loop import commit, free_port, load_task, prepare_workspace  # noqa: E402
from physrsi_eval.sim import Episode  # noqa: E402
from physrsi_eval.sim_server import serve  # noqa: E402
from physrsi_eval.tools import ToolHost  # noqa: E402
from physrsi_eval.transfer import build_seed  # noqa: E402


class FakeSession:
    """Moves each EEF 1 cm toward its target per step; succeeds once the left gripper closes at z < 0.1."""

    def __init__(self, task, horizon, instruction):
        self.task, self.instruction = task, instruction
        self.metadata = dict(task=task, control_dt=.04, max_episode_steps=horizon)
        self.finished, self.step_id, self.success = False, 0, False
        self.pos = np.array([[.2, .3, .3], [.2, -.3, .3]])
        self.quat = np.array([[1., 0, 0, 0], [1., 0, 0, 0]])
        self.grip = np.ones(2)

    def obs(self):
        images = {c: np.full((480, 640, 3), self.step_id % 255, np.uint8) for c in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')}
        states = np.zeros(14, np.float32)
        states[6], states[13] = self.grip
        return dict(images=images, states=states, eef_positions=self.pos.astype(np.float32),
                    eef_quaternions_wxyz=self.quat.astype(np.float32), instruction=self.instruction,
                    step_id=self.step_id, remaining_steps=self.metadata['max_episode_steps'] - self.step_id)

    def outcome(self, reason):
        return dict(reason=reason, valid=True, native_success=self.success, native_score=1.0 if self.success else .2,
                    native_control_steps=self.step_id)

    def dispatch(self, op, args):
        if op == 'metadata':
            return self.metadata
        if op == 'reset':
            return dict(episode_id='fake', step_id=0, instruction=self.instruction, **self.metadata)
        if op == 'observe':
            return self.obs()
        if op == 'act':
            for _ in range(args['steps']):
                for i, arm in enumerate(('left', 'right')):
                    delta = np.asarray(args['targets'][arm]['position']) - self.pos[i]
                    self.pos[i] += delta * min(1, .01 / max(np.linalg.norm(delta), 1e-9))
                    self.grip[i] = 0. if args['targets'][arm]['gripper_closed'] else 1.
                self.step_id += 1
                self.success = bool(self.grip[0] == 0 and self.pos[0, 2] < .1)
                if self.success or self.step_id >= self.metadata['max_episode_steps']:
                    self.finished = True
                    return dict(step_id=self.step_id, executed_steps=1, terminated=self.success,
                                truncated=not self.success, success=self.success)
            return dict(step_id=self.step_id, executed_steps=args['steps'], terminated=False, truncated=False, success=False)
        if op == 'finish':
            self.finished = True
            return self.outcome(args['reason'])
        raise ValueError(op)


def fake_episode(tmp, task, horizon=300):
    port = free_port()
    session = FakeSession(task['task_id'], horizon, task['native_instruction'])
    thread = threading.Thread(target=serve, args=(session, port), daemon=True)
    thread.start()
    import time
    time.sleep(.3)
    return Episode(task, port, tmp / 'sim', tmp / 'client')


def packet_of(content):
    return json.loads(content[0]['text'])


def hold_action(packet, dz=0.0, close=False, steps=5):
    target = {}
    for arm in ('left', 'right'):
        eef = packet['current_eef'][arm]
        p = list(eef['position'])
        if arm == 'left':
            p[2] += dz
        target[arm] = dict(position=p, quaternion_wxyz=eef['quaternion_wxyz'], gripper_closed=close and arm == 'left')
    return dict(request_id=packet['request_id'], mode='eef', reason='test', steps=steps, target=target)


def test_round_trip(tmp_path=None):
    tmp = Path(tmp_path or Path(__file__).parent / '_tmp')
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    task = load_task('insert_key')
    chain = tmp / 'chain'

    # K1: workspace tools, a rejected action, a program, then the agent stops early -> actor_completed
    k1 = chain / 'K01'
    prepare_workspace(k1 / 'workspace')
    host = ToolHost('Harness', task, fake_episode(k1, task), 1, workspace=k1 / 'workspace', chain=chain,
                    program_cfg=dict(sandbox='none', python=sys.executable))
    contract = json.loads(host.call('robodojo_contract', {})['contentItems'][0]['text'])
    assert contract['max_translation_m'] == .05 and contract['prior_harness_outcomes'] == []
    start = host.call('robodojo_start', {})
    packet = packet_of(start['contentItems'])
    assert packet['instruction'] == task['public_instruction'] and len(start['contentItems']) == 7
    too_far = hold_action(packet, dz=-.2)
    assert host.call('robodojo_act', too_far)['success'] is False
    assert host.call('m1_files', dict(operation='write', path='tools/mcp_client.py', text='x'))['success'] is False
    program = ("import sys,json;sys.path.insert(0,'tools');from mcp_client import call\n"
               "r=call('robodojo_act',json.loads(open('memory/a.json').read()))\n")
    host.call('m1_files', dict(operation='write', path='tools/p.py', text=program))
    act = hold_action(packet, dz=-.04)
    host.call('m1_files', dict(operation='write', path='memory/a.json', text=json.dumps(act)))
    reply = host.call('m1_program', dict(path='tools/p.py'))
    assert reply['success'] and json.loads(reply['contentItems'][0]['text'])['exit_code'] == 0
    assert len(host.episode.history) == 1
    receipt = host.close('actor_completed')
    assert receipt['termination_reason'] == 'actor_completed' and not receipt['native_success']
    after = host.call('robodojo_act', act)
    assert after['success'] is False and 'terminal_feedback' in after['contentItems'][-1]['text']
    listing = json.loads(host.call('m1_development', dict(path=''))['contentItems'][0]['text'])
    assert 'H1/result.json' in listing and not any(n.startswith('H1/workspace/') for n in listing)
    assert P.summary_errors(host.workspace, None, 1)
    host.call('m1_files', dict(operation='write', path='memory/improvements/K1.json',
                               text=json.dumps(dict(k=1, summary='went down 4 cm'))))
    assert P.summary_errors(host.workspace, None, 1) == []
    commit(k1)
    assert not (k1 / 'snapshot' / 'tools' / 'mcp_client.py').exists()

    # K2: prior outcome visible, native success ends the episode inside robodojo_act
    k2 = chain / 'K02'
    prepare_workspace(k2 / 'workspace', k1 / 'snapshot')
    host = ToolHost('Harness', task, fake_episode(k2, task), 2, workspace=k2 / 'workspace', chain=chain)
    packet = packet_of(host.call('robodojo_start', {})['contentItems'])
    assert packet['prior_harness_outcomes'][0]['logical_k'] == 1
    for _ in range(8):
        reply = host.call('robodojo_act', hold_action(packet, dz=-.05, close=True))
        packet = packet_of(reply['contentItems'])
        if host.closed:
            break
    assert host.receipt['native_success'] and host.receipt['termination_reason'] == 'native_terminal'
    assert 'terminal_feedback' in reply['contentItems'][-1]['text']
    listing = json.loads(host.call('m1_development', dict(path=''))['contentItems'][0]['text'])
    assert 'H1/workspace/memory/improvements/K1.json' in listing and 'H2/history.json' in listing
    image = host.call('m1_development', dict(path='H1/observations/000/cam_high.png'))
    assert image['contentItems'][0]['imageUrl'].startswith('data:image/jpeg;base64,')
    host.call('m1_files', dict(operation='write', path='memory/improvements/K2.json', text=json.dumps(dict(k=2, summary='ok'))))
    assert P.summary_errors(host.workspace, k1 / 'snapshot', 2) == []
    commit(k2)

    # transfer seed: library is read-only, target excluded
    seed = build_seed('plug_in_charger', {'insert_key': chain}, tmp / 'seed')
    assert (seed / 'memory/library/insert_key/workspace/memory/improvements/K2.json').is_file()
    k1t = tmp / 'transfer' / 'K01'
    prepare_workspace(k1t / 'workspace', seed)
    target = load_task('plug_in_charger')
    host = ToolHost('Harness', target, fake_episode(k1t, target, 400), 1, workspace=k1t / 'workspace', chain=tmp / 'transfer')
    refused = host.call('m1_files', dict(operation='write', path='memory/library/insert_key/x.md', text='x'))
    assert refused['success'] is False
    read = host.call('m1_files', dict(operation='read', path='memory/library/README.md', text=None))
    assert 'read-only' in read['contentItems'][0]['text']
    host.close('actor_completed')
    shutil.rmtree(tmp, ignore_errors=True)
    print('ok')


if __name__ == '__main__':
    test_round_trip()
