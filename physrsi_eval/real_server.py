"""Serve a site robot driver through the SAME RPC used by K0/K1..K17.

python -m physrsi_eval.real_server --driver site_robot:make_driver --config /lizekai/real/driver.json
A driver implements metadata, reset(layout_id), observe(), act(targets, steps), finish(reason).
The driver owns E-stop, force/collision limits and the measured task-success judge. See docs/plug_in_charger/REAL_TODO.md.
No simulator or model is imported/started here. Every ambiguous mutation locks this server until inspected.
"""
import argparse
import fcntl
import importlib
import json
import socket
import uuid
from pathlib import Path

import numpy as np

from .backends import validate_real_profile
from .rpc import VERSION, receive_packet, send_packet
from .sim import ARMS, CAMERAS, validate_action
from .util import write_json


class RealSession:
    def __init__(self, driver, state_dir):
        self.driver = driver
        self.root = Path(state_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / 'owner.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.metadata = dict(driver.metadata)
        validate_real_profile(self.metadata)
        if self.metadata.get('backend') != 'real' or self.metadata.get('task') != 'plug_in_charger':
            raise ValueError('real plug_in_charger metadata required')
        self.active = False
        self.poisoned = (self.root / 'pending.json').exists() or (self.root / 'active.json').exists()
        self.step = 0
        self.terminal = False

    def mutate(self, op, **args):
        if self.poisoned:
            raise RuntimeError('UNKNOWN action/reset: inspect pending.json and hardware ACKs; never replay')
        self.poisoned = True
        pending = self.root / 'pending.json'
        call_id = uuid.uuid4().hex
        write_json(pending, dict(id=call_id, op=op, args=args))
        result = getattr(self.driver, op)(**args)
        # The ACK is retained BEFORE permitting another command.
        write_json(self.root / 'acks' / (call_id + '.json'), dict(op=op, result=result))
        pending.unlink()
        self.poisoned = False
        return result

    def observation(self):
        obs = self.driver.observe()
        states = np.asarray(obs['states'], float)
        positions = np.asarray(obs['eef_positions'], float)
        quats = np.asarray(obs['eef_quaternions_wxyz'], float)
        if (states.shape != (14,) or positions.shape != (2, 3) or quats.shape != (2, 4)
                or not all(np.isfinite(a).all() for a in (states, positions, quats))
                or not np.allclose(np.linalg.norm(quats, axis=1), 1, atol=1e-4)):
            raise ValueError('driver observation must use normalized dual-arm 14D/link6 contract')
        if obs['step_id'] != self.step or obs['remaining_steps'] != self.metadata['max_episode_steps'] - self.step:
            raise ValueError('driver step accounting mismatch')
        images = {}
        for name in CAMERAS:
            image = np.asarray(obs['images'][name])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f'{name} requires uint8 RGB')
            images[name] = image
        # Only public channels are forwarded. Driver truth/judge state never enters the actor packet.
        self.obs = dict(states=states, eef_positions=positions, eef_quaternions_wxyz=quats, images=images,
                        step_id=self.step, remaining_steps=obs['remaining_steps'])
        return self.obs

    def dispatch(self, op, args):
        if op == 'metadata':
            return self.metadata
        if self.poisoned:
            raise RuntimeError('UNKNOWN state; no further physical commands')
        if op == 'reset':
            if self.active:
                raise ValueError('episode already active')
            write_json(self.root / 'active.json', dict(op='reset', layout_id=args.get('layout_id')))
            result = self.mutate('reset', **args)
            self.active, self.step, self.terminal = True, 0, False
            self.observation()
            return result
        if not self.active:
            raise ValueError('no active episode')
        if op == 'observe':
            return self.observation()
        if op == 'act':
            if self.terminal:
                raise ValueError('native terminal reached')
            obs = self.observation()
            args = dict(args)
            executor = args.pop('executor', False)
            if executor and self.metadata.get('executor_targets') != 'bounded_dls_v1':
                raise ValueError('site has not enabled validated bounded-DLS executor targets')
            targets = args['targets']
            clean = {a: {k: v for k, v in targets[a].items() if k != 'gripper_opening'} for a in ARMS}
            for a in ARMS:
                opening = targets[a].get('gripper_opening')
                if opening is not None and (type(opening) not in (float, int) or not np.isfinite(opening) or not 0 <= opening <= 1):
                    raise ValueError('invalid opening')
            current = {a: dict(position=obs['eef_positions'][i], quaternion_wxyz=obs['eef_quaternions_wxyz'][i]) for i, a in enumerate(ARMS)}
            validate_action(dict(request_id='server', mode='eef', reason='validated physical action',
                                 target=clean, steps=args['steps']), {'request_id': 'server'}, current, executor=executor)
            if args['steps'] > self.metadata['max_episode_steps'] - self.step:
                raise ValueError('action exceeds remaining budget')
            ack = self.mutate('act', **args)
            n = ack.get('executed_steps')
            if type(n) is not int or not 0 < n <= args['steps'] or ack.get('step_id') != self.step + n:
                self.poisoned = True
                write_json(self.root / 'pending.json', dict(op='act', reason='malformed ACK', ack=ack))
                raise RuntimeError('malformed driver ACK')
            if any(type(ack.get(k)) is not bool for k in ('terminated', 'truncated', 'success')):
                self.poisoned = True
                write_json(self.root / 'pending.json', dict(op='act', reason='invalid terminal flags'))
                raise RuntimeError('invalid ACK flags')
            self.step += n
            self.terminal = bool(ack['terminated'] or ack['truncated'])
            return ack
        if op == 'finish':
            result = self.mutate('finish', **args)
            if (result.get('valid') is not True or type(result.get('native_success')) is not bool
                    or result.get('native_control_steps') != self.step
                    or type(result.get('native_score')) not in (int, float)
                    or not 0 <= result['native_score'] <= 1):
                self.poisoned = True
                write_json(self.root / 'pending.json', dict(op='finish', reason='invalid judge result'))
                raise RuntimeError('invalid real judge result; cannot count as a completed experiment')
            self.active = False
            (self.root / 'active.json').unlink()
            return result
        raise ValueError('unsupported operation')


def serve(session, port):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', port))
        listener.listen(1)
        print(json.dumps({'event': 'ready', 'backend': 'real', 'port': port}), flush=True)
        while True:
            conn, _ = listener.accept()
            with conn:
                try:
                    while True:
                        req = receive_packet(conn)
                        try:
                            if req.get('version') != VERSION:
                                raise ValueError('RPC version mismatch')
                            result = session.dispatch(req['op'], req.get('args', {}))
                            reply = dict(ok=True, result=result)
                        except Exception as error:
                            reply = dict(ok=False, error=f'{type(error).__name__}: {error}')
                        send_packet(conn, dict(version=VERSION, request_id=req['request_id'], **reply))
                except (EOFError, ConnectionError):
                    if session.active:
                        session.poisoned = True
                        write_json(session.root / 'pending.json', {'reason': 'host disconnected while episode active'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--driver', required=True, help='module:factory (site driver)')
    p.add_argument('--config', required=True)
    p.add_argument('--port', type=int, default=26710)
    p.add_argument('--state-dir', default='/lizekai/RSIws/real/plug_in_charger/service')
    a = p.parse_args()
    module, factory = a.driver.split(':', 1)
    driver = getattr(importlib.import_module(module), factory)(json.loads(Path(a.config).read_text()))
    serve(RealSession(driver, a.state_dir), a.port)


if __name__ == '__main__':
    main()
