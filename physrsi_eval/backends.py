"""One episode interface for Isaac simulation and a site-owned real robot RPC service."""
from contextlib import contextmanager
from pathlib import Path

from .rpc import RPCClient
from .sim import Episode, ActionRejected, launch, stop
from .util import write_json


class RealEpisode(Episode):
    def start(self):
        if self.started or self.start_attempted:
            raise ActionRejected('assigned episode start already attempted; never retry ambiguous reset')
        self.start_attempted = True
        self.rpc = RPCClient(self.port, host=self.rpc_host)
        meta = self.rpc.request('metadata')
        required = dict(backend='real', task=self.task['task_id'], max_episode_steps=self.task['native_horizon_steps'],
                        control_hz=self.robot_profile['control_hz'], frame=self.robot_profile['frame'],
                        eef_link=self.robot_profile['eef_link'], robot=self.robot_profile['robot'],
                        calibration_id=self.robot_profile['calibration_id'])
        for key, value in required.items():
            if meta.get(key) != value:
                self.rpc.close()
                raise RuntimeError(f'real service {key} differs from the deployment profile')
        write_json(self.client / 'metadata.json', meta)
        # A site driver acquires a manually prepared scene. reset must NOT replay a simulated reset trajectory.
        pending = self.client / 'pending-reset.json'
        write_json(pending, {'layout_id': self.layout_id, 'metadata': meta})
        reset = self.rpc.request('reset', layout_id=self.layout_id)
        if (reset.get('instruction') != self.task['native_instruction']
                or reset.get('max_episode_steps') != self.task['native_horizon_steps']
                or not reset.get('episode_id')):
            raise RuntimeError('real reset identity/instruction/budget mismatch')
        self.reset, self.started = reset, True
        write_json(self.client / 'reset.json', reset)
        pending.unlink()
        return self._observation()


def validate_real_profile(profile):
    for key in ('robot', 'frame', 'eef_link', 'control_hz', 'calibration_id', 'calibration_source'):
        if not profile.get(key):
            raise ValueError(f'real profile requires {key}')
    if profile['calibration_source'] != 'real_measured':
        raise ValueError('real calibration must be measured on the target robot, not copied from simulation')
    if profile['control_hz'] != 25:
        raise ValueError('this matched protocol requires a 25 Hz adapter (1..5 ticks per action)')


@contextmanager
def open_episode(cfg, task, folder, port):
    folder = Path(folder)
    mode = cfg.get('backend', 'sim')
    process = None
    if mode == 'sim':
        process, _ = launch(cfg, task['task_id'], folder / 'sim', port, gpu=cfg.get('gpu', 0),
                            vulkan_index=cfg.get('vulkan_index'))
        episode = Episode(task, port, folder / 'sim', folder / 'client', layout_id=cfg.get('layout_id', 0))
    elif mode == 'real':
        real = cfg['real']
        validate_real_profile(real['profile'])
        episode = RealEpisode(task, real['port'], folder / 'real', folder / 'client',
                              layout_id=cfg.get('layout_id', 0), rpc_host=real.get('host', '127.0.0.1'),
                              robot_profile=real['profile'])
    else:
        raise ValueError('backend must be sim or real')
    try:
        yield episode
    finally:
        if episode.rpc is not None:
            episode.rpc.close()
        stop(process)  # real mode never signals a hardware process
