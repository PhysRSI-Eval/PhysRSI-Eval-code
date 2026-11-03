"""Expose the frozen hv2f4 insert_charger executor through the reference ToolHost.

All sub-actions share the reset, ACK ledger and native bounded-joint controller.
Executor final targets are distinct from the baseline 5 cm / 0.35 rad actor proposals.
Baseline and transfer arms never import/register this tool. No automatic takeover.
"""
import copy
import json
import threading
import uuid
from pathlib import Path

import numpy as np

from .sim import ActionRejected, CAMERAS
from .tools import ToolHost, text
from .util import write_json, sha256

_LOCK = threading.Lock()  # vendored executor uses module-local calibration/step constants
SPEC = dict(name='insert_charger', description=(
    'Explicitly execute the frozen charger insertion pipeline after you have grasped and lifted the charger. '
    'It estimates the held pose, puts down/regrasps if needed, localizes outlets, transports and inserts. '
    'Uses the SAME physical episode and remaining native step budget. Estimates may abstain. '
    'Returns control on refusal, terminal or budget; it does not certify task success.'),
    inputSchema=dict(type='object', properties={'arm': {'type': 'string', 'enum': ['left', 'right']}},
                     required=['arm'], additionalProperties=False))


def load_profile(cfg, backend):
    if cfg.get('profile_sha256') and sha256(cfg['profile']) != cfg['profile_sha256']:
        raise ValueError('tool profile changed after experiment freeze')
    profile = json.loads(Path(cfg['profile']).read_text())
    if profile.get('backend') != backend:
        raise ValueError('tool profile backend mismatch')
    if backend == 'real' and profile.get('calibration_source') != 'real_measured':
        raise ValueError('real tools require measured target-robot calibration')
    if backend == 'real' and not profile.get('inference_calibration_id'):
        raise ValueError('real inference services must be recalibrated and identified separately from sim')
    for key in ('calibration', 'robot_kin', 'tcp_offset', 'object_geometry', 'charger_vertices', 'width_calibration'):
        if not profile.get(key):
            raise ValueError(f'tool deployment profile requires {key}')
    if type(profile.get('table_z')) not in (int, float) or not np.isfinite(profile['table_z']):
        raise ValueError('tool deployment requires a finite measured table_z')
    if profile.get('controller_model') != 'dual_x5_bounded_dls_v1':
        raise ValueError('rg3f reach predictor currently supports dual X5 bounded-DLS; other controllers need an adapter')
    for key in ('infer_url', 'infer_url_v4', 'locate_url'):
        if not cfg.get(key):
            raise ValueError(f'tool config requires {key}')
    return profile


class Adapter:
    def __init__(self, host):
        self.host = host
        self.output = str(host.episode.client)
        self.evidence = self.output
        self.terminal_reason = None
        self._grip_cmd = {}
        self.refresh()

    def refresh(self):
        ep = self.host.episode
        self.packet = copy.deepcopy(ep.request)
        images = ep.client / 'observations' / f'{len(ep.history):03d}'
        self.packet['images'] = [dict(camera=c, path=str(images / f'{c}.png')) for c in CAMERAS]
        self.terminal_reason = 'native_terminal' if ep.closed else None

    def _invoke(self, name, args, call_id, on_ack):
        if name != 'robodojo_act':
            raise ActionRejected('executor requested unsupported operation')
        content = self.host.episode.act(args, tool_action=True)
        for arm, target in args['target'].items():
            self._grip_cmd[arm] = float(target.get('gripper_opening', not target['gripper_closed']))
        self.refresh()
        if self.host.episode.closed:
            self.host.close('native_terminal')
        return dict(success=True, contentItems=content)


class PlugToolHost(ToolHost):
    program_extra_methods = ('insert_charger',)
    def __init__(self, *args, plug_cfg, backend='sim', **kwargs):
        super().__init__(*args, **kwargs)
        self.plug_cfg = plug_cfg
        self.profile = load_profile(plug_cfg, backend)
        self.adapter = None

    def physical(self, name, args):
        if name == 'insert_charger':
            return self.insert(args)
        return super().physical(name, args)

    def contract(self, args):
        result = super().contract(args)
        packet = json.loads(result[0]['text'])
        if packet.get('program_stdio'):
            packet['program_stdio']['methods'].append('insert_charger')
        packet['optional_tools'] = ['insert_charger']
        packet['executor_action_contract'] = 'insert_charger may command final EEF targets; native bounded-DLS joint controller enforces per-tick limits. Baseline actor bounds remain unchanged.'
        result[0] = text(packet)
        return result

    def call(self, name, args):
        if name != 'insert_charger' or self.closed or self.limit_notice:
            return super().call(name, args)
        return self._with_feedback(self._safe(self.insert, args))

    def insert(self, args):
        if self.closed or not self.episode.started or self.episode.unknown_action:
            raise ActionRejected('start a valid episode before calling insert_charger')
        if set(args) != {'arm'} or args['arm'] not in ('left', 'right'):
            raise ActionRejected('insert_charger requires one arm')
        from .plug_tools import combined_tools as CT, stage_tools as ST, rg3f_deploy as RG, reach_model as RM, align_target as AT
        if not _LOCK.acquire(blocking=False):
            raise ActionRejected('tool executor busy')
        saved = []
        def setv(module, key, value):
            saved.append((module, key, getattr(module, key)))
            setattr(module, key, value)
        try:
            p = self.profile
            for module in (CT, AT):
                setv(module, 'TCP', np.asarray(p['tcp_offset'], float))
                for key in ('OFFS', 'OUT_Y', 'SEAT_P', 'SEAT_R'):
                    value = p['object_geometry'][key]
                    setv(module, key, np.asarray(value) if key == 'SEAT_R' else value)
            setv(CT, 'CALIB', p['calibration'])
            setv(AT, 'SOCKET_TOP', p['object_geometry']['SOCKET_TOP'])
            setv(AT, 'R_PRONG', p['object_geometry']['R_PRONG'])
            setv(ST, 'WIDTH_CAL', p['width_calibration'])
            # The production executor normally reads a frozen CAD file; use this deployment's explicit CAD.
            cad = self.episode.client / 'tool-cad.json'
            write_json(cad, {'local_vertices': p['charger_vertices']})
            setv(RG, 'CAD', cad)
            self.adapter = self.adapter or Adapter(self)
            self.adapter.refresh()
            if self.episode.history:
                for arm, target in self.episode.history[-1]['response']['target'].items():
                    self.adapter._grip_cmd[arm] = float(target.get('gripper_opening', not target['gripper_closed']))
            co = CT.Co(self.adapter, dict(self.plug_cfg, info_only=True))
            co.start_imgs = CT._imgs(self.adapter.packet)
            tool = ST._Tool(self.adapter, 'insert_charger', uuid.uuid4().hex, None, self.plug_cfg)
            # Width telemetry is optional diagnostic only; do not invent measured finger joints.
            tool.width = lambda arm: None
            rm = RM.Reach(p['robot_kin'])
            start = self.episode.request['step_id']
            cap = min(int(self.plug_cfg.get('step_cap', 340)), self.episode.request['remaining_steps'])
            result = RG.run(co, tool, self.adapter, args['arm'], rm,
                            cfg={'v4_url': self.plug_cfg['infer_url_v4'], 'insert': 'R1', 'table_z': p['table_z']}, step_cap=cap)
            if self.episode.unknown_action:
                raise RuntimeError('executor transport UNKNOWN: no replay or further actor actions')
            used = self.episode.request['step_id'] - start
            result = dict(value=result, uncertainty={'deployment': p['backend']},
                          abstain=used == 0 and bool(result.get('failure')), steps_used=used,
                          note='Executor finished; native task judge remains authoritative.')
            write_json(self.episode.client / f'insert-charger-{start:04d}.json', result)
            return [text(result)] + self.episode.last_observation()
        finally:
            for module, key, value in reversed(saved):
                setattr(module, key, value)
            _LOCK.release()
