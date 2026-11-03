"""No model or hardware: paired protocol, isolation, success provenance and ambiguous ACK tests."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from physrsi_eval.plug_experiment import make_config
from physrsi_eval.workspace import (initialize_environment, bind_reset, file_hashes, export_success,
                                   build_success_seed, portability_audit)
from physrsi_eval.util import write_json
from physrsi_eval.real_server import RealSession
from physrsi_eval.sim import validate_action, ActionRejected
from physrsi_eval.loop import load_task, run_chain
from test_host import FakeSession, hold_action

PROFILE = dict(robot='ARX X5 2023 dual arm', frame='measured_table', eef_link='link6', control_hz=25,
               calibration_id='test-only', calibration_source='real_measured')

@pytest.mark.parametrize('backend', ['sim','real'])
@pytest.mark.parametrize('group', ['scratch','experience','tools'])
def test_three_groups(backend,group):
    cfg=make_config({'plug_tools':{'profile':'x'}},group,backend,'real' if group=='experience' else None)
    assert bool(cfg.get('plug_tools')) == (group=='tools')
    assert cfg['experiment']['task']=='plug_in_charger'
    assert cfg['experiment']['schedule']=='K0x3 + K1..K17'


def test_domain_and_empty_workspace(tmp_path):
    cfg=make_config({},'scratch','real');cfg['real']={'profile':PROFILE}
    initialize_environment(tmp_path,cfg,load_task('plug_in_charger'))
    scene=tmp_path/'memory/environment/scene.json'
    bind_reset(tmp_path,{'episode_id':'one'})
    assert json.loads(scene.read_text())['episode_id']=='one'
    initialize_environment(tmp_path,cfg,load_task('plug_in_charger'))
    assert json.loads(scene.read_text())['objects']=={}
    assert json.loads(scene.read_text())['episode_id'] is None
    assert not (tmp_path/'skills').exists()


class Driver:
    """Test fixture only: synthetic states/outcomes must never enter published success library."""
    def __init__(self):
        task=load_task('plug_in_charger')
        self.fake=FakeSession(task['task_id'],task['native_horizon_steps'],task['native_instruction'])
        self.metadata=dict(self.fake.metadata,**PROFILE,backend='real')
        self.calls=[]
        self.fail=False
    def reset(self,layout_id):
        self.calls.append('reset');return self.fake.dispatch('reset',{'layout_id':layout_id})
    def observe(self): return self.fake.obs()
    def act(self,targets,steps):
        self.calls.append('act')
        if self.fail: raise ConnectionError('ACK lost after send')
        return self.fake.dispatch('act',dict(targets=targets,steps=steps))
    def finish(self,reason): return self.fake.outcome(reason)


def action(session):
    obs=session.observation()
    target={a:dict(position=obs['eef_positions'][i].tolist(),quaternion_wxyz=obs['eef_quaternions_wxyz'][i].tolist(),gripper_closed=False) for i,a in enumerate(('left','right'))}
    return dict(targets=target,steps=5)


def test_real_driver_roundtrip(tmp_path):
    d=Driver();s=RealSession(d,tmp_path)
    r=s.dispatch('reset',{'layout_id':0});assert r['episode_id']=='fake'
    ack=s.dispatch('act',action(s));assert ack['step_id']==5
    result=s.dispatch('finish',{'reason':'actor_completed'})
    assert result['native_control_steps']==5 and not result['native_success']
    assert len(list((tmp_path/'acks').glob('*.json')))==3


def test_unknown_never_replayed(tmp_path):
    d=Driver();s=RealSession(d,tmp_path);s.dispatch('reset',{'layout_id':0});args=action(s);d.fail=True
    with pytest.raises(ConnectionError):s.dispatch('act',args)
    with pytest.raises(RuntimeError,match='UNKNOWN'):s.dispatch('act',args)
    assert d.calls.count('act')==1 and (tmp_path/'pending.json').exists()


def test_malformed_ack_blocks(tmp_path):
    d=Driver();s=RealSession(d,tmp_path);s.dispatch('reset',{'layout_id':0});args=action(s)
    d.act=lambda **kw:dict(executed_steps=5,step_id=90)
    with pytest.raises(RuntimeError):s.dispatch('act',args)
    assert s.poisoned and (tmp_path/'pending.json').exists()


def test_executor_requires_site_controller_optin(tmp_path):
    d=Driver();s=RealSession(d,tmp_path);s.dispatch('reset',{'layout_id':0})
    with pytest.raises(ValueError,match='bounded-DLS'):s.dispatch('act',dict(action(s),executor=True))
    assert 'act' not in d.calls


def source_round(tmp_path):
    r=tmp_path/'round'
    write_json(r/'client/receipt.json',{'task_id':'plug_in_charger','native_success':True})
    write_json(r/'client/result.json',{'native_success':True,'valid':True})
    write_json(r/'client/reset.json',{'episode_id':'synthetic-test-only'})
    write_json(r/'workspace/memory/environment/deployment.json',{'backend':'real'})
    (r/'snapshot/skills').mkdir(parents=True)
    (r/'snapshot/skills/method.md').write_text('Observe the current charger before grasping.')
    (r/'snapshot/skills/old-pose.md').write_text('position xyz = [1,2,3]; /lizekai/source')
    return r


def test_success_transfer_preserves_originals_and_excludes_local(tmp_path):
    r=source_round(tmp_path);b=tmp_path/'bundle'
    export_success(r,b,'real',{'skills/method.md':'portable','skills/old-pose.md':'episode_local'})
    seed=build_success_seed([b],tmp_path/'seed','real')
    assert (seed/'skills/method.md').exists() and not (seed/'skills/old-pose.md').exists()
    assert (seed/'memory/library/real-00/original-workspace/skills/old-pose.md').exists()
    assert portability_audit(r/'snapshot')[1]['needs_review']
    with pytest.raises(ValueError):build_success_seed([b],tmp_path/'wrong','sim')
    (b/'portable/skills/method.md').write_text('tampered')
    with pytest.raises(ValueError,match='hash'):build_success_seed([b],tmp_path/'tampered','real')


def test_hypothetical_success_rejected(tmp_path):
    r=source_round(tmp_path);write_json(r/'client/result.json',{'native_success':False,'valid':True})
    with pytest.raises(ValueError,match='native-success'):export_success(r,tmp_path/'b','real',{})


def test_incomplete_round_not_replayed(tmp_path):
    (tmp_path/'K01').mkdir()
    with pytest.raises(RuntimeError,match='not replayed'):
        run_chain({}, {'name':'test'},load_task('plug_in_charger'),tmp_path)


def test_pipeline_import_and_profile():
    from physrsi_eval.plug_tools import rg3f_deploy,rg3f_lc
    assert rg3f_lc._at() is not None
    assert callable(rg3f_deploy.run)


def test_real_episode_over_same_rpc(tmp_path):
    import threading,time
    from physrsi_eval.real_server import serve
    from physrsi_eval.backends import RealEpisode
    from physrsi_eval.loop import free_port
    d=Driver();session=RealSession(d,tmp_path/'service');port=free_port()
    threading.Thread(target=serve,args=(session,port),daemon=True).start();time.sleep(.05)
    ep=RealEpisode(load_task('plug_in_charger'),port,tmp_path/'real',tmp_path/'client',robot_profile=PROFILE)
    packet=json.loads(ep.start()[0]['text'])
    assert packet['robot_profile']==PROFILE['robot']
    ep.act(hold_action(packet,steps=5))
    assert ep.history[0]['executed_steps']==5
    assert ep.finish('actor_completed')['native_control_steps']==5
    assert not (tmp_path/'service/active.json').exists()


def test_pipeline_abstain_through_program(tmp_path,monkeypatch):
    import sys
    from test_host import fake_episode
    from physrsi_eval.loop import prepare_workspace
    from physrsi_eval.plug_pipeline import PlugToolHost
    from physrsi_eval.plug_tools import rg3f_deploy as RG,reach_model as RM
    task=load_task('plug_in_charger');prepare_workspace(tmp_path/'workspace')
    cfg=json.loads(Path('configs/plug_in_charger/sim.json').read_text())['plug_tools']
    host=PlugToolHost('Harness',task,fake_episode(tmp_path,task,horizon=400),1,
                     workspace=tmp_path/'workspace',chain=tmp_path,
                     program_cfg={'sandbox':'none','python':sys.executable},plug_cfg=cfg)
    monkeypatch.setattr(RM,'Reach',lambda kin:object())
    monkeypatch.setattr(RG,'inhand_v4',lambda *a:(None,None,{'abstain':True,'flip_flag':True,'uncertainty':{},'step':0}))
    host.start({})
    (tmp_path/'workspace/tools/probe.py').write_text("import sys;sys.path.insert(0,'tools');from mcp_client import call\ncall('insert_charger',{'arm':'left'})\n")
    result=host.program({'path':'tools/probe.py'})
    assert json.loads(result[0]['text'])['exit_code']==0
    assert not host.episode.history
    diagnostic=json.loads((tmp_path/'client/insert-charger-0000.json').read_text())
    assert diagnostic['abstain'] and diagnostic['steps_used']==0
    host.close('actor_completed')


def test_baseline_still_rejects_large_target():
    current={a:dict(position=[0,0,0],quaternion_wxyz=[1,0,0,0]) for a in ('left','right')}
    args=dict(request_id='a',mode='eef',reason='test',steps=5,target={a:dict(current[a],gripper_closed=False) for a in current})
    args['target']['left']['position']=[.2,0,0]
    with pytest.raises(ActionRejected):validate_action(args,{'request_id':'a'},current)
    validate_action(args,{'request_id':'a'},current,executor=True)


def test_restart_preserves_unfinished_real_episode(tmp_path):
    import fcntl
    d=Driver();s=RealSession(d,tmp_path);s.dispatch('reset',{'layout_id':0})
    fcntl.flock(s.lock,fcntl.LOCK_UN);s.lock.close()
    restarted=RealSession(Driver(),tmp_path)
    with pytest.raises(RuntimeError,match='UNKNOWN'):restarted.dispatch('reset',{'layout_id':0})
