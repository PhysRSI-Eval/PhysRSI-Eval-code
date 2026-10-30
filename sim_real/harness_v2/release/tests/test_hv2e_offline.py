"""hv2c offline/dry tests (no model calls, no native process). Run with PYTHONPATH=<hv2c release>:/opt/PhysRSI:/opt/PhysRSI/runtime/GPT-as-Policy.
T1 L4: LearningStore._validate (original cohort) with a temp runtime: bad package -> Rejected('L4 experience package: ...') -> public revision
   error (success False, English) and the L4 continuation suffix; second bad -> 'l4_invalid_no_repair' recorded, no raise; another episode with a
   valid package -> 'valid' and a frozen read-only copy; an alias without l4_package -> untouched; l4_note text (ASCII, mentions the path).
T2 probe A: context_frame_items for a recorded hv2a episode (2 steps x cam_high) -> text + image items; off when the field is absent.
T3 probe B: the persisted entry's session class on a fake env/session: resume -> env.step_lim = start + N, metadata max_episode_steps N,
   reset.json rewritten, evaluation_outcome carries resume; step cap only -> 800/400 fields (unit level; the native smoke covered the real env).
   + native_runtime.contract() for an alias with continuation + resume_steps (dry: no plan/launch): native_horizon = N.
T4 probe C: seed -> original-cohort baseline copy (copytree + materialize_baseline) -> workspace copy + host helper -> actor admission manifest
   digest == baseline tree digest; the seeded cards are inside the committed tree.  -> tests/RESULT-hv2c-offline.json"""
import json, os, shutil, sys, tempfile
from pathlib import Path
R = {}
from rsi import rsiws_ext as X, learning as LRN, runtime_bridge as RB, original_semantics as OS
EP = 'ctx-677338bf42b24288-1-f144a448'  # hv2a TREAT success (stored frames)
GOOD = dict(schema='rsi-exp-v1', task='plug_in_charger', episode=EP, outcome=dict(native_success=True, steps=306),
            action_ledger=[dict(purpose='locate the outlets', contact_mode='none', step_span=[0, 105], method='called locate_outlets at the start',
                                tool_calls=[dict(tool='locate_outlets', inputs={}, returned=dict(outlets_xyz='outlets_xyz'), used_next='yes')],
                                outcome='outlets returned', observed_effect='observer arm moved and returned', failure_mechanism=None, next_adjustment=None,
                                reusable_lesson='hover over outlets_xyz before descending', evidence_steps=[5, 105])],
            keyframes=[dict(step=105, camera='cam_high', why='after locate')],
            binding=dict(episode_local_values=[dict(name='outlets_xyz', value=[0.1, 0.2, 0.8], unit='m', recompute=dict(method='tool', tool='locate_mm', args=dict(target='outlets')))],
                         general_facts=['The strip outlets are recomputed each episode (outlets_xyz).']),
            program_index=[], summary='Located outlets first, then seated the charger.')
BAD = json.loads(json.dumps(GOOD)); BAD['action_ledger'][0]['reusable_lesson'] = 'move to x=0.1234 then descend 15 mm'; BAD['action_ledger'][0]['tool_calls'][0]['tool'] = 'magic'
tmp = Path(tempfile.mkdtemp(dir='/lizekai/RSIws/experiments/harness-v2-20261028/release/tests')); local = tmp / 'local'; (local / 'learning').mkdir(parents=True)
X._alias = lambda loc, task: {'l4_package': {}} if task == 'plug_in_charger__l4' else {}
store = LRN.LearningStore.__new__(LRN.LearningStore); store.root = local / 'learning'
chain = dict(lineage=dict(cohort='rsi-original-v1')); is_orig = OS.is_original(chain)
def stage_with(pkg, k=1):
    s = Path(tempfile.mkdtemp(dir=tmp)); b = Path(tempfile.mkdtemp(dir=tmp))
    (s / OS.SUMMARY_PATH.format(k=k)).parent.mkdir(parents=True, exist_ok=True); (s / OS.SUMMARY_PATH.format(k=k)).write_text(json.dumps(dict(k=k, summary='tried')))
    if pkg is not None: (s / 'memory/experience').mkdir(parents=True, exist_ok=True); (s / ('memory/experience/K%d.json' % k)).write_text(json.dumps(pkg))
    return s, b
t1 = dict(chain_is_original=is_orig)
rc = dict(episode_id=EP, task_id='plug_in_charger__l4', k=1)
s, b = stage_with(BAD)
try:
    store._validate(s, b, rc, chain); t1['first_bad'] = 'NOT RAISED'
except LRN.Rejected as e:
    t1['first_bad'] = str(e)[:160]; pub = RB.public_revision_error(e); t1['public'] = dict(success=pub['success'], text=pub['contentItems'][0]['text'][:120])
    t1['continuation'] = OS.continuation_suffix(pub['contentItems'][0]['text'], 'DEFAULT')[:90]
t1['second_bad'] = (store._validate(s, b, rc, chain), json.load(open(local / 'learning/l4' / (EP + '.json')))['status'])
rc2 = dict(rc, episode_id='ctx-677338bf42b24288-2-7c73fd32'); g = dict(GOOD, episode=rc2['episode_id']); s2, b2 = stage_with(g)
store._validate(s2, b2, rc2, chain); f = local / 'learning/experience' / (rc2['episode_id'] + '.json'); t1['good'] = dict(status=json.load(open(local / 'learning/l4' / (rc2['episode_id'] + '.json')))['status'], frozen=f.exists(), mode=oct(f.stat().st_mode & 0o777))
rc3 = dict(rc, task_id='plug_in_charger__plain', episode_id='x-plain'); s3, b3 = stage_with(None); store._validate(s3, b3, rc3, chain); t1['no_alias_untouched'] = not (local / 'learning/l4/x-plain.json').exists()
note = X.l4_note(local, 'plug_in_charger__l4', 3); t1['note'] = dict(ascii=note.isascii(), has_path='memory/experience/K3.json' in note, chars=len(note), off=X.l4_note(local, 'plug_in_charger__plain', 3) == '')
t1['pass'] = bool(is_orig and t1['first_bad'].startswith('L4 experience package') and t1['public']['success'] is False and 'fix the experience package' in t1['continuation']
                  and t1['second_bad'][1] == 'l4_invalid_no_repair' and len(str(t1['second_bad'][0])) == 64 and t1['good'] == dict(status='valid', frozen=True, mode='0o444') and t1['no_alias_untouched'] and all(t1['note'].values()) if isinstance(t1['note'], dict) else False)
t1['pass'] = bool(is_orig and t1['first_bad'].startswith('L4 experience package') and t1['public']['success'] is False and 'fix the experience package' in t1['continuation']
                  and t1['second_bad'][1] == 'l4_invalid_no_repair' and len(str(t1['second_bad'][0])) == 64 and t1['good']['status'] == 'valid' and t1['good']['frozen'] and t1['no_alias_untouched']
                  and t1['note']['ascii'] and t1['note']['has_path'] and t1['note']['off'])
R['T1_L4'] = t1
# T2 probe A
from rsi.image_payload import observation_image_url
X._alias = lambda loc, task: {'context_frames': dict(episode=EP, steps=[100, 200], cameras=['cam_high'])} if task == 'probe' else {}
items = X.context_frame_items(local, 'probe', observation_image_url)
R['T2_context_frames'] = dict(n=len(items), kinds=[i['type'] for i in items], off=X.context_frame_items(local, 'other', observation_image_url) == [],
                              pass_=len(items) == 6 and [i['type'] for i in items].count('inputImage') == 2)
# T3 probe B (entry session class, fake base)
sys.path.insert(0, '/lizekai/RSIws/experiments/native-ext-20261029/src/native'); import rsi_native_entry as E
out = tmp / 'native'; out.mkdir()
class Env: step_lim = 400
class Base:
    def __init__(self, env, output, task, evaluation_identity=None): self.env, self.output, self.metadata, self.step_id = env, Path(output), dict(max_episode_steps=int(env.step_lim)), 0
    def reset(self, seed, source, policy_version):
        self.step_id = 150; return dict(step_id=150, metadata=dict(self.metadata), continuation=dict(start_step=150))
    def _write_summary(self, reason):
        (self.output / 'evaluation_outcome.json').write_text(json.dumps(dict(native_success=False, native_control_steps=self.step_id, native_step_limit=self.metadata['max_episode_steps'])))
Cls = E.step_cap_class(Base)
os.environ.pop('RSI_STEP_CAP_FACTOR', None); os.environ['RSI_RESUME_STEPS'] = '120'
s_ = Cls(Env(), out, 'plug_in_charger'); r_ = s_.reset(0, 'student', 'x'); s_.step_id = 270; s_._write_summary('terminal')
t3 = dict(env_step_lim=s_.env.step_lim, meta_max=s_.metadata['max_episode_steps'], reset_meta=r_['metadata']['max_episode_steps'], reset_json=json.load(open(out / 'reset.json'))['metadata']['max_episode_steps'],
          outcome_resume=json.load(open(out / 'evaluation_outcome.json')).get('resume'))
os.environ.pop('RSI_RESUME_STEPS'); os.environ['RSI_STEP_CAP_FACTOR'] = '2'; out2 = tmp / 'native2'; out2.mkdir()
s2_ = Cls(Env(), out2, 'plug_in_charger'); s2_.step_id = 800; s2_._write_summary('terminal'); o2 = json.load(open(out2 / 'evaluation_outcome.json')); os.environ.pop('RSI_STEP_CAP_FACTOR')
t3['stepcap'] = dict(env_step_lim=s2_.env.step_lim, meta=s2_.metadata['max_episode_steps'], orig=o2.get('native_step_limit_original'), cap=o2.get('extended_step_cap'))
# contract dry with resume_steps
from rsi import native_runtime as NR, ablation as AB
from rsi.storage import Storage
_orig = AB._alias; spec = tmp / 'cont.json'; spec.write_text(json.dumps(dict(episode=EP, start_step=150)))
AB._alias = lambda loc, task: dict(_orig(loc, task) or {'arm': 'TREAT', 'base': 'plug_in_charger'}, continuation=dict(spec=str(spec)), resume_steps=120) if task == 'plug_in_charger__48024e2d' else _orig(loc, task)
class _Sc:  # 12:xxZ test-environment fix (not a release change): the hv2b alias is no longer in the live admitted scope -> pin the scope for this dry call
    def get(self, p, f): return ['plug_in_charger__48024e2d']
_sc0 = NR._SCOPE; NR._SCOPE = _Sc()
c = NR.NativeRuntime(Storage('/root/RSIws-runtime/runtime-config.json')).contract('plug_in_charger__48024e2d'); AB._alias = _orig; NR._SCOPE = _sc0
t3['contract'] = dict(native_horizon=c['native_horizon'], resume_steps=c.get('resume_steps'), native_horizon_native=c.get('native_horizon_native'), note=c['public_task']['episode_budgets'].get('native_limit_note'))
t3['pass'] = bool(t3['env_step_lim'] == 270 and t3['meta_max'] == 120 and t3['reset_json'] == 120 and t3['outcome_resume'] and t3['stepcap'] == dict(env_step_lim=800, meta=800, orig=400, cap=800)
                  and t3['contract']['native_horizon'] == 120 and t3['contract']['native_horizon_native'] == 400)
R['T3_resume_steps'] = t3
# T4 probe C
from rsi.learning import tree, digest
seed = Path('/lizekai/RSIws/experiments/native-ext-20261029/seeds/shared-20261029'); base = tmp / 'baseline'
shutil.copytree(seed, base, symlinks=False); OS.materialize_baseline(base); expected = tree(base)
ws = tmp / 'ws'; shutil.copytree(base, ws, dirs_exist_ok=True, symlinks=True); OS.install_helper(ws)
adm = digest(OS.actor_manifest(ws)) == digest(expected)
R['T4_workspace_seed'] = dict(files=sorted(k for k, v in expected.items() if v != 'DIRECTORY'), admission_digest_equal=adm, helper_outside_hash=OS.CLIENT_HELPER_PATH not in expected,
                              writable_memory=os.access(ws / 'memory', os.W_OK), pass_=adm and len([k for k in expected if k.startswith('skills/shared/')]) >= 7)
R['all_pass'] = bool(R['T1_L4']['pass'] and R['T2_context_frames']['pass_'] and R['T3_resume_steps']['pass'] and R['T4_workspace_seed']['pass_'])
shutil.rmtree(tmp, ignore_errors=True)
Path('/lizekai/RSIws/experiments/harness-v2-20261028/release/tests/RESULT-hv2e-offline.json').write_text(json.dumps(R, indent=1, default=str)); print(json.dumps(R, indent=1, default=str))
