"""hv2f candidate release builder (= hv2e + ROUTER-v0 L3 + hidden check_held_small; built by build_release_hv2f.py from this text) (bottleneck line = harness-layer owner, 10-29 12:xxZ; zero quota, nothing armed).
Source = hv2d 8dc64d35... (hv2 tools + extended step cap + L4 + probe A/B + workspace_seed). Adds, opt-in inside the alias `combined` config:
  * tool insert_charger (executor, explicit call only) = truth-free rg3f deploy backend (rsi/rg3f_deploy.py, byte copy of harness-v2 RC
    rsi_v2/backends/rg3f_deploy.py) + rsi/rg3f_lc.py (byte copy of macro-tools src/loc1cov_lib.py, helpers only) + rsi/charger_bbox.json (CAD box);
  * L3 intervention gate v2 in 'log' / 'offer' mode on per-step history (rsi/harness_l3.py + rsi/gate_v2.json);
  * rsi/hv2e_ext.py glue; three one-line patches: combined_tools.NAMES/PRIVILEGED + run() branch + after_act L3 items, production tool list.
Aliases without `insert_charger` / `l3_policy` are byte-for-byte the hv2d behaviour. Manifest -> release/release-hv2e.json + releases/<name>.json."""
import hashlib, json, shutil, subprocess
from pathlib import Path
X = Path('/lizekai/RSIws/experiments/combined-online-20261026'); OUT = X / 'deploy/releases'; H = Path('/lizekai/RSIws/experiments/harness-v2-20261028')
PY = '/opt/PhysRSI/ENV/robodojo/bin/python'; SRC = OUT / '8dc64d358809a3c169cdac807476b75bf28531b757dffb6a440824f987da77f0'  # hv2d; hv2f re-applies the hv2e patches + hv2f ones
ADD = {'rg3f_deploy.py': H / 'rsi_v2/backends/rg3f_deploy.py', 'rg3f_lc.py': Path('/lizekai/RSIws/experiments/macro-tools-20261027/src/loc1cov_lib.py'),
       'charger_bbox.json': H / 'rsi_v2/cad/charger_bbox.json', 'harness_l3.py': H / 'release/harness_l3.py', 'gate_v2.json': H / 'rsi_v2/policies/gate_v2.json',
       'hv2e_ext.py': H / 'release/hv2e_ext.py', 'hv2f_l3.py': H / 'release/hv2f_l3.py'}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
h = hashlib.sha256(b'combined-online-rel-hv2f:' + SRC.name.encode() + Path(__file__).read_bytes() + b''.join(Path(p).read_bytes() for p in ADD.values())); name = h.hexdigest(); dst = OUT / name
if dst.exists(): shutil.rmtree(dst)
shutil.copytree(SRC, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for k, p in ADD.items(): shutil.copy2(p, dst / 'rsi' / k)
shutil.copytree(H / 'rsi_v2/policies/router_v0', dst / 'rsi' / 'router_v0')


def patch(rel, old, new):
    p = dst / rel; s = p.read_text(encoding='utf-8'); assert s.count(old) == 1, (rel, s.count(old), old[:80]); p.write_text(s.replace(old, new), encoding='utf-8')


# backend inside the release: helpers from the vendored rg3f_lc, CAD box next to the module
patch('rsi/rg3f_deploy.py', "import loc1cov_lib as LC   #", "from . import rg3f_lc as LC   # hv2e: vendored byte copy of loc1cov_lib;")
patch('rsi/rg3f_deploy.py', "CAD = Path(__file__).resolve().parent.parent / 'cad' / 'charger_bbox.json'", "CAD = Path(__file__).resolve().with_name('charger_bbox.json')")
patch('rsi/combined_tools.py', "NAMES = ('locate_outlets', 'estimate_inhand_pose', 'check_insertion_reach', 'check_reach')",
      "NAMES = ('locate_outlets', 'estimate_inhand_pose', 'check_insertion_reach', 'check_reach', 'insert_charger')  # hv2e: insert_charger (opt-in)")
patch('rsi/combined_tools.py', "PRIVILEGED = {'locate_outlets': False, 'estimate_inhand_pose': False, 'check_insertion_reach': False}",
      "PRIVILEGED = {'locate_outlets': False, 'estimate_inhand_pose': False, 'check_insertion_reach': False, 'insert_charger': False}")
patch('rsi/combined_tools.py', "        elif name == 'check_insertion_reach':\n",
      "        elif name == 'insert_charger':  # hv2e executor (explicit call only)\n"
      "            if not isinstance(cfg.get('insert_charger'), dict): raise ST.Refused('not available in this harness')\n"
      "            from . import hv2e_ext as _x\n"
      "            out = _x.insert_charger(ad, co, t, args, cfg)\n"
      "        elif name == 'check_insertion_reach':\n")
patch('rsi/combined_tools.py', "        co = state(ad, cfg); items = co.after(args)\n",
      "        co = state(ad, cfg); items = co.after(args)\n"
      "        if isinstance(cfg.get('l3_policy'), dict):  # hv2e: intervention gate v2 (log / offer), per-step history\n"
      "            from . import hv2e_ext as _x\n"
      "            try: items = list(items or []) + _x.l3_after_act(ad, cfg)\n"
      "            except Exception as e:  # noqa: BLE001\n"
      "                (Path(ad.evidence) / 'l3-gate-error.txt').write_text('%s: %s' % (type(e).__name__, str(e)[:300]))\n")
patch('rsi/production.py', "                from .combined_tools import SPECS_HV2 as _CO_SPECS\n",
      "                from .combined_tools import SPECS_HV2 as _CO_SPECS\n"
      "            if isinstance(((_rgb_alias(_rgb_local, request['task_id']) or {}).get('combined') or {}).get('insert_charger'), dict):  # hv2e\n"
      "                from .hv2e_ext import SPEC_INSERT_CHARGER as _IC; _CO_SPECS = list(_CO_SPECS) + [_IC]\n")
patch('rsi/hv2e_ext.py', "        st['gate'] = HL.GateV2(HERE / 'gate_v2.json'); st['log'] = HL.StepLog(ad.output); st['offered'] = False\n",
      "        if pc.get('name') == 'router_v0':  # hv2f: ROUTER-v0 (log / offer only, plug + insert_charger only)\n"
      "            from . import hv2f_l3 as _F; st['gate'] = _F.RouterV0(HERE / 'router_v0')\n"
      "        else:\n"
      "            st['gate'] = HL.GateV2(HERE / 'gate_v2.json')\n"
      "        st['log'] = HL.StepLog(ad.output); st['offered'] = False\n")
patch('rsi/hv2e_ext.py', "               executor='insert_charger' if isinstance(cfg.get('insert_charger'), dict) else None)\n",
      "               executor='insert_charger' if isinstance(cfg.get('insert_charger'), dict) else None,\n"
      "               images={i_['camera']: i_['path'] for i_ in ad.packet.get('images', [])}, tool_facts=dict(ad.__dict__.get('_hv2f_tool_facts', {})))\n")
r = subprocess.run([PY, '-B', '-m', 'compileall', '-q', str(dst / 'rsi')], capture_output=True, text=True); assert r.returncode == 0, r.stdout + r.stderr
for p in dst.rglob('__pycache__'): shutil.rmtree(p)
src_m = json.loads((H.parent / 'online-plug-hv2b-20261029/release-hv2d.json').read_text())
m = dict(release=str(dst), release_name=name, source=str(SRC), source_name=SRC.name, label='combined-online-hv2f-router-l3', entry_wrapper=src_m['entry_wrapper'],
         entry_wrapper_sha256=src_m['entry_wrapper_sha256'], pkg_files=dict(src_m['pkg_files'], **{k: sha(dst / 'rsi' / k) for k in ADD}),
         added_from={k: dict(src=str(p), sha256=sha(p)) for k, p in ADD.items()},
         external_frozen={str(p): sha(p) for p in ['/lizekai/RSIws/experiments/macro-tools-20261027/pkg-v2/rsi/align_target.py', '/lizekai/RSIws/experiments/macro-tools-20261027/pkg-v2/rsi/grasp_compat2.py',
                                                   '/lizekai/RSIws/experiments/macro-tools-20261027/pkg-v15/rsi/reach_model.py', '/lizekai/RSIws/experiments/combined-online-20261026/plug-offline-eval/v14/src/reach.py',
                                                   '/lizekai/RSIws/experiments/combined-online-20261026/plug-offline-eval/v14/out/kin.json', '/lizekai/RSIws/experiments/innerloop-20261026/track1-contact/src/detector_frozen.json',
                                                   '/lizekai/RSIws/release/PhysRSI-Eval-code/physrsi_eval/sim_server.py']},
         unchanged_vs_hv2d={f: sha(dst / 'rsi' / f) == sha(SRC / 'rsi' / f) for f in ('native_runtime.py', 'native_adapter.py', 'learning.py', 'original_semantics.py', 'rsiws_ext.py', 'experience_l4.py', 'monitor_hub.py', 'stage_tools.py')},
         builder_sha256=sha(__file__),
         patches=['hv2d unchanged (step cap, L4, probe A/B, workspace_seed)', 'rg3f_deploy: vendored LC + CAD path', 'combined_tools: NAMES/PRIVILEGED + insert_charger branch + after_act L3 items',
                  'production: insert_charger spec appended when combined.insert_charger is a dict', 'hv2f: l3_policy.name=router_v0 -> RouterV0 (rsi/router_v0, log/offer only); hidden check_held_small helper in hv2f_l3 (not listed)'])
(H / 'release/release-hv2f.json').write_text(json.dumps(m, indent=1)); (OUT / (name + '.json')).write_text(json.dumps(m, indent=1)); print(name)
