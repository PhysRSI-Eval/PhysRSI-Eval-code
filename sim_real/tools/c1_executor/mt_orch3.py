"""mt_orch3.py (capability line 10-29): mt_orch2 + per-job sweep of the job's /root/pe-1026/sim scratch (job field "scratch": kill its server, rm -rf the dir).
mt_orch2.py (21:3xZ fix of the host1 leak): each driver in its own session/process group; on the 2400 s timeout the whole group is
killed, and after every job any simulator server still running under that job's scratch name is killed (mt_orch.py SIGKILLed only the
driver, leaving its server running and holding ~5 GB GPU -> 13 orphans on host1, cascading TimeoutErrors).
"""
"""macro-tools batch runner (copy of regrasp rg_orch.py; tmux mt-orch-<name>-1027): jobs file (JSON list of {name, script, args}) -> src/<script>, GPU slots (<= 3 sims per GPU),
STATE in ../STATE.json['batches'][name]; known sim port race retried once.  python rg_orch.py --name N --jobs jobs.json --gpus 0,1,2,3,4,5,6 --per-gpu 3"""
import argparse, json, os, signal, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
HERE = Path(__file__).resolve().parent; V = HERE.parent; PY = '/opt/PhysRSI/ENV/robodojo/bin/python'
ap = argparse.ArgumentParser(); ap.add_argument('--name', required=True); ap.add_argument('--jobs', required=True); ap.add_argument('--gpus', default='0,1,2,3,4,5,6'); ap.add_argument('--per-gpu', type=int, default=3)
a = ap.parse_args(); assert a.per_gpu <= 3; jobs = json.load(open(a.jobs))
slots = [int(g) for g in a.gpus.split(',') for _ in range(a.per_gpu)]; free = list(slots); run = {}; done = []; retried = []; od = V / 'out' / a.name; od.mkdir(parents=True, exist_ok=True)
def state(ph):
    try: _state(ph)
    except Exception as e: print('STATE write failed (ignored):', e, flush=True)
def _state(ph):
    p = V / 'STATE.json'; s = json.loads(p.read_text()) if p.exists() else {}
    s.setdefault('batches', {})[a.name] = dict(updated_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), phase=ph, jobs=len(jobs), done=len(done), running=sorted(run), retried=retried, out=str(od), session='mt-orch-%s-1027' % a.name)
    t = p.with_name('STATE.json.tmp.%s.%d' % (a.name, os.getpid())); t.write_text(json.dumps(s, indent=1)); os.replace(t, p)  # per-batch tmp: a shared STATE.tmp crashed 2 of 3 dev8 orchestrators
q = list(jobs); state('RUNNING')
while q or run:
    while q and free:
        j = q.pop(0); g = free.pop(0); log = open(V / 'logs' / ('%s-%s.log' % (a.name, j['name'])), 'a')
        run[j['name']] = (subprocess.Popen([PY, '-B', '-u', str(HERE / j.get('script', 'key_run.py'))] + j['args'] + ['--gpu', str(g), '--out', str(od / j['name'])], stdout=log, stderr=subprocess.STDOUT, cwd=str(V), start_new_session=True,
                                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1')), g, time.time(), j); time.sleep(4)
    for n, (p, g, t0, j) in list(run.items()):
        if p.poll() is None and time.time() - t0 < 2400: continue
        if p.poll() is None:
            try: os.killpg(p.pid, signal.SIGKILL)  # v2: the driver's whole process group (its simulator server too)
            except Exception: p.kill()  # noqa: BLE001
            p.wait()
        subprocess.run(['pkill', '-9', '-f', '[-]-output /lizekai/RSIws/experiments/regrasp-20261027/env/sim/[A-Z0-9]*-%s' % n])  # v2: sweep a leaked server of this job
        if j.get('scratch'):  # v3: the plug drivers' scratch (Kit cache copy + sim logs) lives in /root/pe-1026/sim/<scratch>
            subprocess.run(['pkill', '-9', '-f', '[-]-output /root/pe-1026/sim/%s' % j['scratch']]); subprocess.run(['rm', '-rf', '/root/pe-1026/sim/%s' % j['scratch']])
        del run[n]; free.append(g)
        try:
            r = json.loads((od / n / 'result.json').read_text()); race = 'Address already in use' in str(r.get('error', ''))
        except Exception:  # noqa: BLE001
            race = False
        if race and n not in retried: retried.append(n); q.append(j)
        else: done.append(n)
        state('RUNNING %d/%d' % (len(done), len(jobs)))
    time.sleep(3)
state('DONE'); (V / 'logs' / ('%s.done' % a.name)).write_text('done\n')
