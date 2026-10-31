"""C1 (10-31): build mt_orch3 job lists for bc_rollout.py.  python make_bc_jobs.py READY.json TAG SHARD NSHARDS OUT.json [smoke]
READY.json (written by the tool line when the BC policy exists): {"policy": "/abs/POLICY.py", "ckpt": "/abs/...", "exec_n": 8}.
Rows: the 83 unique plug handover states (c1/unique-starts.json) + one public start state per layout (--t 0, --start 1, budget 400)."""
import json, sys
V = '/lizekai/RSIws/experiments/macro-tools-20261027'
ready, tag, shard, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
extra = [x for x in sys.argv[6:] if x != 'smoke']; smoke = 'smoke' in sys.argv[6:]   # extra = bc_rollout flags, e.g. --gripgate 1
import hashlib
_src = open(V + '/c1/src/bc_rollout.py', 'rb').read()
assert b'RELEASE_DXY_MM' in _src, 'bc_rollout.py lacks the 10-31 release fix (e9503b6b or later)'   # every later rollout labels with the fixed expert
print('bc_rollout sha256', hashlib.sha256(_src).hexdigest()[:16])
R = json.load(open(ready)); base = {}
for h in ('aux', 'h1', 'h2'):
    for j in json.load(open(V + '/c1/jobs-o6t-%s.json' % h)):
        a = j['args'][j['args'].index('src/loc1cov.py'):]; base[a[a.index('--episode') + 1]] = a
U = json.load(open(V + '/c1/unique-starts.json'))['unique']
meta = {}
for f in ('/lizekai/RSIws/experiments/macro-tools-20261027/loc1cov/meta_r4.json', '/lizekai/RSIws/experiments/macro-tools-20261027/run/m3_meta_extra.json'):
    try: meta.update(json.load(open(f)))
    except Exception: pass
rows = [(e, False) for e in sorted(U)]
seen = set()
for e in sorted(U):
    lay = (meta.get(e) or {}).get('layout')
    if lay is not None and lay not in seen: seen.add(lay); rows.append((e, True))
if smoke: rows = rows[:2]
J = []
for i, (e, start) in enumerate(rows):
    if i % n != shard: continue
    a = list(base[e]); a[a.index('--tag') + 1] = tag + ('s' if start else ''); a[a.index('--exec-mode') + 1] = 'bc'; a[a.index('--chain') + 1] = 'bc'
    if start: a = a + ['--t', '0']
    pre = extra + ['--policy', R['policy'], '--ckpt', R.get('ckpt', ''), '--exec_n', str(R.get('exec_n', 8)), '--budget', '400' if start else ('40' if smoke else '250'), '--start', '1' if start else '0']
    tid = tag + ('s' if start else '')
    J.append(dict(name='%s-%s' % (tid.upper(), e[-8:]), script='../c1/src/bc_rollout.py', args=pre + a,
                  scratch='c1-LCV-fix2w-bcR1bc-%s-%s-tools-loc1-est' % (tid, e[-8:])))
json.dump(J, open(out, 'w'), indent=0); print(len(J), 'jobs ->', out)
