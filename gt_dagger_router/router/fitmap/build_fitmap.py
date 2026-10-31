"""Method-fit map (user 08:3xZ, METHOD-FIT-20261030): per unique plug state (context, handover step), which method helped:
rg3f executor (comp, native <= 400), VS executor (C1 line rounds o1..o5, latest round per state; truth keypoints = ceiling of that method),
Astra alone (archived). Columns are state features available at the handover (step, lift-time v4 abstain from the rg3f comp row, hold arm)
+ each method's outcome; 'best' = first method that succeeded in the order VS, rg3f, Astra, else 'none'.  -> fitmap.json / FITMAP.md"""
import glob, json
from pathlib import Path
V = Path('/lizekai/RSIws/experiments/macro-tools-20261027'); NAT = Path('/lizekai/RSIws/artifacts/native'); S = {}
def key(ep, t): return ('-'.join(ep.split('-')[:3]), int(t))
for pat in ('out/rgfh1/RGFC*', 'out/rgfaux/RGFC*', 'out/rgffresh/RGFC*'):
    for f in glob.glob(str(V / pat / 'result.json')):
        r = json.load(open(f)); c = r.get('chain') or {}
        if not r.get('restore') or r.get('error'): continue
        k = key(r['start']['episode'], r['restore']['step']); calls = c.get('inhand_calls') or []
        e = S.setdefault(k, dict(ctx=k[0], t=k[1], episode=r['start']['episode'])); e.setdefault('eps', set()).add(r['start']['episode'])
        e.update(rg3f=bool(c.get('inserted')) and (c.get('end_step') or 9999) <= 400, rg3f_fail=c.get('fail') or (None if c.get('inserted') else 'insert'),
                 lift_abstain=any(x.get('tag') == 'lift' and x.get('abstain') for x in calls), hold_arm=r.get('arm'))
for rd in sorted(glob.glob(str(V / 'out/c1o*dev'))):
    for f in glob.glob(rd + '/*/result.json'):
        r = json.load(open(f)); vs = r.get('vs') or {}
        if not r.get('restore'): continue
        k = key(r['start']['episode'], r['restore']['step']); e = S.setdefault(k, dict(ctx=k[0], t=k[1], episode=r['start']['episode']))
        o = vs.get('outcome'); e['vs_round'] = Path(rd).name; e['vs'] = bool(o in ('inserted', 'success') or vs.get('inserted')); e['vs_outcome'] = o
O7 = json.load(open(V / 'c1/out/TRUTH-CEILING-combined-o7.json'))   # C1 line, o7 code, 83 unique states (truth keypoints = VS ceiling)
for e in S.values():
    hit = [O7[x]['outcome'] for x in sorted(e.get('eps', {e['episode']})) if x in O7]
    if hit: e['vs_round'] = 'c1o7'; e['vs_outcome'] = hit[0]; e['vs'] = hit[0] == 'inserted'
    e['eps'] = sorted(e.get('eps', []))
for e in S.values():
    p = NAT / e['episode'] / 'evaluation_outcome.json'
    e['astra'] = bool(json.load(open(p)).get('native_success')) if p.exists() else None
    e['best'] = 'VS' if e.get('vs') else 'rg3f' if e.get('rg3f') else 'Astra' if e.get('astra') else 'none'
rows = sorted(S.values(), key=lambda e: (e['ctx'], e['t'])); json.dump(rows, open(V / 'router-rgb/fitmap/fitmap.json', 'w'), indent=1)
import collections
def bucket(e): return ('t<=100' if e['t'] <= 100 else 't>100', 'lift_abstain' if e.get('lift_abstain') else 'v4_ok' if 'lift_abstain' in e else 'n/a')
T = collections.defaultdict(collections.Counter)
for e in rows: T[bucket(e)][e['best']] += 1
md = ['# 方法适配表 v0（插充电器，唯一状态；%d 个状态，其中 %d 个有 VS 结果）' % (len(rows), sum('vs' in e for e in rows)), '',
      '| 接手步 | 抬起时 v4 | VS 成功 | rg3f 成功 | Astra 成功 | 都不成功 |', '|---|---|---|---|---|---|']
for b in sorted(T): md.append('| %s | %s | %d | %d | %d | %d |' % (b[0], b[1], T[b]['VS'], T[b]['rg3f'], T[b]['Astra'], T[b]['none']))
md.append(''); md.append('"成功"取第一个成功的方法，按 VS → rg3f → Astra 的顺序判。VS 结果优先取 C1 线 o7 的真值上限（TRUTH-CEILING-combined-o7.json，83 个唯一状态，用真值关键点），没有 o7 时取 o1–o5 开发轮次。逐行数据见 fitmap.json。')
open(V / 'router-rgb/fitmap/FITMAP.md', 'w').write('\n'.join(md)); print('\n'.join(md))
