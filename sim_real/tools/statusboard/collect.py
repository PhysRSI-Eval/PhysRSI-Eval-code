"""Status board collector (rsiws-dc, 2026-10-30; user: rewrite the front end with running tasks per host, account quota, line status).
Every 30 s writes ops/statusboard/www/status.json from shared-disk sources only (no ssh):
  hosts (hb.py heartbeats; main live), job queues ops/jobq/*, mt_orch batches (macro-tools STATE.json), account quota (tmux rsi-metadata pane),
  line STATE files (C1, KPNet factory, router RGB, Track C), milestones (MILESTONES-20261030.md), H200 relay mirror.
  python3 collect.py [--loop]"""
import argparse, json, os, re, subprocess, time
from pathlib import Path

R = Path('/lizekai/RSIws'); SB = R / 'ops/statusboard'; WWW = SB / 'www'
import sys
sys.path.insert(0, str(SB)); import hb  # noqa: E402

LINES = {'ee': 'Track C 对齐/路由第 1 层（rsiws-dc）', 'c1': 'C1 执行器（能力线）', 'cp': '能力线 / 路由器 Track D', 'ip4': '后训练线 KPNet',
         'mt': '编排器 / 服务', 'bk': '卡点线', 'bn': '卡点线服务', 'rsi': '额度采集', 'disk': '磁盘守护', 'kfc': 'KPNet 工厂'}


def fresh(t, s=180): return t is not None and time.time() - t < s


def hosts():
    out = {'main': hb.snapshot('main')}
    for f in (SB / 'hosts').glob('*.json'):
        if f.stem == 'main': continue
        try: d = json.loads(f.read_text()); d['stale'] = not fresh(d.get('time')); out[f.stem] = d
        except Exception: pass  # noqa: BLE001  half-written heartbeat: skip this round
    return out


def queues():
    res = {}
    for q in sorted((R / 'ops/jobq').iterdir()):
        if not q.is_dir(): continue
        n = lambda d: sorted(x.name for x in (q / d).glob('*.sh')) if (q / d).is_dir() else []
        log = (q / 'jobq.log').read_text().strip().splitlines()[-3:] if (q / 'jobq.log').exists() else []
        res[q.name] = dict(running=n('running'), pending=n('pending'), done=len(n('done')), log=log)
    return res


def batches():
    try: b = json.loads((R / 'experiments/macro-tools-20261027/STATE.json').read_text()).get('batches', {})
    except Exception: return []  # noqa: BLE001
    rows = {k: dict(name=k, phase=v.get('phase'), jobs=v.get('jobs'), done=v.get('done'), running=len(v.get('running') or []), updated=v.get('updated_utc'), session=v.get('session')) for k, v in b.items()}
    # 三台主机的编排器同时整份重写同一个 STATE.json 会互相覆盖（10-30 c1o6t-h1 就丢了登记），所以再直接扫输出目录：
    # 最近 3 小时有更新的批次，作业数 = 子目录数，已完成 = 带 result.json 的子目录数
    od = R / 'experiments/macro-tools-20261027/out'
    for d in od.iterdir():
        try:
            if not d.is_dir() or time.time() - d.stat().st_mtime > 10800: continue
            sub = [x for x in d.iterdir() if x.is_dir()]; done = sum((x / 'result.json').exists() for x in sub)
        except OSError: continue
        r = rows.setdefault(d.name, dict(name=d.name, phase='（STATE 未登记，按输出目录统计）', jobs=None, done=None, running=None, updated=None, session=None))
        r.update(out_dirs=len(sub), out_done=done, updated=r['updated'] or time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(d.stat().st_mtime)))
    return sorted(rows.values(), key=lambda r: r['updated'] or '', reverse=True)[:12]


def quota():
    txt = subprocess.run("tmux capture-pane -p -t rsi-metadata -S -2000 2>/dev/null", shell=True, capture_output=True, text=True).stdout
    last, err = {}, 0
    for l in txt.splitlines():
        m = re.match(r'^(\d\d:\d\d:\d\d) (\S+) (\S+) \[(.*)\]$', l.strip())
        if m: last[m.group(2)] = dict(time=m.group(1), status=m.group(3), used=[float(x) for x in m.group(4).split(',') if x.strip() not in ('', 'None')])
        elif 'JSONDecodeError' in l: err += 1
    return dict(accounts=last, errors_in_pane=err, note='usedPercent per quota window, refreshed every 20 min by ops/metadata_loop.sh; native_cli has no readable metadata')


def jl(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None  # noqa: BLE001


def lines():
    out = []
    c1 = jl(R / 'experiments/macro-tools-20261027/c1/STATE.json')
    if c1:
        r = c1['rounds'][-1]; out.append(dict(line='C1 执行器', updated=c1.get('updated_utc'), now='%s：%s' % (r.get('round'), r.get('status')), detail=str(r.get('result') or r.get('change'))[:300], next=str(c1.get('next', ''))[:300]))
    for f in sorted((R / 'experiments/kpfactory-20261029').glob('STATE-*.json')):
        d = jl(f)
        if d: out.append(dict(line='KPNet ' + f.stem[6:], updated=d.get('updated_utc'), now=d.get('stage'), detail=str(d.get('note'))[:300], next=''))
    rr = jl(R / 'experiments/macro-tools-20261027/router-rgb/STATE.json')
    if rr: out.append(dict(line='路由器 Track D', updated=rr.get('updated_utc') or rr.get('updated'), now=str(rr.get('stage') or rr.get('status') or '')[:120], detail=str(rr.get('result') or rr.get('summary') or '')[:300], next=str(rr.get('next', ''))[:300]))
    tc = jl(R / 'experiments/ee-physword-20261029/STATE.json')
    if tc: out.append(dict(line='Track C（rsiws-dc）', updated=tc.get('updated_utc'), now=str(tc.get('status') or tc.get('stage') or '见 TRACK-C.md')[:120], detail=str(tc.get('summary') or '')[:300], next=str(tc.get('next') or '')[:300]))
    return out


def milestones():
    p = R / 'reports/progress/MILESTONES-20261030.md'
    if not p.exists(): return []
    rows = []
    for l in p.read_text().splitlines():
        c = [x.strip() for x in l.strip().strip('|').split('|')]
        if len(c) >= 2 and re.match(r'^\d\d:\d\d$', c[0]): rows.append(dict(time=c[0], milestone=c[1], items=c[2:6]))
    return rows


def h200():
    """H200 has no shared disk and only dispatcher / C line / liaison may send relay commands, so the board shows what the relay already has:
    recent commands (outbox/done + inbox results, output after the login banner), tmux sessions seen in those outputs, and mirror file times."""
    rl = R / 'ops/relay-h200'; cmds, tmux_seen = [], {}
    for f in sorted((rl / 'outbox/done').glob('*.json'))[-10:]:
        c = jl(f) or {}; res = jl(rl / 'inbox' / f.name) or {}; out = str(res.get('stdout', '')); out = out[out.rfind('╯') + 1:].strip()
        for m in re.finditer(r'^(\S+): (\d+) windows? \(created ([^)]*)\)', out, re.M): tmux_seen[m.group(1)] = dict(created=m.group(3), seen=c.get('created_at'))
        cmds.append(dict(id=c.get('id', f.stem), frm=c.get('from'), utc=c.get('created_at'), cmd=str(c.get('cmd', ''))[:160], rc=res.get('rc'), out=out[-300:]))
    mirror = []
    for f in (rl / 'from-h200').rglob('*'):
        if f.is_file(): mirror.append(dict(file=str(f.relative_to(rl / 'from-h200')), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(f.stat().st_mtime))))
    hb = jl(rl / 'from-h200/action-preview/receipts/h200_heartbeat.json')
    return dict(commands=cmds[::-1], tmux_seen=tmux_seen, mirror=sorted(mirror, key=lambda x: x['utc'], reverse=True)[:6], heartbeat=hb)


def collect():
    s = dict(generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), hosts=hosts(), queues=queues(), batches=batches(), quota=quota(),
             lines=lines(), milestones=milestones(), h200=h200(), line_names=LINES)
    WWW.mkdir(parents=True, exist_ok=True); tmp = WWW / 'status.json.tmp'; tmp.write_text(json.dumps(s, ensure_ascii=False)); os.replace(tmp, WWW / 'status.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--loop', action='store_true'); a = ap.parse_args()
    while True:
        try: collect()
        except Exception as e: print(time.strftime('%H:%M:%S'), 'collect failed:', repr(e), flush=True)  # noqa: BLE001  keep serving the last good snapshot
        if not a.loop: break
        time.sleep(30)
