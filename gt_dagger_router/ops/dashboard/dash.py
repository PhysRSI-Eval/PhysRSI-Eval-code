#!/usr/bin/env python3
"""每 3 分钟生成一次 reports/progress/DASHBOARD.md：各主机显卡利用率 + 每条线队列当前在跑什么、进度 n/N、最近结果。
只读，不改任何实验文件；不消耗 token（替代"问一次子代理才知道进度"）。"""
import json, glob, os, re, subprocess, time, collections
R = '/lizekai/RSIws'; V = R + '/experiments/macro-tools-20261027'; OUT = R + '/reports/progress/DASHBOARD.md'
# 主机地址和 ssh 私钥不进仓库：用环境变量 DASH_HOST1/DASH_HOST2/DASH_AUX、DASH_SSH_KEY 提供
HOSTS = [('main', None), ('host1', os.environ.get('DASH_HOST1')), ('host2', os.environ.get('DASH_HOST2')), ('aux', os.environ.get('DASH_AUX'))]
LINE = {'c1': 'C1 执行器', 'kp': 'KPNet 工具', 'rt': '路由器'}
def sh(cmd, host=None):
    if host: cmd = ['timeout', '20', 'ssh', '-o', 'ConnectTimeout=8', '-i', os.environ.get('DASH_SSH_KEY', ''), 'root@' + host, cmd]
    else: cmd = ['bash', '-c', cmd]
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e: return 'ERR ' + str(e)
def outcome(f):
    try: d = json.load(open(f))
    except Exception: return 'unreadable'
    vs = d.get('vs') if isinstance(d.get('vs'), dict) else {}
    return str(vs.get('outcome') or vs.get('result') or vs.get('status') or d.get('failure'))[:30]
def progress(script):
    # mt_orch3 作业：从 --name/--jobs 推出 out/<name>/ 的完成数和结果分布
    t = open(script).read(); m = re.search(r'--name (\S+) --jobs (\S+)', t)
    if not m: return ''
    name, jobs = m.groups(); n = len(json.load(open(os.path.join(V, jobs)))) if os.path.exists(os.path.join(V, jobs)) else '?'
    c = collections.Counter(outcome(f) for f in glob.glob(f'{V}/out/{name}/*/result.json'))
    return f'{sum(c.values())}/{n} ' + ', '.join(f'{k} {v}' for k, v in c.most_common())
while True:
    L = [f'# 实验看板（自动生成 {time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())}，每 3 分钟刷新）', '', '## 显卡利用率（%）', '| 主机 | GPU0..7 | 显存已用 (GB) |', '|---|---|---|']
    for h, ip in HOSTS:
        s = sh('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits', ip)
        rows = [r.split(',') for r in s.splitlines() if ',' in r]
        L.append(f'| {h} | ' + ' '.join(r[0].strip() for r in rows) + ' | ' + ' '.join(str(round(int(r[1]) / 1024)) for r in rows) + ' |')
    L += ['', '## 各线程队列（ops/jobq）', '| 线程 | 队列 | 在跑 | 进度 | 排队中 | 最近完成 |', '|---|---|---|---|---|---|']
    for q in sorted(glob.glob(R + '/ops/jobq/*/')):
        k = os.path.basename(q.rstrip('/')); line = LINE.get(k.split('-')[-1], k)
        run = sorted(glob.glob(q + 'running/*.sh')); pend = [os.path.basename(p) for p in sorted(glob.glob(q + 'pending/*.sh'))]
        last = (open(q + 'jobq.log').read().strip().splitlines() or [''])[-1] if os.path.exists(q + 'jobq.log') else ''
        L.append(f'| {line} | {k} | {", ".join(os.path.basename(r) for r in run) or "（空闲）"} | {"; ".join(progress(r) for r in run)} | {", ".join(pend) or "—"} | {last} |')
    st = R + '/ops/relay-h200/inbox'
    L += ['', '## H200', '- 阶段模型对比在 tmux rt-stage-g0/g1（GPU0–1）；结果按需 pull，不在此轮询。']
    open(OUT + '.tmp', 'w').write('\n'.join(L) + '\n'); os.replace(OUT + '.tmp', OUT)
    time.sleep(180)
