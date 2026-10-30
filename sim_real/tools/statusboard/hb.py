"""Host heartbeat for the status board (rsiws-dc, 2026-10-30). Writes ops/statusboard/hosts/<label>.json every 60 s:
GPUs (util, memory, compute-process count), host memory, load, tmux sessions grouped by line prefix, simulator process count.
Runs on main directly (collect.py imports snapshot()) and on host1/host2/aux via their ops/jobq queue (detached tmux statusboard-hb).
  python3 hb.py --label h1 [--loop]"""
import argparse, json, os, socket, subprocess, time
from pathlib import Path

OUT = Path('/lizekai/RSIws/ops/statusboard/hosts')
SIM_KEYS = ('ip3_server_ext', 'c1_server_ext', 'replay_server', 'gpd_server_ext', 'pe_server_ext', 'cp_server_ext', 'wx_server_ext')


def sh(cmd):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout
    except Exception: return ''  # noqa: BLE001  a missing tool on one host must not stop the heartbeat


def snapshot(label):
    gpus = []
    uuid = {}
    for l in sh('nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits').splitlines():
        p = [x.strip() for x in l.split(',')]
        if len(p) == 5: uuid[p[1]] = int(p[0]); gpus.append(dict(index=int(p[0]), util=int(p[2]), mem_used=int(p[3]), mem_total=int(p[4]), procs=0))
    for l in sh('nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader').splitlines():
        i = uuid.get(l.strip())
        if i is not None: gpus[i]['procs'] += 1
    mem = sh("free -g | awk 'NR==2{print $2, $7}'").split()
    sessions = [s for s in sh("tmux ls -F '#{session_name}' 2>/dev/null").split() if s]
    groups = {}
    for s in sessions: groups.setdefault(s.split('-')[0], []).append(s)
    ps = sh('ps -eo args')
    sims = sum(1 for l in ps.splitlines() if any(k in l for k in SIM_KEYS) and 'python' in l)
    return dict(label=label, hostname=socket.gethostname(), time=time.time(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                gpus=gpus, mem_total_gb=int(mem[0]) if mem else None, mem_avail_gb=int(mem[1]) if len(mem) > 1 else None,
                load=os.getloadavg()[0], ncpu=os.cpu_count(), tmux=groups, n_tmux=len(sessions), sims=sims)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--label', required=True); ap.add_argument('--loop', action='store_true'); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    while True:
        tmp = OUT / ('%s.json.tmp' % a.label); tmp.write_text(json.dumps(snapshot(a.label))); os.replace(tmp, OUT / ('%s.json' % a.label))
        if not a.loop: break
        time.sleep(60)


if __name__ == '__main__':
    main()
