"""C1 (10-31): write the jobq items for the recorder smoke (h1), the gated expert-demo re-run (aux/h1/h2) and the merge (h1)."""
V = '/lizekai/RSIws/experiments/macro-tools-20261027'; Q = '/lizekai/RSIws/ops/jobq'; PY = '/opt/PhysRSI/ENV/robodojo/bin/python'
OK, FAIL = V + '/c1/out/RECORD-SMOKE-OK', V + '/c1/out/RECORD-SMOKE-FAIL'
smoke = f"""# C1 recorder smoke (1 state): vs_plug --record 1 after the server-name fix; writes RECORD-SMOKE-OK only if labels.jsonl has rows
cd {V}; rm -f {OK} {FAIL}
{PY} -B src/mt_orch3.py --name c1ert --jobs c1/jobs-ert-smoke.json --gpus 0 --per-gpu 1
n=$(cat out/c1ert/*/labels.jsonl 2>/dev/null | wc -l); echo "labels rows: $n"
if [ "$n" -gt 0 ]; then echo $n > {OK}; else echo $n > {FAIL}; fi
"""
open(Q + '/h1-c1/pending/E50-record-smoke.sh', 'w').write(smoke)
gpus = dict(aux=('0,1,2', 3), h1=('0,1,2,3', 3), h2=('0,1,2,3,4', 2))
for h, (g, p) in gpus.items():
    for k in (0, 1, 2):
        s = f"""# C1 expert demonstrations, re-run with the fixed recorder (perturb {k}, this host's third of the 83 handover states); starts only after the
# 1-state recorder smoke wrote rows (RECORD-SMOKE-OK); stops if the smoke failed
until [ -f {OK} ] || [ -f {FAIL} ]; do sleep 60; done; [ -f {OK} ] || exit 3
cd {V}; {PY} -B src/mt_orch3.py --name c1er{k}-{h} --jobs c1/jobs-er{k}-{h}.json --gpus {g} --per-gpu {p}
"""
        open(Q + '/%s-c1/pending/E6%d-expert-demo-er%d-%s.sh' % (h, k, k, h), 'w').write(s)
merge = f"""# C1: merge the re-run expert demonstrations -> c1/out/expert-demos (MANIFEST.json only if positive_rows > 0, else MANIFEST-EMPTY.json)
cd {V}; [ -f {FAIL} ] && exit 3
for k in 0 1 2; do for h in aux h1 h2; do until [ -f logs/c1er$k-$h.done ]; do sleep 120; done; done; done
python3 c1/tools/expert_demo_aggregate.py c1/out/expert-demos c1er0 c1er1 c1er2
"""
open(Q + '/h1-c1/pending/E69-expert-demos-aggregate.sh', 'w').write(merge)
print('written')
