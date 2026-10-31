"""C1 P3 (10-31): merge DAgger expert labels of a BC rollout round.  python dagger_aggregate.py ORCH_PREFIX OUTDIR
Input: out/<ORCH_PREFIX>-*/<job>/labels.jsonl (bc_rollout.py) + result.json (rollout outcome).
Output: OUTDIR/labels.jsonl (one visited state per line: frames paths, state 14, eef 2x7, instruction, expert_phase, expert_joint (H, 14),
expert_eef (H, 16), episode, step, rollout outcome) and OUTDIR/MANIFEST.json (counts by phase / outcome). Wrong 'release' labels (10-31 fix) become
'hold_closed_relabel' (gripper kept at its measured value; arm held). Rows whose expert phase is
'charger_not_held' carry no action (no grasp expert yet) and are kept only as failure states."""
import collections, glob, json, os, sys
V = '/lizekai/RSIws/experiments/macro-tools-20261027'
pre, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); FIX = '--no-fix' not in sys.argv; fixed = [0]
ph, oc, n = collections.Counter(), collections.Counter(), 0
with open(os.path.join(out, 'labels.jsonl'), 'w') as w:
    for d in sorted(glob.glob(V + '/out/%s-*/*' % pre)):
        try: r = json.load(open(d + '/result.json'))
        except Exception: continue
        o = (r.get('bc') or {}).get('outcome'); oc[o] += 1
        if not os.path.exists(d + '/labels.jsonl'): continue
        for line in open(d + '/labels.jsonl'):
            x = json.loads(line)
            if x['expert_phase'] == 'release' and FIX:   # 10-31 expert fix applied offline: release only when all checks pass or depth + dxy <= 3 mm
                info = x.get('expert_info') or {}; ck = info.get('checks') or [0, 0, 0]; dxy = (info.get('residual') or {}).get('dxy_mm', 1e9)
                if not (min(ck) >= 1.0 or (ck[0] >= 1.0 and dxy <= 3.0)):
                    gi = 6 if x.get('hold_arm') == 'left' else 13; ge = 7 if x.get('hold_arm') == 'left' else 15; g = float(x['state'][gi])
                    for row in x.get('expert_joint') or []: row[gi] = g      # keep the gripper closed (as measured); arm target unchanged (hold)
                    for row in x.get('expert_eef') or []: row[ge] = g
                    x['expert_phase'] = 'hold_closed_relabel'; fixed[0] += 1
            x['rollout_outcome'] = o; x['job'] = os.path.basename(d); ph[x['expert_phase']] += 1; n += 1; w.write(json.dumps(x) + '\n')
json.dump(dict(rows=n, release_relabelled=fixed[0], by_phase=ph, rollout_outcomes=oc, source=pre, note='DAgger it1 relabels; truth used only for the expert labels'), open(os.path.join(out, 'MANIFEST.json'), 'w'), indent=1)
print(n, dict(ph), dict(oc))
