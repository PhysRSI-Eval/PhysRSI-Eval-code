"""C1 (10-31): collect truth-keypoint VS executor demonstrations (vs_plug.py --record 1) into an action-chunk dataset.
  python expert_demo_aggregate.py OUTDIR PREFIX [PREFIX ...]      (PREFIX = mt_orch3 batch-name prefix, e.g. c1ed0)
OUTDIR/labels.jsonl    : every recorded act boundary of the SUCCESSFUL rollouts (outcome inserted), i.e. the whole executed segment
                         from the handover to the insertion = "correct trajectories" (frames, state 14, hold_arm, expert_joint (H, 14)
                         = the executed next-H native actions, expert_phase = executor act reason, episode, step, perturb);
OUTDIR/negatives.jsonl : the same rows of the failed rollouts (router negatives), with the failure outcome;
OUTDIR/MANIFEST.json   : counts per outcome / perturbation, rows, source batches."""
import collections, glob, json, os, sys
V = '/lizekai/RSIws/experiments/macro-tools-20261027'
out, pres = sys.argv[1], sys.argv[2:]; os.makedirs(out, exist_ok=True)
oc, n_pos, n_neg, eps = collections.Counter(), 0, 0, collections.Counter()
with open(os.path.join(out, 'labels.jsonl'), 'w') as wp, open(os.path.join(out, 'negatives.jsonl'), 'w') as wn:
    for pre in pres:
        for d in sorted(glob.glob(V + '/out/%s-*/*' % pre)):
            try: r = json.load(open(d + '/result.json'))
            except Exception: continue
            v = r.get('vs') or {}; o = v.get('outcome'); oc[o] += 1
            if not os.path.exists(d + '/labels.jsonl'): continue
            good = o == 'inserted'; x = {}   # labels.jsonl 可能为空（未录到动作边界），避免下面引用未定义的 x
            for line in open(d + '/labels.jsonl'):
                x = json.loads(line); x['rollout_outcome'] = o; x['job'] = os.path.basename(d); x['steps_total'] = v.get('steps')
                (wp if good else wn).write(json.dumps(x) + '\n')
                if good: n_pos += 1
                else: n_neg += 1
            if good: eps[x.get('perturb', 0)] += 1
json.dump(dict(positive_rows=n_pos, negative_rows=n_neg, rollout_outcomes=oc, successful_rollouts_by_perturb=eps, batches=pres,
               note='truth-keypoint VS executor (o7 code) demonstrations from Astra-hold handover states; truth used by the expert only'),
          open(os.path.join(out, 'MANIFEST.json' if n_pos > 0 else 'MANIFEST-EMPTY.json'), 'w'), indent=1)   # downstream fires on MANIFEST.json only
print(n_pos, n_neg, dict(oc))
sys.exit(0 if n_pos > 0 else 2)   # 无正样本时非零退出，让队列日志显式暴露失败（下游只认 MANIFEST.json）
