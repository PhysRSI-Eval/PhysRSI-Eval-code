"""Router side of the DAgger loop (GT-DAGGER-PLAN 'Router side'): turn a C1 aggregate (c1/tools/dagger_aggregate.py OUTDIR/labels.jsonl,
one policy-visited state per line with frames / state / eef / expert_phase / rollout_outcome) into ROUTER-BIG rows and merge with big_v3.
  * V: router_rgb.dino_features of the 3 logged frames -> pca_v2 (same as big_v2 / v3);  P: build_big_v2.prop_from convention from
    state gripper readings (state[6], state[13]; native state = official state) and eef z / xy per arm, over the rollout's steps.
  * stage (for the failure-per-stage head): expert phase align / insert -> 4 (align), release -> 5, charger_not_held -> -1 (unknown).
  * succ: rollout outcome inserted / native_end_success -> 1, budget / charger_dropped / native_end -> 0, error -> nan.
  * recovery label (take, first state of each rollout only): BC (3) if the BC rollout succeeded, else the fitmap v1 outcome of the same
    handover state (episode in fitmap 'eps', nearest t <= the first step): rg3f (1), VS (2), Astra (0); none -> Astra (fitmap convention).
    Priority = deployable first (BC, rg3f), then VS (truth-keypoint ceiling), then Astra.
  * split: the v3 split of the same group (ctx) when the ctx is already in big_v3, else 80 / 10 / 10 by a stable hash of the ctx.
python ingest_dagger.py TAG BASE.npz C1_OUTDIR [C1_OUTDIR ...] -> big_dagger_TAG.npz (+ per-row step 'tstep', per-rollout 'drop_step') +
  big_TAG.npz (= BASE + these rows).  Phases: align_transport / hold_closed_relabel (*-fixed) -> transport if the expert residual dxy > 5 cm else align; align / insert -> 4, release -> 5."""
import collections, hashlib, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
H = Path(__file__).resolve().parent; RR = H.parent; sys.path.insert(0, str(RR)); import router_rgb as R   # noqa: E402
from transformers import AutoModel
tag, base, src_dirs = sys.argv[1], sys.argv[2], [Path(x) for x in sys.argv[3:]]
rows = [json.loads(l) for d in src_dirs for l in open(d / 'labels.jsonl')]; jobs = collections.defaultdict(list)
for r in rows: jobs[r['job']].append(r)
PL = json.load(open(H / 'tasks_v2.json')).index('plug_in_charger')
pca = np.load(H / 'pca_v2.npz'); mu, W = pca['mu'], pca['W']; V3 = np.load(H / base, allow_pickle=True); V3 = {k: V3[k] for k in V3.files}; gsplit = dict(zip(V3['group'], V3['split']))   # each npz key read once
FM = [e for e in json.load(open(RR / 'fitmap/fitmap.json')) if e.get('vs') is not None]
dev = torch.device('cuda:0'); vit = AutoModel.from_pretrained(R.DINO).to(dev).eval(); nreg = vit.config.num_register_tokens
ST = {'align': 4, 'insert': 4, 'release': 5}; OK = {'inserted', 'native_end_success'}; BAD = {'budget', 'charger_dropped', 'native_end'}
GEO = {'align_transport', 'hold_closed_relabel'}   # by geometry (the stage definition): expert residual dxy > 5 cm -> 3 transport, else 4 align
def stage_of(r):
    if r['expert_phase'] in GEO: return 3 if (r.get('expert_info') or {}).get('residual', {}).get('dxy_mm', 1e9) > 50 else 4
    return ST.get(r['expert_phase'], -1)
def prop(g, z, xy, t):
    out = []
    for gi, zi, xyi in zip(g, z, xy):
        L = len(gi); closed = gi < 0.9; since = np.full(L, -1.0); zc = np.zeros(L); cur = None
        for k in range(L):
            if closed[k] and cur is None: cur = k
            if not closed[k]: cur = None
            since[k] = (t[k] - t[cur]) / 100.0 if cur is not None else -1.0; zc[k] = (zi[k] - zi[cur]) * 10 if cur is not None else 0.0
        dz = np.r_[0, np.diff(zi)] * 100 / np.maximum(np.r_[1, np.diff(t)], 1); dxy = np.r_[0, np.linalg.norm(np.diff(xyi, axis=0), axis=1)] * 100 / np.maximum(np.r_[1, np.diff(t)], 1)
        out.append(np.c_[gi, (~closed).astype(float), since, zi, dz, dxy, zc])
    return np.concatenate(out, 1).astype(np.float32)
out = dict(V=[], P=[], task=[], stage=[], prog=[], succ=[], take=[], src=[], dev=[], group=[], split=[]); extra = dict(tstep=[], drop_step=[], outcome=[], job=[]); stats = collections.Counter(); match_dt = []
for job, rs in sorted(jobs.items()):
    rs = sorted(rs, key=lambda r: r['step']); X = []
    for r in rs:
        ims = {c: np.asarray(Image.open(p).convert('RGB')) for c, p in r['frames'].items()}; X.append(R.dino_features(vit, nreg, ims, dev).reshape(-1))
    V = ((np.asarray(X, np.float32) - mu) @ W).astype(np.float16)
    S = np.asarray([r['state'] for r in rs], float); E = np.asarray([r['eef'] for r in rs], float); t = np.asarray([r['step'] for r in rs], float)
    P = prop([S[:, 6], S[:, 13]], [E[:, 0, 2], E[:, 1, 2]], [E[:, 0, :2], E[:, 1, :2]], t)
    o = rs[0].get('rollout_outcome'); succ = 1.0 if o in OK else 0.0 if o in BAD else np.nan; stats[o] += 1
    take = np.full(len(rs), np.nan)
    if succ == 1.0: take[0] = 3
    else:
        c = [e for e in FM if rs[0]['episode'] in (e.get('eps') or [e['episode']]) and e['t'] <= rs[0]['step']]
        if c:
            e = max(c, key=lambda e: e['t']); match_dt.append(rs[0]['step'] - e['t']); take[0] = 1 if e.get('rg3f') else 2 if e['vs'] else 0
    ctx = '-'.join(rs[0]['episode'].split('-')[:3]); sp = int(gsplit[ctx]) if ctx in gsplit else (lambda h: 0 if h < 8 else 1 if h < 9 else 2)(int(hashlib.md5(ctx.encode()).hexdigest(), 16) % 10)
    for k, v in (('V', V), ('P', P), ('task', PL), ('stage', np.array([stage_of(r) for r in rs], np.int8)), ('prog', np.full(len(rs), np.nan, np.float32)),
                 ('succ', succ), ('take', take.astype(np.float32)), ('src', 'dagger_' + tag), ('dev', sp != 0), ('group', ctx), ('split', sp)): out[k].append(v)
    extra['tstep'].append(t.astype(np.int32)); extra['drop_step'].append(next((r['step'] for r in rs if r['expert_phase'] == 'charger_not_held'), -1)); extra['outcome'].append(str(o)); extra['job'].append(job)
np.savez_compressed(H / ('big_dagger_%s.npz' % tag), **{k: np.array(v, dtype=object) for k, v in {**out, **extra}.items()})
M = {k: np.concatenate([V3[k].astype(object), np.array(out[k], dtype=object)]) for k in V3}; M['split'] = M['split'].astype(np.int8)
np.savez_compressed(H / ('big_%s.npz' % tag), **M)
lab = collections.Counter(int(x[0]) for x in out['take'] if not np.isnan(x[0]))
print(json.dumps(dict(rollouts=len(jobs), states=len(rows), outcomes=stats, recovery_labels={['Astra', 'rg3f', 'VS', 'BC'][k]: v for k, v in lab.items()},
                      fitmap_match_dt=dict(n=len(match_dt), max=max(match_dt) if match_dt else None), split=collections.Counter(out['split']))))
