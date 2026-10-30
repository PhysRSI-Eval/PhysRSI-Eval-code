"""K1 stage labels + features (router v4 step 2, dev). Episodes: astra_v1 K1 tasks. Every 4th step with truth:
  held  = gripper reading < 0.85 and an insertable rigid object within 5 cm of that arm's TCP (rsiws-dc E1 rule)
  stage = 0 approach (not held, nothing placed yet / re-attempt after a drop), 1 grasp (held, object raised < 1 cm), 2 lift (1-3 cm),
          3 transport (>= 3 cm, object > 5 cm from the target in xy), 4 align (held, <= 5 cm), 5 inserted/placed (not held, the last
          handled object within 3 cm of the target in xy).
Truth only for labels. Features = router_rgb.dino_features of the 640x480 native frames (head, left, right).
  python -I build_stage.py SHARD NSHARD GPU -> feat/shard-k.npz"""
import json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import router_rgb as RR   # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'e1copy')); from train_k2 import Tm, TCP   # noqa: E402
A = Path('/lizekai/RSIws/experiments/mva-20261029/astra_v1'); NAT = Path('/lizekai/RSIws/artifacts/native')
K1 = {'plug_in_charger': (('charger',), 'socket'), 'insert_key': (('key_0',), 'key_slot'), 'deposit_coin': (('coin',), 'piggy_bank'),
      'fill_pen_holder': (('pen', 'oil_pen'), 'pen_holder'), 'insert_tubes': (('test_tube_2',), 'test_tube_slot')}
EPS = [json.loads(l) for l in open(A / 'MANIFEST.jsonl')]; EPS = [m for m in EPS if m.get('valid') and m['task'] in K1]
k, n, g = map(int, sys.argv[1:4]); mine = EPS[k::n]
from transformers import AutoModel
dev = torch.device('cuda:%d' % g); vit = AutoModel.from_pretrained(RR.DINO).to(dev).eval(); nreg = vit.config.num_register_tokens
X, Y, META = [], [], []
for m in mine:
    task, ep = m['task'], m['episode']; ins, tgt = K1[task]
    d = np.load(A / 'arrays' / task / ('%s.npz' % ep)); tr = json.load(open(A / 'truth' / task / ('%s.json' % ep)))
    inner = [p for p in (NAT / ep).iterdir() if p.is_dir() and (p / 'action_000000.json').exists()][0]; obs = inner / 'observations'
    steps = sorted(int(s) for s in tr)
    first = tr[str(steps[0])]; z0 = {nm: v['position'][2] for nm, v in first.items() if v.get('kind') == 'rigid' and any(nm.startswith(p) for p in ins)}
    last_obj = None; seq_x, seq_y, seq_t = [], [], []
    for t in steps:
        if t % 4 or t >= int(d['T']): continue
        o = tr[str(t)]; T_ = [v['position'] for nm, v in o.items() if nm.startswith(tgt) and 'position' in v]
        if not T_: continue
        tg = np.asarray(T_[0]); held_obj = None
        for arm in ('left', 'right'):
            e = d['state/%s_ee_poses' % arm][t]; E = Tm(e[:3], e[3:7]); tcp = E[:3, :3] @ TCP + E[:3, 3]
            if d['state/%s_ee_joint_states' % arm][t, 0] >= 0.85: continue
            for nm, v in o.items():
                if nm in z0 and 'position' in v and np.linalg.norm(np.asarray(v['position']) - tcp) < 0.05: held_obj = nm
        if held_obj:
            last_obj = held_obj; p = np.asarray(o[held_obj]['position']); dz = p[2] - z0[held_obj]; dxy = np.linalg.norm(p[:2] - tg[:2])
            st = 1 if dz < 0.01 else 2 if dz < 0.03 else (4 if dxy <= 0.05 else 3)
            if dz >= 0.01 and dxy <= 0.05: st = 4
        else:
            st = 5 if (last_obj and 'position' in o.get(last_obj, {}) and np.linalg.norm(np.asarray(o[last_obj]['position'])[:2] - tg[:2]) <= 0.03) else 0
        fs = [obs / ('%06d-%s.png' % (t, c)) for c in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')]
        if not all(f.exists() for f in fs): continue
        ims = {c: np.asarray(Image.open(f).convert('RGB')) for c, f in zip(('cam_high', 'cam_left_wrist', 'cam_right_wrist'), fs)}
        seq_x.append(RR.dino_features(vit, nreg, ims, dev).astype(np.float16)); seq_y.append(st); seq_t.append(t)
    X.append(np.stack(seq_x) if seq_x else np.zeros((0, 3, 1536), np.float16)); Y.append(np.asarray(seq_y, np.int8))
    META.append(dict(task=task, episode=ep, ctx='-'.join(ep.split('-')[:3]), layout=m['layout'], steps=seq_t, native_success=m.get('native_success')))
    print(k, ep, len(seq_y), np.bincount(seq_y, minlength=6).tolist() if seq_y else [], flush=True)
np.savez_compressed(Path(__file__).parent / 'feat' / ('shard-%02d.npz' % k), X=np.array(X, dtype=object), Y=np.array(Y, dtype=object), allow_pickle=True)
json.dump(META, open(Path(__file__).parent / 'feat' / ('shard-%02d.json' % k), 'w')); print('done', k, flush=True)
