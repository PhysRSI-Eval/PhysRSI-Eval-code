"""D2 hold-wrist charger labels from astra_v1 plug episodes (Astra's own holds; read-only export, no sim). For every 2nd exported step
where the charger is held (its CAD centre within 6 cm of an arm's TCP = link6 + 0.14 x) and lifted (> 2 cm above its reset height):
the holding arm's wrist frame (320x240) with 13 charger points (prong tips, roots, face centre, 8 housing corners; charger CAD frame =
sim object frame) projected with the stored FK camera; visibility = in image AND not behind the charger in a CAD z-buffer (fingers not
modelled). Episodes in --exclude (the 20 FITMAP lift-abstain states) are left out. Split by episode: sha1 % 5 == 3 DEV, else TRAIN.
  python make_labels_hold_astra.py --out kp-hold --procs 64"""
import argparse, hashlib, json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/lizekai/RSIws/experiments/finepose-20261028/tools')
import cad_render as CR  # noqa: E402  (z-buffer rasteriser only)
A = Path('/lizekai/RSIws/experiments/mva-20261029/astra_v1'); CH = np.load('/lizekai/RSIws/experiments/kpfactory-20261029/cad/charger.npz')
CHV, CHF = CH['vertices'], CH['faces']
P13 = np.vstack([[[-0.0055, -0.0152, 0], [0.0055, -0.0152, 0], [-0.0055, 0, 0], [0.0055, 0, 0], [0, 0, 0]], [[x, y, z] for y in (0.0, 0.0479) for x in (-0.021, 0.021) for z in (-0.0135, 0.0135)]])
NAMES = ['tip_l', 'tip_r', 'root_l', 'root_r', 'face_c'] + ['corner_%d' % i for i in range(8)]


def T_of(p, q): M = np.eye(4); M[:3, :3] = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); M[:3, 3] = p; return M


def one(ep):
    tr = json.load(open(A / 'truth' / 'plug_in_charger' / (ep + '.json'))); z = np.load(A / 'arrays' / 'plug_in_charger' / (ep + '.npz')); out = []
    spi = 1 if int(hashlib.sha1(ep.encode()).hexdigest(), 16) % 5 == 3 else 0; z0 = None; ctr = (CHV.max(0) + CHV.min(0)) / 2
    for st in sorted(tr, key=int):
        s = int(st); o = tr[st]; cn = next((n for n in o if n.startswith('charger') and 'position' in o[n]), None)
        if cn is None or s >= z['T']: continue
        Tc = T_of(o[cn]['position'], o[cn]['quaternion_wxyz']); z0 = Tc[2, 3] if z0 is None else z0; cw = Tc[:3, :3] @ ctr + Tc[:3, 3]
        if Tc[2, 3] - z0 < 0.02: continue
        for arm in ('left', 'right'):
            ee = z['state/%s_ee_poses' % arm][s]; L = T_of(ee[:3], ee[3:7]); tcp = L[:3, :3] @ np.array([0.14, 0, 0]) + L[:3, 3]
            if np.linalg.norm(cw - tcp) > 0.06: continue
            cam = 'cam_%s_wrist' % arm; img = A / 'frames' / 'plug_in_charger' / ep / ('%04d-%s.jpg' % (s, cam))
            if not img.exists(): continue
            Kin = np.asarray(z['%s/intrinsic' % cam], float).copy(); Kin[:2] *= 0.5; Tk = np.asarray(z['%s/extrinsic' % cam][s], float)
            P = P13 @ Tc[:3, :3].T + Tc[:3, 3]; uv, zz = CR._cam(Kin, Tk, P); _, dep = CR.render(Kin, Tk, [(1, CHV @ Tc[:3, :3].T + Tc[:3, 3], CHF)], W=320, H=240)
            vis = [bool(d > 0.02 and 0 <= u < 320 and 0 <= v < 240 and dep[int(v), int(u)] >= d - 0.003) for (u, v), d in zip(uv, zz)]
            if sum(vis) < 2: continue
            G = np.linalg.inv(L) @ Tc
            out.append(dict(sample_id='%s-%04d-%s' % (ep, s, cam), image='%s/%04d-%s.jpg' % (ep, s, cam), layout=0, seed=spi, split=spi, camera=cam, episode=ep, step=s, arm=arm,
                            intrinsics=Kin.tolist(), extrinsics_cam_to_world=Tk.tolist(), outlets=P.tolist(), outlets_uv=uv.tolist(), in_image=vis, keypoint_names=NAMES, G=np.round(G, 6).tolist()))
    return out


def main(a):
    ex = set(json.load(open(a.exclude))) if a.exclude else set(); m = [json.loads(l) for l in open(A / 'MANIFEST.jsonl')]
    eps = [e['episode'] for e in m if e.get('task') == 'plug_in_charger' and e.get('status') == 'used' and e['episode'] not in ex and (A / 'truth' / 'plug_in_charger' / (e['episode'] + '.json')).exists()]
    d = Path(a.out) / 'trdev'; d.mkdir(parents=True, exist_ok=True)
    if not (d / 'images').exists(): os.symlink(str(A / 'frames' / 'plug_in_charger'), d / 'images')
    n = 0
    with Pool(a.procs) as p, open(d / 'L000-s0.jsonl', 'w') as g:
        for res in p.imap_unordered(one, eps):
            for r in res: g.write(json.dumps(r) + '\n'); n += 1
    print('episodes', len(eps), 'excluded', len(ex), 'samples', n)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--exclude', default=''); ap.add_argument('--procs', type=int, default=64); main(ap.parse_args())
