"""keypoint factory, insert_key labels from rsiws-dc's astra_v1 export (real Astra episodes, exact replays; read-only). No sim needed:
per kept step (every --stride steps) and camera (cam_head, cam_left_wrist, cam_right_wrist; 320x240 frames), the 8 finepose keypoints
(key bbox-centre frame: tip, tooth1, tooth2, blade_root, bow_centre; lock frame: keyhole centre, slot end, far rim) are projected with
the stored per-step cameras (wrist = FK link6 x mount, checked by the exporter; intrinsics scaled 640 -> 320). Visibility = in image AND
not behind the key / lock in a CAD z-buffer (finepose cad_render, read-only copy; fingers not modelled). Frames with no visible point
are dropped. Split by episode: sha1 % 5 == 4 TEST, == 3 DEV (split 1), else TRAIN (0) -> <out>/<cam>/{trdev,test}/L000-s0.jsonl.
  python make_labels_key_astra.py --out kp-key --stride 4 --procs 48"""
import argparse, glob, hashlib, json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/lizekai/RSIws/experiments/finepose-20261028/tools')
import cad_render as CR  # noqa: E402  (read-only; sha in FINEPOSE-REUSE.sha256)
A = Path('/lizekai/RSIws/experiments/mva-20261029/astra_v1'); KEY_KP = np.array([[-0.0021, 0, -0.0314], [0.0106, 0, -0.0258], [0.0106, 0, -0.0150], [-0.0021, 0, 0.0114], [-0.0021, 0, 0.0214]])
NAMES = ['tip', 'tooth1', 'tooth2', 'blade_root', 'bow_centre', 'keyhole_centre', 'keyhole_slot_end', 'keyhole_far_rim']; CAMS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')


def T_of(p, q): M = np.eye(4); M[:3, :3] = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); M[:3, 3] = p; return M


def one(args):
    ep, stride = args; tr = json.load(open(A / 'truth' / 'insert_key' / (ep + '.json'))); z = np.load(A / 'arrays' / 'insert_key' / (ep + '.npz')); out = []
    sp = 'test' if int(hashlib.sha1(ep.encode()).hexdigest(), 16) % 5 == 4 else 'trdev'; spi = 1 if int(hashlib.sha1(ep.encode()).hexdigest(), 16) % 5 == 3 else 0
    for st in sorted(tr, key=int):
        s = int(st)
        if s % stride or s >= z['T']: continue
        o = tr[st]; kn = next((n for n in o if n.startswith('key') and 'slot' not in n and 'position' in o[n]), None); ln = next((n for n in o if 'slot' in n and 'position' in o[n]), None)
        if kn is None or ln is None: continue
        Ta = T_of(o[kn]['position'], o[kn]['quaternion_wxyz']); Kp = Ta.copy(); Kp[:3, 3] = Ta[:3, :3] @ CR.KEY_CL + Ta[:3, 3]; Tl = T_of(o[ln]['position'], o[ln]['quaternion_wxyz'])
        P = np.vstack([KEY_KP @ Kp[:3, :3].T + Kp[:3, 3], CR.KEYHOLE_KP @ Tl[:3, :3].T + Tl[:3, 3]])
        for cam in CAMS:
            img = A / 'frames' / 'insert_key' / ep / ('%04d-%s.jpg' % (s, cam))
            if not img.exists(): continue
            Kin = np.asarray(z['%s/intrinsic' % cam], float).copy(); Kin[:2] *= 0.5; Tc = np.asarray(z['%s/extrinsic' % cam][s], float)
            uv, zz = CR._cam(Kin, Tc, P); lab, dep = CR.render(Kin, Tc, [(1, CR.lock_world(Tl), CR.LOCK['faces']), (2, CR.key_world(Kp), CR.KEY['faces'])], W=320, H=240); vis = []
            for (u, v), d in zip(uv, zz):
                ok = bool(d > 0.02 and 0 <= u < 320 and 0 <= v < 240)
                if ok: ok = bool(dep[int(v), int(u)] >= d - 0.003)
                vis.append(ok)
            if not any(vis): continue
            out.append((cam, sp, dict(sample_id='%s-%04d-%s' % (ep, s, cam), image='%s/%04d-%s.jpg' % (ep, s, cam), layout=0, seed=spi, split=spi, camera=cam, episode=ep, step=s,
                                      intrinsics=Kin.tolist(), extrinsics_cam_to_world=Tc.tolist(), outlets=P.tolist(), outlets_uv=uv.tolist(), in_image=vis, keypoint_names=NAMES)))
    return out


def main(a):
    eps = [json.loads(l) for l in open(A / 'MANIFEST.jsonl')]; eps = [e['episode'] for e in eps if e.get('task') == 'insert_key' and e.get('status') == 'used' and (A / 'truth' / 'insert_key' / (e['episode'] + '.json')).exists()]
    out = Path(a.out); files = {}
    with Pool(a.procs) as p:
        for res in p.imap_unordered(one, [(e, a.stride) for e in eps]):
            for cam, sp, r in res: files.setdefault((cam, sp), []).append(r)
    for (cam, sp), rows in files.items():
        d = out / cam / sp; d.mkdir(parents=True, exist_ok=True)
        if not (d / 'images').exists(): os.symlink(str(A / 'frames' / 'insert_key'), d / 'images')
        with open(d / 'L000-s0.jsonl', 'w') as g:
            for r in rows: g.write(json.dumps(r) + '\n')
    print(json.dumps({'%s/%s' % k: len(v) for k, v in sorted(files.items())}), 'episodes', len(eps))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--stride', type=int, default=4); ap.add_argument('--procs', type=int, default=48); main(ap.parse_args())
