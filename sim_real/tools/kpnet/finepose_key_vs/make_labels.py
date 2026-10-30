"""finepose T5-CAD labels (10-29): for every record of a kp_gen data dir, render CAD labels (0 bg / 1 lock / 2 key) into masks/<image>.png and write
<out>/L*.jsonl with 8 keypoints = 5 key points (CAD) + 3 keyhole points (lock CAD: round-hole centre, slot end, far rim; visible only if not occluded
by the key/lock in the z-buffer). Lock pose = layout-0 constant (recorded). Robot fingers are not modelled.
  python make_labels.py <data dir> <out dir> [--procs 32]"""
import glob, json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).resolve().parent)); import cad_render as CR  # noqa: E402
T_LOCK = np.eye(4); T_LOCK[:3, :3] = R.from_quat([0, 0, 0.1311132311820984, 0.9913673996925354]).as_matrix(); T_LOCK[:3, 3] = [0.110232874751091, -0.19773045182228088, 0.7649999856948853]


def one(args):
    r, data, out = args
    lab, dep = CR.labels_for(r, T_LOCK); Image.fromarray(lab).save(Path(out) / 'masks' / (Path(r['image']).stem + '.png'))
    P = (T_LOCK[:3, :3] @ CR.KEYHOLE_KP.T).T + T_LOCK[:3, 3]; uv, z = CR._cam(np.asarray(r['intrinsics'], float), np.asarray(r['extrinsics_cam_to_world'], float), P); vis = []
    for (u, v), zz in zip(uv, z):
        ok = zz > 0 and 0 <= u < 640 and 0 <= v < 480
        if ok: iu, iv = int(u), int(v); ok = bool(dep[iv, iu] >= zz - 0.002)  # not hidden behind the key (keyhole lies on the lock top surface)
        vis.append(bool(ok))
    r2 = dict(r); r2['outlets'] = r['outlets'] + P.tolist(); r2['outlets_uv'] = r['outlets_uv'] + [[float(u), float(v)] if zz > 0 else None for (u, v), zz in zip(uv, z)]
    r2['in_image'] = r['in_image'] + vis; r2['keypoint_names'] = r.get('keypoint_names', []) + ['keyhole_centre', 'keyhole_slot_end', 'keyhole_far_rim']
    r2['mask'] = 'masks/' + Path(r['image']).stem + '.png'; r2['lock_pose'] = T_LOCK.tolist(); r2['key_px'] = int((lab == 2).sum()); r2['lock_px'] = int((lab == 1).sum()); return r2


def main():
    data, out = Path(sys.argv[1]), Path(sys.argv[2]); procs = int(sys.argv[sys.argv.index('--procs') + 1]) if '--procs' in sys.argv else 32
    (out / 'masks').mkdir(parents=True, exist_ok=True)
    if not (out / 'images').exists(): os.symlink(data.resolve() / 'images', out / 'images')
    for f in sorted(glob.glob(str(data / 'L*-s*.jsonl'))):
        dst = out / Path(f).name
        if dst.exists() and dst.stat().st_size > 0: continue
        recs = [json.loads(l) for l in open(f)]
        with Pool(procs) as pool: res = pool.map(one, [(r, str(data), str(out)) for r in recs])
        with open(dst, 'w') as g:
            for r in res: g.write(json.dumps(r) + '\n')
    print('labels done', len(glob.glob(str(out / 'L*.jsonl'))))


if __name__ == '__main__':
    main()
