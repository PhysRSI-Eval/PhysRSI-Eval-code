"""第 0 层的换场景检验数据（第 1 步，robodojo 环境，有 h5py）：从已下完的官方示范 HDF5（A 类 9 个任务，每任务 100 条、
场景随机化）每条均匀抽 K 个时刻的三路相机 JPEG 原始字节 -> data-official/frames/<task>/<episode>.npz（含相机内外参第 0 帧，
用于核对官方 cam_head 与 Astra cam_high 的视角是否一致）。只读官方数据。
  /opt/PhysRSI/ENV/robodojo/bin/python -B extract_official_frames.py [--procs 24]"""
import argparse, json, os
from multiprocessing import Pool
from pathlib import Path
import numpy as np

O = Path('/lizekai/RSIws/data/official'); OUT = Path('/lizekai/RSIws/experiments/ee-physword-20261029/data-official/frames')
CAMS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist'); K = 16


def one(f):
    import h5py
    p = O / f['relative_path']; od = OUT / f['task']; out = od / ('%07d.npz' % f['episode'])
    if out.exists(): return 'skip'
    try:
        with h5py.File(p, 'r') as h:
            T = h['vision/cam_head/colors'].shape[0]; steps = np.unique(np.linspace(0, T - 1, min(K, T)).round().astype(int))
            arr = {'%s/jpg%02d' % (c, i): np.frombuffer(bytes(h['vision/%s/colors' % c][t]), np.uint8) for c in CAMS for i, t in enumerate(steps)}
            for c in CAMS:
                arr['%s/intrinsic' % c] = h['vision/%s/intrinsic_matrix' % c][()]; arr['%s/extrinsic0' % c] = h['vision/%s/extrinsic_matrix' % c][0]
            ins = h['instruction'][()]
        od.mkdir(parents=True, exist_ok=True); tmp = od / ('%07d.tmp.npz' % f['episode'])
        np.savez(tmp, steps=steps, T=T, task=f['task'], instruction=ins.decode() if isinstance(ins, bytes) else str(ins), **arr); os.replace(tmp, out)
        return 'ok'
    except Exception as e:  # noqa: BLE001  个别文件损坏或仍在下载：记下，不中断整批
        return 'err %s %s' % (f['relative_path'], str(e)[:120])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--procs', type=int, default=24); a = ap.parse_args()
    m = json.load(open(O / 'sim_train' / 'MANIFEST.json'))['files']
    todo = [f for f in m if (O / f['relative_path']).exists() and (O / f['relative_path']).stat().st_size == f['bytes']]
    print('complete files', len(todo), flush=True)
    with Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(one, todo)):
            if r.startswith('err') or i % 100 == 0: print(i, r, flush=True)


if __name__ == '__main__':
    main()
