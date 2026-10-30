"""keypoint factory labels, plug_in_charger (spec: mailbox/to-ip4.md 12:0xZ). For every captured view of data-charger-hover/*:
14 keypoints = charger_0_1 (5: prong tip L/R, prong root L/R, housing front-face centre; charger CAD frame = sim object frame) +
socket_0_2 (3 outlets x [hole entry a, hole entry b, outlet centre]); hole entry := charger prong tip mapped by the seated pose
S_k = [SEAT_R, (OFFS[k], OUT_Y, SEAT_P)] in the strip frame (sim socket pose), put on the strip top (z = origin + 0.02157); the pair is
ordered by strip-x. Projection with the FK cameras (link6 x mount, OpenGL); head camera = static calibration.
Visibility: inside the image AND the sim depth at the pixel >= point depth - 5 mm (fingers, the charger and the strip occlude for real).
Writes kpnet3 records per camera group to <out>/<group>/{trdev,test}/L<layout>-s0.jsonl (+ images symlink): groups free (idle-arm
A/B parks), hold (holding wrist), head. Split by episode: TEST layouts 3/8/13; else sha1(episode)%5==3 -> DEV (split 1), else TRAIN (0).
  python make_labels_kp.py --data data-charger-hover --out kp-charger"""
import argparse, glob, hashlib, json, os
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

K = Path('/lizekai/RSIws/experiments/kpfactory-20261029'); N = Path('/lizekai/RSIws/artifacts/native')
OFFS = (-0.0295, 0.0070, 0.0435); OUT_Y = 0.0064; SEAT_P = 0.0202; SEAT_R = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]); TOP = 0.02157
TIPS = np.array([[-0.0055, -0.0152, 0.0], [0.0055, -0.0152, 0.0]]); CH_KP = np.vstack([TIPS, [[-0.0055, 0, 0], [0.0055, 0, 0], [0, 0, 0]]])
CORNERS = np.array([[x, y, z] for y in (0.0, 0.0479) for x in (-0.021, 0.021) for z in (-0.0135, 0.0135)])  # v2: housing bbox corners (front y=0, back y=0.0479)
NAMES = ['tip_l', 'tip_r', 'root_l', 'root_r', 'face_c'] + ['o%d_%s' % (k, n) for k in range(3) for n in ('hole_a', 'hole_b', 'centre')]
HEAD = json.load(open('/lizekai/RSIws/experiments/multiasset-hold-20261028/pose/cad/head_cam.json')); KW = [[397.04127893104277, 0, 320.0], [0, 397.04127893104277, 240.0], [0, 0, 1.0]]


def T_of(p, q): M = np.eye(4); M[:3, :3] = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); M[:3, 3] = p; return M


def world_kp(charger, socket):
    Tc = T_of(charger['position'], charger['quaternion_wxyz']); Ts = T_of(socket['position'], socket['quaternion_wxyz'])
    P = [Tc[:3, :3] @ p + Tc[:3, 3] for p in CH_KP]; ztop = Ts[2, 3] + TOP
    for ox in OFFS:
        S = np.eye(4); S[:3, :3] = SEAT_R; S[:3, 3] = [ox, OUT_Y, SEAT_P]; W = Ts @ S; holes = [W[:3, :3] @ t + W[:3, 3] for t in TIPS]
        holes = sorted(holes, key=lambda h: (Ts[:3, :3].T @ (h - Ts[:3, 3]))[0])
        for h in holes: h = h.copy(); h[2] = ztop; P.append(h)
        c = Ts[:3, :3] @ np.array([ox, OUT_Y, 0.0]) + Ts[:3, 3]; c[2] = ztop; P.append(c)
    return np.array(P)


def project(Kin, Tc2w, P):
    Tw2c = np.linalg.inv(np.asarray(Tc2w)); q = P @ Tw2c[:3, :3].T + Tw2c[:3, 3]; z = -q[:, 2]; Kin = np.asarray(Kin)
    return np.c_[Kin[0, 0] * q[:, 0] / np.maximum(z, 1e-4) + Kin[0, 2], -Kin[1, 1] * q[:, 1] / np.maximum(z, 1e-4) + Kin[1, 2]], z


def main(a):
    jobs = {}  # 13:5xZ erratum fix: the episode is the arg after --episode (args[2] was wrong for some rows); + held-out test jobs
    for jf in ('hover_jobs.json', 'hover_jobs_test.json', 'hover_jobs_all.json'):
        if (K / 'src' / jf).exists():
            for j in json.load(open(K / 'src' / jf)): ep = j['args'][j['args'].index('--episode') + 1]; jobs[ep[-8:]] = ep
    out = Path(a.out); stats = {}
    for g in ('free', 'hold', 'head'):
        for sub in ('trdev', 'test'):
            d = out / g / sub; d.mkdir(parents=True, exist_ok=True)
            if not (d / 'images').exists(): os.symlink(str(Path(a.data).resolve()), d / 'images')
    files = {}
    V2 = a.version == 2
    if V2: NAMES.extend(['corner_%d' % i for i in range(8)])
    for f in sorted(glob.glob(str(Path(a.data) / '*' / 'record.json'))):
        nm = Path(f).parent.name; ep = jobs.get(nm)
        if ep is None: continue
        lay = int((json.load(open(N / ep / 'reset.json')).get('metadata') or {}).get('layout_id') or 0)
        if V2:  # v2 split (PREREG-KPF-plug-v2): TEST = layouts 5/11/16; DEV = layouts 3/8/13 + sha1 % 5 == 3; else TRAIN
            sp = 'test' if lay in (5, 11, 16) else 'trdev'; spi = 1 if (lay in (3, 8, 13) or int(hashlib.sha1(ep.encode()).hexdigest(), 16) % 5 == 3) else 0
        else:
            sp = 'test' if lay in (3, 8, 13) else 'trdev'; spi = 1 if int(hashlib.sha1(ep.encode()).hexdigest(), 16) % 5 == 3 else 0
        rec = json.load(open(f))
        for st in rec['states']:
            for v in st['views']:
                P = world_kp(v['charger'], v['socket'])
                if V2: Tc_ = T_of(v['charger']['position'], v['charger']['quaternion_wxyz']); P = np.vstack([P, CORNERS @ Tc_[:3, :3].T + Tc_[:3, 3]])
                for cam, img in v['images'].items():
                    g = 'head' if cam == 'cam_high' else ('hold' if v['tag'].endswith('hold') else 'free')
                    Kin, Tc2w = (HEAD['intrinsics'], HEAD['extrinsics_cam_to_world']) if cam == 'cam_high' else (KW, v['cam_to_world'][cam])
                    uv, z = project(Kin, Tc2w, P); dep = np.asarray(Image.open(Path(f).parent / v['depth'][cam]), float) / 1000.0; vis = []
                    for (u, w_), zz in zip(uv, z):
                        ok = zz > 0.02 and 0 <= u < 640 and 0 <= w_ < 480
                        if ok: dd = dep[int(w_), int(u)]; ok = bool(dd <= 0 or dd >= zz - 0.005)
                        vis.append(bool(ok))
                    r = dict(sample_id='%s-%s-%s' % (nm, v['tag'], cam), image='%s/%s' % (nm, img), layout=lay, seed=spi, split=spi, camera=cam, episode=ep, intrinsics=Kin,
                             extrinsics_cam_to_world=Tc2w, outlets=P.tolist(), outlets_uv=uv.tolist(), in_image=vis, keypoint_names=NAMES, kind=st['kind'], state=st['k'])
                    files.setdefault((g, sp, lay), []).append(r); stats[(g, sp)] = stats.get((g, sp), 0) + 1
    for (g, sp, lay), rows in files.items():
        with open(out / g / sp / ('L%03d-s0.jsonl' % lay), 'w') as h:
            for r in rows: h.write(json.dumps(r) + '\n')
    print(json.dumps({'%s/%s' % k: v for k, v in sorted(stats.items())}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--version', type=int, default=1); main(ap.parse_args())
