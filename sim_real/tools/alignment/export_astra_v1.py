"""Astra episodes -> mva-20261029/astra_v1 (format agreed with the post-training line 09:2xZ; rsiws-dc 2026-10-29; zero quota, replay only).
相对第一版（export_astra_mva.py）的改动：
  - 键名与 data/arrays 完全一致：关节 (T,6)、夹爪 (T,1)、末端位姿 (T,7)；action/* 填执行动作（关节与夹爪为执行指令，末端位姿为执行后实测）。
  - 相机：头部用 rpc('camera') 的固定参数；腕部按约定用关节 FK（reach.Reach）乘 cplabels.MOUNT 复算（OpenGL 相机到世界），不用回放的 camera RPC。
  - 重放偏差容差 0.03 rad（第一版 1e-3 拒掉了 80 个回合，偏差中位 0.0018、最大 0.0235；wastex 已采用回合 p90 为 0.0128），实际值写进清单。
  - 每 2 步存物体真值 truth/<task>/<ep>.json（K1 投影 CAD 点、K2 关键点对齐的目标）；接触真值服务端无此 RPC，暂缺，清单注明。
帧文件名用 cam_head（与 data/frames、prep_mva tokens 一致）。scratch 每回合结束删除。
  python export_astra_v1.py --episodes FILE --gpu G --shard i/n"""
import argparse, hashlib, json, shutil, sys, time, traceback
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/lizekai/RSIws/experiments/checkhold-v2-20261028/src'); sys.path.insert(0, '/lizekai/RSIws/experiments/ee-coinpose-20261028/src')
import export_ch2 as E  # noqa: E402  (共享 Kit 缓存硬链接、replay server 路径设置都在其中)
import cplabels as CL  # noqa: E402
RA, RM = E.RA, E.RM
OUT = Path('/lizekai/RSIws/experiments/mva-20261029/astra_v1'); NATIVE = Path('/lizekai/RSIws/artifacts/native')
CAMS = {'cam_high': 'cam_head', 'cam_left_wrist': 'cam_left_wrist', 'cam_right_wrist': 'cam_right_wrist'}; ARMS = ('left', 'right')
TOL = 0.03; HOLD_T = {'insert_tubes', 'play_Xylophone'}; HOLD_L = {3, 8, 13}; INSTR = {}


def split_of(task, ep, lay):
    if task in HOLD_T: return 'HOLD-T'
    h = int(hashlib.sha1(ep.encode()).hexdigest(), 16) % 5
    if h == 4: return 'TEST'
    if h == 3: return 'DEV'
    return 'HOLD-L' if lay in HOLD_L else 'TRAIN'


def pose7(M):
    q = R.from_matrix(M[:3, :3]).as_quat(); return np.r_[M[:3, 3], q[3], q[:3]]     # xyz + wxyz


def export(ep, gpu):
    from PIL import Image
    row = dict(episode=ep, status='failed'); t0 = time.time(); sim = None; SCR = Path('/root/ip3-1027/sim') / ('av1-' + ep[-8:])
    try:
        task, lay, actions = RA.recorded(ep); row.update(task=task, layout=lay, split=split_of(task, ep, lay))
        # 部分旧回合没有 evaluation_outcome.json，回退到 summary.json 的 success；都没有则记 None
        oc, sm = NATIVE / ep / 'evaluation_outcome.json', NATIVE / ep / 'summary.json'
        row['native_success'] = (bool(json.loads(oc.read_text()).get('native_success')) if oc.exists()
                                 else bool(json.loads(sm.read_text()).get('success')) if sm.exists() else None)
        sim = RA.Sim(task, SCR, gpu); rpc = sim.rpc.request; rpc('reset', layout_id=lay); rm = RM.Reach(RA.plain(rpc('kin')))
        head = RA.plain(rpc('camera'))['cam_head']
        fd = OUT / 'frames' / task / ep; fd.mkdir(parents=True, exist_ok=True)
        S = {k: [] for k in ('q', 'eef', 'grip')}; A = {k: [] for k in ('q', 'grip')}; objs = {}; worst = 0.0

        def snap(t):
            ob = rpc('observe'); q = np.asarray(ob['states'], float); S['q'].append(q)
            S['eef'].append([np.r_[ob['eef_positions'][i], ob['eef_quaternions_wxyz'][i]] for i in range(2)])
            if t % 2 == 0:
                for c, n in CAMS.items():
                    Image.fromarray(np.asarray(ob['images'][c], np.uint8)).resize((320, 240), Image.BILINEAR).save(fd / ('%04d-%s.jpg' % (t, n)), quality=90)
                objs[t] = RA.plain(rpc('objects'))
        snap(0)
        for a in actions:
            g = rpc('replay', actions=[a['action']])[0]
            if a['states']: worst = max(worst, max(abs(x - y) for x, y in zip(g['states'], a['states'])))
            if worst > TOL: raise RuntimeError('replay diverged at step %d (err %.4f)' % (a['step'], worst))
            A['q'].append(np.asarray(a['action'], float)); snap(a['step'])
            if g.get('ended'): break
        T = len(A['q']); q = np.asarray(S['q']); eef = np.asarray(S['eef']); aq = np.asarray(A['q'])
        arr = dict(T=T, freq=25, instruction=INSTR.get(task, ''))
        for i, arm in enumerate(ARMS):
            arr['state/%s_arm_joint_states' % arm] = q[:T, 7 * i:7 * i + 6]; arr['state/%s_ee_joint_states' % arm] = q[:T, 7 * i + 6:7 * i + 7]
            arr['state/%s_ee_poses' % arm] = eef[:T, i]
            arr['action/%s_arm_joint_states' % arm] = aq[:, 7 * i:7 * i + 6]; arr['action/%s_ee_joint_states' % arm] = aq[:, 7 * i + 6:7 * i + 7]
            arr['action/%s_ee_poses' % arm] = eef[1:T + 1, i]
            arr['cam_%s_wrist/extrinsic' % arm] = np.stack([rm.pose(arm, q[t, 7 * i:7 * i + 6]) @ CL.MOUNT for t in range(T)])
            arr['cam_%s_wrist/intrinsic' % arm] = np.asarray(CL.K_WRIST, float)
        arr['cam_head/extrinsic'] = np.repeat(np.asarray(head['extrinsics_cam_to_world'], float)[None], T, 0)
        arr['cam_head/intrinsic'] = np.asarray(head['intrinsics'], float)
        ad = OUT / 'arrays' / task; ad.mkdir(parents=True, exist_ok=True); np.savez_compressed(ad / (ep + '.npz'), **arr)
        (OUT / 'truth' / task).mkdir(parents=True, exist_ok=True); (OUT / 'truth' / task / (ep + '.json')).write_text(json.dumps(objs))
        # 自检：FK*MOUNT 与 eef*MOUNT 的腕部相机位置差（eef 是否即 link6），写进清单供后训练线核对
        d = [np.linalg.norm(arr['cam_%s_wrist/extrinsic' % arm][0][:3, 3] - (E.T_of(eef[0, i][:3], eef[0, i][3:]) @ CL.MOUNT)[:3, 3]) for i, arm in enumerate(ARMS)]
        row.update(status='used', T=T, valid=True, replay_max_state_err=worst, wrist_fk_vs_eef_m=max(d), contacts='not available (no server RPC)')
    except Exception as e:  # noqa: BLE001  逐回合记录错误，不让一个坏回合中断整片
        row.update(error='%s: %s' % (type(e).__name__, str(e)[:300]), valid=False)
    finally:
        if sim is not None:
            try: sim.close()
            except Exception: pass  # noqa: BLE001
        shutil.rmtree(str(SCR), ignore_errors=True); shutil.rmtree('/root/rgbhint-sim-1023/' + SCR.name, ignore_errors=True)
        row['seconds'] = round(time.time() - t0)
        (OUT / 'episodes').mkdir(parents=True, exist_ok=True); (OUT / 'episodes' / (ep + '.json')).write_text(json.dumps(RA.plain(row)))
        with open(OUT / 'MANIFEST.jsonl', 'a') as f: f.write(json.dumps(RA.plain({k: v for k, v in row.items() if k != 'tb'})) + '\n')
        print(json.dumps({k: row.get(k) for k in ('episode', 'task', 'T', 'status', 'error', 'seconds')}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--episodes', required=True); ap.add_argument('--gpu', type=int, required=True); ap.add_argument('--shard', default='0/1')
    a = ap.parse_args(); i, n = map(int, a.shard.split('/'))
    for f in Path('/lizekai/RSIws/experiments/mva-20261029/data/arrays').glob('*/*.npz'):
        INSTR.setdefault(f.parent.name, str(np.load(f)['instruction']))
    for k, ep in enumerate(json.load(open(a.episodes))):
        if k % n == i and not (OUT / 'episodes' / (ep + '.json')).exists(): export(ep, a.gpu)
