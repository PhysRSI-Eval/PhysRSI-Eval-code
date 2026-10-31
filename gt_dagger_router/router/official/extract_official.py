"""官方 RoboDojo sim_train 示教（9 个任务 × 100 条，均为成功示教）→ 大路由器的训练特征。
每 2 步取一帧：head / 左腕 / 右腕 JPEG 解码 → router_rgb.dino_features（与路由器部署时同一特征函数，保证训练/部署一致）。
同时保存部署可用的本体（关节、夹爪读数、末端位姿）和时间进度 t/T；阶段标签由训练脚本从本体推导（夹爪闭合 + 末端 z 变化），
官方数据里没有物体位姿，所以这里不做物体相关标签。
  python -I extract_official.py SHARD NSHARD GPU  -> feat/<task>/<episode>.npz（已存在则跳过，可断点续跑）"""
import io, json, sys
from pathlib import Path
import h5py, numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import router_rgb as RR   # noqa: E402
from transformers import AutoModel
SRC = Path('/lizekai/RSIws/data/official/sim_train'); OUT = Path(__file__).resolve().parent / 'feat'
EPS = sorted(p for p in SRC.glob('*/episode_*.hdf5'))
k, n, g = map(int, sys.argv[1:4]); dev = torch.device('cuda:%d' % g)
vit = AutoModel.from_pretrained(RR.DINO).to(dev).eval(); nreg = vit.config.num_register_tokens
CAM = {'cam_high': 'cam_head', 'cam_left_wrist': 'cam_left_wrist', 'cam_right_wrist': 'cam_right_wrist'}
import os
MINE = EPS[k::n][::-1] if os.environ.get('REV') else EPS[k::n]   # REV=1：倒序跑同一分片，帮慢分片收尾（输出已存在即跳过）
for p in MINE:
    o = OUT / p.parent.name / (p.stem + '.npz')
    if o.exists(): continue
    o.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(p, 'r') as f:
        T = f['state/left_arm_joint_states'].shape[0]; ts = list(range(0, T, 2)); X = []
        for t in ts:
            ims = {a: np.asarray(Image.open(io.BytesIO(f['vision/%s/colors' % b][t])).convert('RGB')) for a, b in CAM.items()}
            X.append(RR.dino_features(vit, nreg, ims, dev).astype(np.float16))
        st = {kk: f['state/' + kk][ts] for kk in ('left_arm_joint_states', 'right_arm_joint_states', 'left_ee_joint_states', 'right_ee_joint_states', 'left_ee_poses', 'right_ee_poses')}
        ins = f['instruction'][()]; ins = ins.decode() if isinstance(ins, bytes) else str(ins)
    np.savez(o.with_suffix('.tmp%d.npz' % os.getpid()), X=np.stack(X), t=np.asarray(ts), T=T, task=p.parent.name, instruction=ins, **st)
    o.with_suffix('.tmp%d.npz' % os.getpid()).rename(o); print(p.parent.name, p.stem, T, flush=True)
