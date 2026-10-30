"""Router 第 0 层数据：从 artifacts/native 每个 Astra 回合的决策点画面（host-client/observations/NNN/{cam}.png，
即 Astra 每轮决策时看到的三路图像，也正是 router 部署时的输入时机）均匀抽 K 轮，生成 DINOv3-B/16 池化 token
(K,3,25,768) fp16 -> data-native/pooled/<task>/<episode>.npy，并写 .json（轮次编号、布局）。不需要仿真。
模型、分辨率与池化方式同 prep_mva / train_pw（320x240 -> 20x15 patch -> 5x5 平均池化）。
  python -I extract_native_turns.py --gpu G --shard i/n [--k 16]"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

NAT = Path('/lizekai/RSIws/artifacts/native'); OUT = Path('/lizekai/RSIws/experiments/ee-physword-20261029/data-native')
CAMS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


def load_turns(ep, k):
    from PIL import Image
    d = NAT / ep / 'host-client' / 'observations'
    if not d.is_dir(): return None, None
    turns = sorted(t for t in os.listdir(d) if (d / t / 'cam_high.png').exists())
    if not turns: return None, None
    pick = [turns[i] for i in np.unique(np.linspace(0, len(turns) - 1, min(k, len(turns))).round().astype(int))]
    keep, ims = [], []
    for t in pick:   # 少数归档 PNG 损坏（10-30 08:41Z 三个分片因此崩溃）：跳过该决策点并记录，不丢整个回合
        try: ims.append([np.asarray(Image.open(d / t / ('%s.png' % c)).convert('RGB').resize((320, 240), Image.BILINEAR), np.float32) / 255.0 for c in CAMS]); keep.append(t)
        except Exception as e:  # noqa: BLE001
            print('bad image', ep, t, type(e).__name__, flush=True)
    return (keep, np.asarray(ims)) if keep else (None, None)       # (K,3,240,320,3)


def main():
    import torch, torch.nn.functional as F
    from transformers import AutoModel
    ap = argparse.ArgumentParser(); ap.add_argument('--gpu', type=int, required=True); ap.add_argument('--shard', default='0/1'); ap.add_argument('--k', type=int, default=16)
    a = ap.parse_args(); i, n = map(int, a.shard.split('/'))
    eps = json.load(open(Path(__file__).parent / 'native_png_episodes.json'))
    todo = [(t, e, lay) for t, v in sorted(eps.items()) for e, _, lay in v]
    todo = [x for j, x in enumerate(todo) if j % n == i and not (OUT / 'pooled' / x[0] / ('%s.npy' % x[1])).exists()]
    dev = torch.device('cuda:%d' % a.gpu); vit = AutoModel.from_pretrained('/lizekai/RSIws/svc/dinov3-weights/dinov3-vitb16').to(dev).eval(); nreg = vit.config.num_register_tokens
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    pool = ThreadPoolExecutor(8); futs = {}
    for j in range(min(4, len(todo))): futs[j] = pool.submit(load_turns, todo[j][1], a.k)   # 预取，读 NAS 与 GPU 计算重叠
    for j, (task, ep, lay) in enumerate(todo):
        if j + 4 < len(todo): futs[j + 4] = pool.submit(load_turns, todo[j + 4][1], a.k)
        pick, ims = futs.pop(j).result()
        if pick is None: continue
        x = torch.from_numpy(ims.reshape(-1, 240, 320, 3)).permute(0, 3, 1, 2).to(dev)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            tk = vit(pixel_values=(x - mean) / std).last_hidden_state[:, 1 + nreg:].float()      # (K*3,300,768)
        g = tk.view(-1, 15, 20, 768).permute(0, 3, 1, 2); pooled = F.adaptive_avg_pool2d(g, (5, 5)).flatten(2).transpose(1, 2)
        od = OUT / 'pooled' / task; od.mkdir(parents=True, exist_ok=True)
        np.save(od / ('%s.tmp.npy' % ep), pooled.view(len(pick), 3, 25, 768).cpu().numpy().astype(np.float16)); os.replace(od / ('%s.tmp.npy' % ep), od / ('%s.npy' % ep))
        (od / ('%s.json' % ep)).write_text(json.dumps(dict(task=task, episode=ep, layout=lay, turns=pick)))
        if j % 50 == 0: print(j, len(todo), task, ep, flush=True)


if __name__ == '__main__':
    main()
