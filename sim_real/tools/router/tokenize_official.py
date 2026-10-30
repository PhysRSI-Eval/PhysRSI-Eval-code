"""第 0 层换场景检验数据（第 2 步，preview 环境）：data-official/frames/*.npz（每条 16 个时刻 x 三路 JPEG）-> DINOv3-B/16
池化 token (K,3,25,768) fp16 -> data-official/pooled/<task>/<episode>.npy + .json。官方 cam_head 与 Astra cam_high 内外参一致，
处理方式与 extract_native_turns.py 相同（320x240 -> 20x15 patch -> 5x5 平均池化）。  python -I tokenize_official.py --gpu G"""
import argparse, io, json, os
from pathlib import Path
import numpy as np

D = Path('/lizekai/RSIws/experiments/ee-physword-20261029/data-official'); CAMS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')


def main():
    import torch, torch.nn.functional as F
    from PIL import Image
    from transformers import AutoModel
    ap = argparse.ArgumentParser(); ap.add_argument('--gpu', type=int, required=True); a = ap.parse_args()
    dev = torch.device('cuda:%d' % a.gpu); vit = AutoModel.from_pretrained('/lizekai/RSIws/svc/dinov3-weights/dinov3-vitb16').to(dev).eval(); nreg = vit.config.num_register_tokens
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    for f in sorted(D.glob('frames/*/*.npz')):
        if '.tmp' in f.name: continue
        task = f.parent.name; od = D / 'pooled' / task; out = od / (f.stem + '.npy')
        if out.exists(): continue
        z = np.load(f); K = len(z['steps'])
        ims = [np.asarray(Image.open(io.BytesIO(z['%s/jpg%02d' % (c, i)].tobytes())).convert('RGB').resize((320, 240), Image.BILINEAR), np.float32) / 255.0 for i in range(K) for c in CAMS]
        x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).to(dev)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            tk = vit(pixel_values=(x - mean) / std).last_hidden_state[:, 1 + nreg:].float()
        pooled = F.adaptive_avg_pool2d(tk.view(-1, 15, 20, 768).permute(0, 3, 1, 2), (5, 5)).flatten(2).transpose(1, 2)
        od.mkdir(parents=True, exist_ok=True); np.save(od / (f.stem + '.tmp.npy'), pooled.view(K, 3, 25, 768).cpu().numpy().astype(np.float16)); os.replace(od / (f.stem + '.tmp.npy'), out)
        (od / (f.stem + '.json')).write_text(json.dumps(dict(task=task, episode='official/%s/%s' % (task, f.stem), layout=None, steps=z['steps'].tolist())))


if __name__ == '__main__':
    main()
