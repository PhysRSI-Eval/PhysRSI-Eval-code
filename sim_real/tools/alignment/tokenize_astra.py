"""DINOv3-B/16 tokens for the Astra-replay MVA export (same model, resolution, stride and layout as prep_mva.py tokens; rsiws-dc 2026-10-29).
轮询 data-astra/episodes 中已成功导出的回合，每 4 步取三路相机帧 -> tokens/<task>/<ep>.npy (N,3,300,768) fp16 + .steps.json。
  python tokenize_astra.py --gpu G [--loop-min 120]"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
V = Path('/lizekai/RSIws/experiments/ee-physword-20261029/data-astra'); CAMS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist'); TSTRIDE = 4


def main():
    import torch
    from PIL import Image
    from transformers import AutoModel
    ap = argparse.ArgumentParser(); ap.add_argument('--gpu', type=int, default=0); ap.add_argument('--loop-min', type=int, default=120)
    ap.add_argument('--root', default=str(V)); ap.add_argument('--marker', default='PW EXPORT DONE'); a = ap.parse_args(); R = Path(a.root)
    dev = torch.device('cuda:%d' % a.gpu); vit = AutoModel.from_pretrained('/lizekai/RSIws/svc/dinov3-weights/dinov3-vitb16').to(dev).eval(); nreg = vit.config.num_register_tokens
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    end = time.time() + 60 * a.loop_min
    while True:
        for f in sorted((R / 'episodes').glob('*.json')):
            r = json.load(open(f))
            if r.get('error') or not r.get('T'): continue
            td = R / 'tokens' / r['task']
            if (td / (r['episode'] + '.npy')).exists(): continue
            fd = R / 'frames' / r['task'] / r['episode']; steps = list(range(0, r['T'], TSTRIDE)); out = np.zeros((len(steps), 3, 300, 768), np.float16)
            for i0 in range(0, len(steps), 32):
                ss = steps[i0:i0 + 32]
                x = torch.from_numpy(np.stack([np.asarray(Image.open(fd / ('%04d-%s.jpg' % (t, c))), np.float32) / 255.0 for t in ss for c in CAMS])).permute(0, 3, 1, 2).to(dev)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    tk = vit(pixel_values=(x - mean) / std).last_hidden_state[:, 1 + nreg:]
                out[i0:i0 + len(ss)] = tk.float().cpu().numpy().reshape(len(ss), 3, 300, 768).astype(np.float16)
            td.mkdir(parents=True, exist_ok=True); np.save(td / (r['episode'] + '.tmp.npy'), out)
            os.replace(td / (r['episode'] + '.tmp.npy'), td / (r['episode'] + '.npy')); (td / (r['episode'] + '.steps.json')).write_text(json.dumps(steps))
            print('tokens', r['task'], r['episode'], out.shape, flush=True)
        if time.time() > end or (V.parent / 'logs' / 'queue.log').exists() and a.marker in (V.parent / 'logs' / 'queue.log').read_text(): break
        time.sleep(60)


if __name__ == '__main__':
    main()
