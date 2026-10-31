"""ROUTER-RGB-v1 deploy interface for the harness L3 router slot (hv2g; capability line, Track D). Inputs: deploy packet only (3 RGB
frames + proprio/progress); no task id, no task text, no tool facts.
Decision = route to the C1 executor iff  E1.route_c1(packet) > tau_route  AND  takeover(packet) > 0.65.
  * E1 (rsiws-dc, ee-physword train_router.py, checkpoints out/router-dev-rgb-f*.pt): C1-scene + holding recognition from pooled DINOv3
    tokens; its feature function is rsiws-dc's (train_pw.pool_cam over DINOv3 tokens) and is called through e1_score() when available.
  * takeover head (this file): ROUTER-v0 transformer without the facts token, trained on the 67 deduplicated plug rows (rg3f executor)
    on frozen DINOv3-B/16 [CLS, mean patch] of the 640x480 head / left / right frames + gate-v1 proprio features.
Usage:  R = RouterRGB('cuda:0');  R.takeover(images, feats) -> p;  images = {'cam_high': HxWx3, 'cam_left_wrist': .., 'cam_right_wrist': ..}
        feats = gate-v1 proprio/progress dict (build_dataset_v2.features keys) computed from the action history up to now."""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
H = Path(__file__).resolve().parent; DINO = '/lizekai/RSIws/svc/dinov3-weights/dinov3-vitb16'


def dino_features(vit, nreg, images, dev):
    """[CLS, mean patch] per view, views in the order head, left wrist, right wrist -> (3, 1536); identical to gate/build_router_v0.py."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    x = torch.stack([torch.from_numpy(np.asarray(images[c], np.float32)[..., :3] / 255.0).permute(2, 0, 1) for c in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')]).to(dev)
    with torch.no_grad(): h = vit(pixel_values=(x - mean) / std).last_hidden_state
    return torch.cat([h[:, 0], h[:, 1 + nreg:].mean(1)], 1).float().cpu().numpy()


class Router(nn.Module):
    """= gate/train_router_v0.Router (2-layer transformer, d 128, 4 heads) used with a constant 1-d facts token (no facts)."""
    def __init__(self, d_tab, d_fact, d_pw=0, d=128):
        super().__init__()
        self.img = nn.Linear(1536, d); self.view = nn.Parameter(torch.zeros(3, d)); self.tab = nn.Linear(d_tab, d); self.fact = nn.Linear(d_fact, d)
        self.pw = nn.Linear(d_pw, d) if d_pw else None; self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 256, dropout=0.1, batch_first=True), 2); self.out = nn.Linear(d, 1)

    def forward(self, xi, xt, xf, xp=None):
        toks = [self.img(xi) + self.view, self.tab(xt)[:, None], self.fact(xf)[:, None]] + ([self.pw(xp)[:, None]] if self.pw is not None else [])
        h = torch.cat([self.cls.expand(xi.shape[0], -1, -1)] + toks, 1); return self.out(self.enc(h)[:, 0, :])[:, 0]


def tab(f):
    return [f['budget_used'], f['grip_l'], f['grip_r'], f['cmd_l'], f['cmd_r'], f['holding_l'], f['holding_r'], f['closes_l'], f['closes_r'], f['travel_rad'] / 10.0] + [v / 3.0 for v in f['joints']]


class RouterRGB:
    def __init__(self, device='cuda:0'):
        from transformers import AutoModel
        self.dev = torch.device(device); self.vit = AutoModel.from_pretrained(DINO).to(self.dev).eval(); self.nreg = self.vit.config.num_register_tokens
        ck = torch.load(H / 'models/takeover.pt', map_location='cpu'); self.heads = []
        for sd in ck['state_dicts']:
            m = Router(ck['d_tab'], 1).to(self.dev); m.load_state_dict(sd); m.eval(); self.heads.append(m)

    def takeover(self, images, feats):
        xi = torch.tensor(dino_features(self.vit, self.nreg, images, self.dev)[None], dtype=torch.float32, device=self.dev)
        xt = torch.tensor([tab(feats)], dtype=torch.float32, device=self.dev); xf = torch.zeros(1, 1, device=self.dev)
        with torch.no_grad(): return float(np.mean([torch.sigmoid(m(xi, xt, xf)).item() for m in self.heads]))
