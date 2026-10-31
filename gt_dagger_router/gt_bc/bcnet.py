"""K1 insert-skill BC policy network + data (GT-DAgger P1, tool line ip4). Inputs at step t (t even, where the router's DINOv3 cache has
features): 3 view tokens ([CLS, mean patch] DINOv3 ViT-B/16, head / left wrist / right wrist), the current 14-d state, the GT history of the
last HIST steps (state, action) = teacher forcing, and the task token. Output: action chunk a[t .. t+H-1] (14-d each), masked past the
episode end. Target parametrisation (--target): 'res' = arm joints as a[t+i] - s[t] (needs ABS joint targets, P0), grippers absolute;
'abs' = all absolute. Everything normalised with train-only statistics (norm.json, 'K1')."""
import json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
G = Path('/lizekai/RSIws/experiments/gt-bc-20261031'); FEAT = Path('/lizekai/RSIws/experiments/macro-tools-20261027/router-rgb/official/feat')
K1 = ['plug_in_charger', 'insert_key', 'insert_tubes', 'deposit_coin']; TASKS = K1 + ['make_kong', 'organize_table', 'play_Xylophone', 'pour_liquid_into_cup', 'push_T']; GRIP = [6, 13]; ARMM = np.ones(14, np.float32); ARMM[GRIP] = 0


class Norm:
    def __init__(self, key='K1'):
        n = json.load(open(G / 'norm.json'))[key]; self.sm, self.ss = np.array(n['state_mean'], np.float32), np.array(n['state_std'], np.float32)
        self.am, self.as_ = np.array(n['action_mean'], np.float32), np.array(n['action_std'], np.float32)

    def enc_y(self, a, s, target):  # a (H,14), s (14,)
        return ((a - s * ARMM) / self.as_) if target == 'res' else ((a - self.am) / self.as_)

    def dec_y(self, y, s, target):
        return (y * self.as_ + s * ARMM) if target == 'res' else (y * self.as_ + self.am)


def load_episode(task, ep):
    z = np.load(FEAT / task / (ep + '.npz')); a = np.load(G / 'act' / task / (ep + '.npz'))
    return dict(task=task, ep=ep, X=z['X'].astype(np.float32), tX=z['t'], state=a['state'], action=a['action'], T=int(a['T']))


def sample(E, i, nm, hist, H, target):
    """i = index into E['tX'] (t = tX[i]) -> tensors for one sample (teacher-forced GT history)."""
    t = int(E['tX'][i]); T = E['T']; s = E['state'][t]
    hs = np.stack([E['state'][max(0, t - j)] for j in range(hist, 0, -1)] or [np.zeros(14, np.float32)])[:hist]
    ha = np.stack([E['action'][max(0, t - j)] if t - j >= 0 else E['state'][0] for j in range(hist, 0, -1)] or [np.zeros(14, np.float32)])[:hist]
    hm = np.array([1.0 if t - j >= 0 else 0.0 for j in range(hist, 0, -1)], np.float32)
    a = np.stack([E['action'][min(T - 1, t + j)] for j in range(H)]); m = np.array([1.0 if t + j < T else 0.0 for j in range(H)], np.float32)
    h = np.concatenate([(hs - nm.sm) / nm.ss, (ha - nm.am) / nm.as_], 1)
    return E['X'][i], (s - nm.sm) / nm.ss, h, hm, nm.enc_y(a, s, target), m


class BC(nn.Module):
    def __init__(self, H, hist, ntask=4, d=256, L=4):
        super().__init__(); self.H = H
        self.img = nn.Linear(1536, d); self.view = nn.Parameter(torch.zeros(3, d)); self.st = nn.Linear(14, d); self.hi = nn.Linear(28, d); self.hpos = nn.Parameter(torch.zeros(max(1, hist), d))
        self.task = nn.Embedding(ntask, d); self.cls = nn.Parameter(torch.zeros(1, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True), L); self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, H * 14))

    def forward(self, X, s, h, hm, task):
        B = X.shape[0]; tok = [self.cls.expand(B, 1, -1), self.img(X) + self.view, self.st(s)[:, None], self.task(task)[:, None]]; pad = [torch.zeros(B, 6, dtype=torch.bool, device=X.device)]
        if h.shape[1]: tok.append(self.hi(h) + self.hpos); pad.append(hm < 0.5)
        z = self.enc(torch.cat(tok, 1), src_key_padding_mask=torch.cat(pad, 1)); return self.out(z[:, 0]).view(B, self.H, 14)
