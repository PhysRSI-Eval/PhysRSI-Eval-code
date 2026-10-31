"""ROUTER v4 deploy interface (M-R2, capability line Track D): dedicated task ID -> ROUTER-BIG (stage / progress / success / tool head)
-> tool table v1.  Deploy inputs only: 3 RGB frames per step + proprio (gripper reading / command, eef z and xy per arm), plus the
lift-time v4 abstain flag that the KPNet held-object check already emits.  No task id, task text or object poses.
  * task ID: models/taskid_text.pt (taskid/fit_final_taskid_text.py) = instruction text (given at deploy) + first-frame dino_features
    (paraphrase test: text+image 1.00 on every rewording kind, text-only breaks); without an instruction: models/taskid.pt (image only).
  * BIG: big/model-h200-v2-dense-d512L8.pt (big/train_big_v2.py on big_v2.npz; causal window of W frames; input = PCA-768 of the
    dino_features + 14 proprio); Net / Block / MoE below are copies of the trainer's modules (same parameter names).
  * tool table v1 (tooltable/TOOL-TABLE-v0.md + fitmap v1): plug_in_charger: lift-time v4 abstain -> VS (rule, outside the net);
    held (lift / transport / align) -> the BIG tool head's argmax (Astra / rg3f / VS); everything else -> Astra.
Usage:  R = RouterV4('cuda:0'); R.reset(instruction); for each step: out = R.step(images, arm_state, t, v4_abstain=False) -> dict(task, stage,
        progress, success, tool_p, method);  arm_state = dict(g=(gl, gr), z=(zl, zr), xy=((xl, yl), (xr, yr))) at step t."""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import router_rgb as RR
H = Path(__file__).resolve().parent
def load_ck(path):
    """torch.load of an H200 checkpoint: numpy-2 pickles reference numpy._core.*; the cluster env has numpy 1.x. Alias only during the
    load (a permanent alias breaks scipy / sklearn imports that probe numpy._core)."""
    import sys
    if hasattr(np, '_core'): return torch.load(path, map_location='cpu', weights_only=False)
    import numpy.core as nc; keys = ('numpy._core', 'numpy._core.multiarray'); sys.modules.update(zip(keys, (nc, nc.multiarray)))
    try: return torch.load(path, map_location='cpu', weights_only=False)
    finally:
        for k in keys: sys.modules.pop(k, None)


STAGES = ('approach', 'grasp', 'lift', 'transport', 'align', 'inserted'); TOOLS = ('Astra', 'rg3f', 'VS')


class MoE(nn.Module):
    def __init__(s, d, e):
        super().__init__(); s.g = nn.Linear(d, e); s.ex = nn.ModuleList([nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for _ in range(e)])

    def forward(s, x):
        w = F.softmax(s.g(x), -1); k = w.argmax(-1); out = torch.zeros_like(x)
        for i, ex in enumerate(s.ex):
            m = k == i
            if m.any(): out[m] = (ex(x[m]) * w[m][:, i:i + 1]).to(out.dtype)
        return out


class Block(nn.Module):
    def __init__(s, d, e):
        super().__init__(); s.n1 = nn.LayerNorm(d); s.att = nn.MultiheadAttention(d, 8, batch_first=True, dropout=0.1); s.n2 = nn.LayerNorm(d)
        s.ff = MoE(d, e) if e else nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(s, x, mask):
        h = s.n1(x); x = x + s.att(h, h, h, attn_mask=mask, need_weights=False)[0]; return x + s.ff(s.n2(x))


class Net(nn.Module):
    def __init__(s, d, L, e, W, NT=42):
        super().__init__(); s.inp = nn.Linear(768 + 14, d); s.pos = nn.Embedding(W, d); s.blocks = nn.ModuleList([Block(d, e) for _ in range(L)]); s.n = nn.LayerNorm(d)
        s.task = nn.Linear(d, NT); s.stage = nn.Linear(d, 6); s.prog = nn.Linear(d, 1); s.succ = nn.Linear(d, 1); s.take = nn.Linear(d, 3)

    def forward(s, x):
        T = x.shape[1]; mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1); h = s.inp(x) + s.pos(torch.arange(T, device=x.device))[None]
        for b in s.blocks: h = b(h, mask)
        h = s.n(h); return s.task(h), s.stage(h), s.prog(h)[..., 0], s.succ(h)[..., 0], s.take(h)


def prop14(G, Z, XY, T):
    """= big/build_big_v2.prop_from on the history so far; G, Z (L, 2), XY (L, 2, 2), T (L,) -> (L, 14)."""
    out = []
    for a in range(2):
        gi, zi, xyi = G[:, a], Z[:, a], XY[:, a]; L = len(gi); closed = gi < 0.9; since = np.full(L, -1.0); zc = np.zeros(L); cur = None
        for k in range(L):
            if closed[k] and cur is None: cur = k
            if not closed[k]: cur = None
            since[k] = (T[k] - T[cur]) / 100.0 if cur is not None else -1.0; zc[k] = (zi[k] - zi[cur]) * 10 if cur is not None else 0.0
        dz = np.r_[0, np.diff(zi)] * 100 / np.maximum(np.r_[1, np.diff(T)], 1); dxy = np.r_[0, np.linalg.norm(np.diff(xyi, axis=0), axis=1)] * 100 / np.maximum(np.r_[1, np.diff(T)], 1)
        out.append(np.c_[gi, (~closed).astype(float), since, zi, dz, dxy, zc])
    return np.concatenate(out, 1).astype(np.float32)


def tool_table(task, stage, tool_p, v4_abstain):
    if task != 'plug_in_charger': return 'Astra'
    held = STAGES[stage] in ('lift', 'transport', 'align')
    if held and v4_abstain: return 'VS'          # rule: lift-time v4 abstain -> VS (fitmap v1: VS 11/20 there, Astra / rg3f 0)
    return TOOLS[int(np.argmax(tool_p))] if held else 'Astra'


class RouterV4:
    def __init__(self, device='cuda:0', big=H / 'big/model-h200-v2-dense-d512L8.pt'):
        from transformers import AutoModel
        self.dev = torch.device(device); self.vit = AutoModel.from_pretrained(RR.DINO).to(self.dev).eval(); self.nreg = self.vit.config.num_register_tokens
        ck = torch.load(H / 'models/taskid.pt', map_location='cpu', weights_only=False); self.tasks = ck['tasks']; self.tmu, self.tsd = ck['mu'], ck['sd']
        self.tid = nn.Sequential(nn.Linear(3 * 1536, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, len(self.tasks))); self.tid.load_state_dict(ck['state_dict']); self.tid.eval()
        self.ttx = None
        if (H / 'models/taskid_text.pt').exists():
            from taskid.taskid_text_model import TextImageTaskID
            self.ttx = TextImageTaskID(torch.load(H / 'models/taskid_text.pt', map_location='cpu', weights_only=False))
        p = np.load(H / 'big/pca_v2.npz'); self.mu, self.W = p['mu'], p['W']
        ck = load_ck(big); a = ck['args']; self.win = a['W']; self.pm, self.ps = np.asarray(ck['pm'], np.float32), np.asarray(ck['ps'], np.float32)
        self.net = Net(a['d'], a['layers'], a['experts'], a['W']).to(self.dev); self.net.load_state_dict(ck['state_dict']); self.net.eval(); self.reset()

    def reset(self, instruction=None):
        self.instruction = instruction; self.task = None; self.V, self.G, self.Z, self.XY, self.T = [], [], [], [], []

    def task_id(self, f):
        if self.ttx is not None and self.instruction: return self.ttx(self.instruction, f.reshape(-1))
        with torch.no_grad(): return self.tasks[int(self.tid(torch.tensor((f.reshape(1, -1) - self.tmu) / self.tsd, dtype=torch.float32)).argmax())]

    def step_feat(self, v768, p14_hist, v4_abstain=False):
        """BIG on the PCA feature history (v768: (L, 768)) and proprio history (L, 14) -> head outputs + method (offline check entry)."""
        x = np.c_[v768[-self.win:].astype(np.float32), (p14_hist[-self.win:] - self.pm) / self.ps]
        with torch.no_grad(): _, sg, pr, sc, ta = self.net(torch.tensor(x[None], dtype=torch.float32, device=self.dev))
        st = int(sg[0, -1].argmax()); tp = torch.softmax(ta[0, -1].float(), -1).cpu().numpy()
        return dict(task=self.task, stage=STAGES[st], progress=float(torch.sigmoid(pr[0, -1])), success=float(torch.sigmoid(sc[0, -1])), tool_p=tp.round(3).tolist(),
                    method=tool_table(self.task, st, tp, v4_abstain))

    def step(self, images, arm_state, t, v4_abstain=False):
        f = RR.dino_features(self.vit, self.nreg, images, self.dev)
        if self.task is None: self.task = self.task_id(f)
        self.V.append(((f.reshape(-1) - self.mu) @ self.W).astype(np.float32)); self.G.append(arm_state['g']); self.Z.append(arm_state['z']); self.XY.append(arm_state['xy']); self.T.append(t)
        P = prop14(np.asarray(self.G, float), np.asarray(self.Z, float), np.asarray(self.XY, float), np.asarray(self.T, float))
        return self.step_feat(np.asarray(self.V), P, v4_abstain)
