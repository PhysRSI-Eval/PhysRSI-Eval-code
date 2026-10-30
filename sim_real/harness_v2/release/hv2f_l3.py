"""hv2f additions to the L3 glue (bottleneck line, 10-29 14:xxZ): ROUTER-v0 as the successor of gate v2 (log / offer only, plug_in_charger +
insert_charger only) and the hidden coin check_held_small helper (not in Astra's tool list; harness-internal). Vendored from harness-v2 RC
rsi_v2/policy.py RouterV0 and rsi_v2/registry.py t_check_held_small; sources and shas are in the release manifest."""
import json
from pathlib import Path
import numpy as np
from .harness_l3 import gate_features

HERE = Path(__file__).resolve().parent


class RouterV0:
    name = 'router_v0'; modes_allowed = ('log', 'offer'); VIEWS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')

    def __init__(self, path=HERE / 'router_v0', device='cpu'):
        self.a = json.load(open(Path(path) / 'meta.json'))
        if self.a.get('verdict') != 'PASS': raise ValueError('ROUTER-v0 verdict is %s' % self.a.get('verdict'))
        import torch, torch.nn as nn
        class Router(nn.Module):   # verbatim architecture of macro-tools-20261027/gate/train_router_v0.py
            def __init__(self, d_tab, d_fact, d_pw=0, d=128):
                super().__init__()
                self.img = nn.Linear(1536, d); self.view = nn.Parameter(torch.zeros(3, d)); self.tab = nn.Linear(d_tab, d); self.fact = nn.Linear(d_fact, d)
                self.pw = nn.Linear(d_pw, d) if d_pw else None; self.cls = nn.Parameter(torch.zeros(1, 1, d))
                self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 256, dropout=0.1, batch_first=True), 2); self.out = nn.Linear(d, 1)
            def forward(self, xi, xt, xf):
                toks = [self.img(xi) + self.view, self.tab(xt)[:, None], self.fact(xf)[:, None]]
                h = torch.cat([self.cls.expand(xi.shape[0], -1, -1)] + toks, 1); return self.out(self.enc(h)[:, 0, :])[:, 0]
        self.torch = torch; self.dev = torch.device(device); self.models = []
        for f in sorted(Path(path).glob('model_s*.pt')):
            m = Router(22, 4); m.load_state_dict(torch.load(f, map_location='cpu')); m.eval().to(self.dev); self.models.append(m)
        self.vit = None

    def _img_feats(self, images):
        torch = self.torch
        if self.vit is None:
            from transformers import AutoModel
            self.vit = AutoModel.from_pretrained('/lizekai/RSIws/svc/dinov3-weights/dinov3-vitb16', torch_dtype=torch.float32).to(self.dev).eval()
            self.nreg = self.vit.config.num_register_tokens
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1); self.std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)
        from PIL import Image
        ims = [torch.from_numpy(np.asarray(Image.open(images[v]).convert('RGB'), np.float32) / 255.0).permute(2, 0, 1) for v in self.VIEWS]
        with torch.no_grad():
            h = self.vit(pixel_values=(torch.stack(ims).to(self.dev) - self.mean) / self.std).last_hidden_state
            return torch.cat([h[:, 0], h[:, 1 + self.nreg:].mean(1)], 1).float()[None]

    def decide(self, obs):
        ex = obs.get('executor')
        if obs['task'] not in self.a['scope'] or ex != self.a['executor']:
            return dict(decision='astra', p=None, executor=ex, reason='router v0 scope: plug_in_charger with insert_charger only')
        f = gate_features(obs)
        if not (f['holding_l'] or f['holding_r']):   # router rows were all holds; no image pass without a hold (cost)
            return dict(decision='astra', p=None, executor=ex, reason='no hold')
        torch = self.torch; q = np.asarray(obs['states_hist'][-1], float)[np.r_[0:6, 7:13]]
        tab = [f['budget_used'], f['grip_l'], f['grip_r'], f['cmd_l'], f['cmd_r'], f['holding_l'], f['holding_r'], f['closes_l'], f['closes_r'], f['travel_rad'] / 10.0] + [round(float(v), 4) / 3.0 for v in q]
        tf = obs.get('tool_facts') or {}; lia = float(tf.get('lift_inhand_abstain', 0)); fac = [1.0, 0.0, float(tf.get('chain_abstained', lia)), lia]
        xi = self._img_feats(obs['images']); xt = torch.tensor([tab], dtype=torch.float32, device=self.dev); xf = torch.tensor([fac], dtype=torch.float32, device=self.dev)
        with torch.no_grad(): p = float(np.mean([torch.sigmoid(m(xi, xt, xf)).item() for m in self.models]))
        return dict(decision='intervene' if (p > self.a['threshold'] and not lia) else 'astra', p=round(p, 4), executor=ex, reason='router v0')


HELD_PX = 25.0


def check_held_small(images_wrist_close, image_wrist_now, device='cpu'):
    """Hidden (not listed to Astra; dispatcher 14:3xZ): coin post-lift check pending re-validation of its reference-frame rule."""
    import importlib.util as U, torch, torch.nn.functional as F
    from PIL import Image
    sp = U.spec_from_file_location('sseg_train', '/lizekai/RSIws/experiments/smallseg-20261029/src/train_seg.py'); T = U.module_from_spec(sp); sp.loader.exec_module(T)
    net = T.Net(); ck = torch.load('/lizekai/RSIws/experiments/smallseg-20261029/models/sseg-v1.pt', map_location='cpu'); net.load_state_dict(ck['state'], strict=False); net.eval().to(device)
    def cen(p):
        x = torch.from_numpy(np.asarray(Image.open(p).convert('RGB'), np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
        with torch.no_grad(): m = (F.interpolate(net(x), size=(480, 640), mode='bilinear', align_corners=False)[0, T.CLS.index('coin')] > 0).cpu().numpy()
        ys, xs = np.nonzero(m); return None if len(xs) < 20 else (float(xs.mean()), float(ys.mean()))
    m0, m1 = cen(images_wrist_close), cen(image_wrist_now)
    if m0 is None: return 'unsure'
    return 'yes' if (m1 is not None and np.hypot(m1[0] - m0[0], m1[1] - m0[1]) <= HELD_PX) else 'no'
