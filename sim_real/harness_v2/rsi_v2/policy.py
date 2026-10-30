"""L3 learned intervention policy (harness v2; bottleneck line, 10-29 12:xxZ). DIRECTION-v6 s1: intervene only where the intervention beats
Astra alone from that state; the decision uses deploy inputs only.

Interface every L3 policy implements (gate v2 now; ROUTER-v0 plugs in here as its successor):
  policy.decide(obs) -> {decision: 'astra' | 'intervene' | 'abstain', p: float | None, executor: str | None, reason: str}
  obs = {task, step, step_limit, states_hist: [T x 14], cmd_hist: [T x 14] (commanded gripper values at 6 and 13, 1 open / 0 closed),
         executor: name of the task's deploy executor in L1 (e.g. 'insert_charger'),
         images: {cam_high, cam_left_wrist, cam_right_wrist} (current frames; used by ROUTER-v0, ignored by gate v2),
         tool_facts: {...} (facts already available without motion, e.g. inhand_v4_abstain_at_lift; ROUTER-v0 input)}
Modes (harness cfg 'l3_policy': {name, mode, executor, auto_approved}):
  'log'   decisions are recorded only (shadow mode; no text, no motion) -> paired data for the next gate/router;
  'offer' the first 'intervene' decision surfaces ONE availability fact for the executor through the L3 advice filter and rate limit;
          the executor still runs only if Astra calls it (R3);
  'auto'  the harness calls the executor itself (explicit=POLICY_TOKEN). Allowed only with auto_approved=true, which needs a
          pre-registered online design and first-hand user approval; otherwise 'auto' is downgraded to 'offer' and the downgrade is logged.
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def gate_features(obs):
    """Exactly gate/build_dataset_v2.features (budget_used, grip, last cmd, holding, closes, travel) from the deploy history."""
    S = np.asarray(obs['states_hist'], float); A = np.asarray(obs['cmd_hist'], float); q = S[-1]; arm = np.r_[0:6, 7:13]
    travel = float(np.abs(np.diff(S[:, arm], axis=0)).sum()) if len(S) > 1 else 0.0
    closes = {i: int(np.sum((A[1:, 6 + 7 * i] < 0.5) & (A[:-1, 6 + 7 * i] >= 0.5))) for i in (0, 1)}
    return dict(budget_used=obs['step'] / float(obs['step_limit']), grip_l=float(q[6]), grip_r=float(q[13]), cmd_l=float(A[-1, 6]), cmd_r=float(A[-1, 13]),
                holding_l=float(A[-1, 6] < 0.5 and 0.02 < q[6] < 0.9), holding_r=float(A[-1, 13] < 0.5 and 0.02 < q[13] < 0.9),
                closes_l=closes[0], closes_r=closes[1], travel_rad=travel)


class GateV2:
    """Frozen logistic gate (policies/gate_v2.json, fitted by tools/fit_gate_v2_final.py; CV PASS 0.178 vs always-Astra 0.061 vs always-executor 0.135)."""
    name = 'gate_v2'

    def __init__(self, path=HERE / 'policies' / 'gate_v2.json'):
        self.a = json.load(open(path)); self.w = np.asarray(self.a['coef'], float); self.b = float(self.a['intercept'])

    def decide(self, obs):
        ex = obs.get('executor'); kind = self.a['executor_kind_map'].get(ex)
        if kind is None or obs['task'] not in self.a['tasks']:
            return dict(decision='astra', p=None, executor=ex, reason='no validated executor for this task')
        f = gate_features(obs)   # no extra precondition here: the gate was trained on all rg3f rows (7/52 had no 'holding' flag); the executor checks the hold itself
        x = [f['budget_used'], f['grip_l'], f['grip_r'], f['cmd_l'], f['cmd_r'], f['holding_l'], f['holding_r'], f['closes_l'], f['closes_r'], f['travel_rad'] / 10.0] + \
            [float(obs['task'] == t) for t in self.a['tasks']] + [float(kind == k) for k in self.a['kinds']]
        p = float(1.0 / (1.0 + np.exp(-(self.w @ np.asarray(x) + self.b)))); lo, hi = self.a['abstain_band']
        d = 'intervene' if p > self.a['intervene_above'] else ('abstain' if lo <= p <= hi else 'astra')
        return dict(decision=d, p=round(p, 4), executor=ex, reason='gate v2 logistic, features %s' % {k: round(v, 3) for k, v in f.items()})


class RouterV0:
    """Successor of gate v2 (capability line ROUTER-v0, PASS single-task: plug_in_charger + frozen rg3f; dedup 67 rows, CV routed 0.254 vs
    always-Astra 0.075, always-rg3f 0.209, logistic 0.239; 13 takeovers, 0 harmful). Frozen deployment fit: policies/router_v0/{model_s*.pt,
    meta.json} (tools/fit_router_v0_final.py). Same decide(obs) contract as GateV2; uses obs['images'] (cam_high, cam_left_wrist,
    cam_right_wrist: paths or HxWx3) and obs['tool_facts'] ({'lift_inhand_abstain': 0/1}). Modes: log / offer only (never auto)."""
    name = 'router_v0'; modes_allowed = ('log', 'offer')
    VIEWS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')

    def __init__(self, path=HERE / 'policies' / 'router_v0', device='cpu'):
        meta = Path(path) / 'meta.json'
        if not meta.exists(): raise FileNotFoundError('ROUTER-v0 artifact not frozen: %s' % meta)
        self.a = json.load(open(meta))
        if self.a.get('verdict') != 'PASS': raise ValueError('ROUTER-v0 verdict is %s; gate v2 remains the L3 policy' % self.a.get('verdict'))
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
        ims = []
        for v in self.VIEWS:
            x = images[v]; a = np.asarray(Image.open(x).convert('RGB') if isinstance(x, (str, Path)) else x, np.float32) / 255.0
            ims.append(torch.from_numpy(a).permute(2, 0, 1))
        with torch.no_grad():
            h = self.vit(pixel_values=(torch.stack(ims).to(self.dev) - self.mean) / self.std).last_hidden_state
            return torch.cat([h[:, 0], h[:, 1 + self.nreg:].mean(1)], 1).float()[None]   # 1 x 3 x 1536

    def decide(self, obs):
        ex = obs.get('executor')
        if obs['task'] not in self.a['scope'] or ex != self.a['executor']:
            return dict(decision='astra', p=None, executor=ex, reason='router v0 scope: plug_in_charger with insert_charger only')
        torch = self.torch; f = gate_features(obs); q = np.asarray(obs['states_hist'][-1], float)[np.r_[0:6, 7:13]]
        tab = [f['budget_used'], f['grip_l'], f['grip_r'], f['cmd_l'], f['cmd_r'], f['holding_l'], f['holding_r'], f['closes_l'], f['closes_r'], f['travel_rad'] / 10.0] + [round(float(v), 4) / 3.0 for v in q]
        tf = obs.get('tool_facts') or {}; lia = float(tf.get('lift_inhand_abstain', 0)); fac = [1.0, 0.0, float(tf.get('chain_abstained', lia)), lia]   # deploy: chain_abstained := lift_inhand_abstain (meta.deploy_note)
        xi = self._img_feats(obs['images']); xt = torch.tensor([tab], dtype=torch.float32, device=self.dev); xf = torch.tensor([fac], dtype=torch.float32, device=self.dev)
        with torch.no_grad(): p = float(np.mean([torch.sigmoid(m(xi, xt, xf)).item() for m in self.models]))
        d = 'intervene' if (p > self.a['threshold'] and not lia) else 'astra'
        return dict(decision=d, p=round(p, 4), executor=ex, reason='router v0 (3-seed mean), lift_inhand_abstain=%d' % lia)


POLICIES = {'gate_v2': GateV2, 'router_v0': RouterV0}


def load(name='gate_v2'):
    try:
        return POLICIES[name]()
    except (FileNotFoundError, ValueError):
        if name == 'gate_v2': raise
        return GateV2()
