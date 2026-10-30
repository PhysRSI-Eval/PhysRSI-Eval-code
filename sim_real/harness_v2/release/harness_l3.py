"""hv2e vendored copy of harness-v2 RC rsi_v2/policy.py (GateV2 + gate_features) and rsi_v2/steplog.py (StepLog). Byte-level sources and shas
are listed in the release manifest; keep in sync only through a new release."""
import json
from pathlib import Path
import numpy as np


def gate_features(obs):
    S = np.asarray(obs['states_hist'], float); A = np.asarray(obs['cmd_hist'], float); q = S[-1]; arm = np.r_[0:6, 7:13]
    travel = float(np.abs(np.diff(S[:, arm], axis=0)).sum()) if len(S) > 1 else 0.0
    closes = {i: int(np.sum((A[1:, 6 + 7 * i] < 0.5) & (A[:-1, 6 + 7 * i] >= 0.5))) for i in (0, 1)}
    return dict(budget_used=obs['step'] / float(obs['step_limit']), grip_l=float(q[6]), grip_r=float(q[13]), cmd_l=float(A[-1, 6]), cmd_r=float(A[-1, 13]),
                holding_l=float(A[-1, 6] < 0.5 and 0.02 < q[6] < 0.9), holding_r=float(A[-1, 13] < 0.5 and 0.02 < q[13] < 0.9),
                closes_l=closes[0], closes_r=closes[1], travel_rad=travel)


class GateV2:
    def __init__(self, path):
        self.a = json.load(open(path)); self.w = np.asarray(self.a['coef'], float); self.b = float(self.a['intercept'])

    def decide(self, obs):
        ex = obs.get('executor'); kind = self.a['executor_kind_map'].get(ex or 'insert_charger')
        if obs['task'] not in self.a['tasks']: return dict(decision='astra', p=None, executor=ex, reason='task not covered')
        f = gate_features(obs)
        x = [f['budget_used'], f['grip_l'], f['grip_r'], f['cmd_l'], f['cmd_r'], f['holding_l'], f['holding_r'], f['closes_l'], f['closes_r'], f['travel_rad'] / 10.0] + \
            [float(obs['task'] == t) for t in self.a['tasks']] + [float(kind == k) for k in self.a['kinds']]
        p = float(1.0 / (1.0 + np.exp(-(self.w @ np.asarray(x) + self.b)))); lo, hi = self.a['abstain_band']
        d = 'intervene' if p > self.a['intervene_above'] else ('abstain' if lo <= p <= hi else 'astra')
        return dict(decision=d, p=round(p, 4), executor=ex, reason='gate v2')


class StepLog:
    def __init__(self, output_dir):
        self.root = Path(output_dir); self.inner = None; self.k = 0; self.S = []; self.A = []

    def __call__(self):
        if self.inner is None:
            c = [p for p in self.root.iterdir() if p.is_dir() and (p / 'action_000000.json').exists()] if self.root.is_dir() else []
            if len(c) == 1: self.inner = c[0]
        if self.inner is None: return self.S, self.A
        while True:
            f = self.inner / ('action_%06d.json' % self.k)
            if not f.exists(): break
            try: r = json.loads(f.read_text())
            except (ValueError, OSError): break
            st = (r.get('obs') or {}).get('states')
            if st: self.S.append(st)
            self.A.append(r['executed_action']); self.k += 1
        return self.S, self.A
