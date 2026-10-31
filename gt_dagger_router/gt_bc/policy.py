"""K1 insert-skill BC policy, callable from the C1 sim harness (GT-DAgger P1/P2; tool line ip4).
    from policy import InsertSkillPolicy
    pi = InsertSkillPolicy('/lizekai/RSIws/experiments/gt-bc-20261031/models/bc-r2-k1.pt', gpu=4)
    pi.reset()                                   # at episode start (clears the state/action history)
    chunk = pi(obs)                              # (H, 14) ABSOLUTE next joint states, 25 Hz
obs = dict(task='plug_in_charger', state=(14,) [L arm 6, L grip 1 (1 open .. 0 closed), R arm 6, R grip 1],
           images={'cam_high': HxWx3 uint8 RGB, 'cam_left_wrist': ..., 'cam_right_wrist': ...})   # the deploy cameras, RGB only
Semantics (P0, GT-P0-NOTE.md): the official action[t] == state[t+1], so chunk[j] is the joint state to reach at step t+1+j. The
harness should send chunk[0..E-1] as absolute joint-position targets (E = 8 matches the offline free-running evaluation), call
pi.executed(actions) with what it actually sent, and query again. Features = router_rgb.dino_features (the same function and
DINOv3 ViT-B/16 weights as the cache the policy was trained on)."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, '/lizekai/RSIws/experiments/macro-tools-20261027/router-rgb')
import bcnet as B  # noqa: E402
import router_rgb as RR  # noqa: E402


class InsertSkillPolicy:
    def __init__(self, ckpt, gpu=0):
        self.dev = torch.device('cuda:%d' % gpu if torch.cuda.is_available() else 'cpu'); ck = torch.load(ckpt, map_location=self.dev)
        self.H, self.hist, self.target = ck['H'], ck['hist'], ck['target']; self.nm = B.Norm(ck['norm'])
        self.net = B.BC(self.H, self.hist, ntask=ck.get('ntask', 4)).to(self.dev); self.net.load_state_dict(ck['state']); self.net.eval()
        from transformers import AutoModel
        self.vit = AutoModel.from_pretrained(RR.DINO).to(self.dev).eval(); self.nreg = self.vit.config.num_register_tokens; self.reset()

    def reset(self):
        self.S, self.A = [], []  # 已执行步的状态 / 动作历史（与训练时 GT 历史同义）

    def executed(self, actions):
        # 不变量：S[i] / A[i] = 第 i 步的状态 / 动作。官方语义 state[t+1] == action[t]（P0），所以两次查询之间未观测的中间状态 =
        # 已执行动作的前 n-1 个；下一次 __call__ 再追加真实观测状态。这样历史与训练时的逐步 GT 历史同义（harness 每 exec_n 步才查询一次）。
        acts = [np.asarray(x, np.float32) for x in actions]; self.A.extend(acts); self.S.extend(acts[:-1])

    @torch.no_grad()
    def __call__(self, obs):
        s = np.asarray(obs['state'], np.float32); self.S.append(s); X = RR.dino_features(self.vit, self.nreg, obs['images'], self.dev).astype(np.float32)
        n = self.hist; hs = [self.S[-1 - j] if len(self.S) > j else self.S[0] for j in range(n, 0, -1)]
        ha = [self.A[-j] if len(self.A) >= j else self.S[0] for j in range(n, 0, -1)]; hm = np.array([1.0 if len(self.A) >= j else 0.0 for j in range(n, 0, -1)], np.float32)
        h = np.concatenate([(np.stack(hs) - self.nm.sm) / self.nm.ss, (np.stack(ha) - self.nm.am) / self.nm.as_], 1) if n else np.zeros((0, 28), np.float32)
        T = lambda x: torch.from_numpy(np.asarray(x, np.float32)[None]).to(self.dev)
        y = self.net(T(X), T((s - self.nm.sm) / self.nm.ss), T(h), T(hm), torch.tensor([B.TASKS.index(obs['task'])], device=self.dev)).cpu().numpy()[0]
        return self.nm.dec_y(y, s, self.target)


# ---- C1 harness contract (reports/progress/mailbox/to-ip4.md 10-31 02:5xZ): load(ckpt) -> obj; obj.act(obs) -> {"kind": "joint", "actions": (H, 14)}
TASK_WORDS = (('charger', 'plug_in_charger'), ('key', 'insert_key'), ('tube', 'insert_tubes'), ('coin', 'deposit_coin'))


class _Harness:
    def __init__(self, ckpt, exec_n=8, gpu=0):
        self.pi = InsertSkillPolicy(ckpt, gpu); self.exec_n = exec_n; self.last = None; self.step = None

    def act(self, obs):
        st = int(obs.get('step', 0))
        if self.step is None or st <= self.step and st == 0: self.pi.reset(); self.last = None
        elif self.last is not None: self.pi.executed(self.last[:self.exec_n])  # 假定 harness 执行了上一块的前 exec_n 步
        self.step = st; ins = str(obs.get('instruction', '')).lower(); task = obs.get('task') or next((t for w, t in TASK_WORDS if w in ins), 'plug_in_charger')
        ch = self.pi(dict(task=task, state=obs['state'], images=obs['images'])); self.last = ch
        return {'kind': 'joint', 'actions': np.asarray(ch, np.float32)}


def load(ckpt, exec_n=8):
    import os
    return _Harness(ckpt, exec_n, int(os.environ.get('BC_GPU', '0')))
