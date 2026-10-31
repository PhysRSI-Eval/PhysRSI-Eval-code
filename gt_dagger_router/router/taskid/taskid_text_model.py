"""Deploy wrapper of models/taskid_text.pt (same network as taskid_text.M with use_t = use_i = 1; kept import-light: taskid_text.py
loads its training data at import)."""
import re
import numpy as np, torch, torch.nn as nn


class TextImageTaskID:
    def __init__(self, ck):
        self.voc, self.tasks = ck['vocab'], ck['tasks']; self.mu, self.sd = np.asarray(ck['mu'], np.float32), np.asarray(ck['sd'], np.float32)
        self.emb = nn.EmbeddingBag(len(self.voc), 64, mode='mean'); self.mlp = nn.Sequential(nn.Linear(64 + 4608, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, len(self.tasks)))
        sd = ck['state_dict']; self.emb.load_state_dict({'weight': sd['emb.weight']}); self.mlp.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith('mlp.')}); self.mlp.eval()

    def __call__(self, instruction, f4608):
        ids = [self.voc.get(w, 0) for w in re.findall(r"[a-z]+", instruction.lower())] or [0]
        with torch.no_grad():
            h = torch.cat([self.emb(torch.tensor(ids), torch.tensor([0])), torch.tensor((f4608[None] - self.mu) / self.sd, dtype=torch.float32)], 1)
            return self.tasks[int(self.mlp(h).argmax())]
