"""GT-DAgger P3 data source: C1 bc_rollout relabels (<out>/labels.jsonl: frames paths of the policy-visited state, state14, expert_joint
(H, 14) = the privileged expert's corrective chunk, expert_phase) -> DINOv3 features with the SAME function as the router cache / policy
(router_rgb.dino_features) -> one npz per labels file: X (n, 3, 1536) fp16, state (n, 14), expert (n, H, 14), phase, episode, step.
Rows without an expert chunk (charger not held / no grasp expert) are skipped and counted.
  /opt/PhysRSI/ENV/robodojo/bin/python dagger_feats.py LABELS.jsonl OUT.npz GPU"""
import json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, '/lizekai/RSIws/experiments/macro-tools-20261027/router-rgb'); import router_rgb as RR  # noqa: E402
from transformers import AutoModel
lab, out, gpu = sys.argv[1], sys.argv[2], int(sys.argv[3]); dev = torch.device('cuda:%d' % gpu)
vit = AutoModel.from_pretrained(RR.DINO).to(dev).eval(); nreg = vit.config.num_register_tokens
X, S, E, PH, EP, ST, INF, HA, OC = [], [], [], [], [], [], [], [], []; skip = {}
for l in open(lab):
    r = json.loads(l)
    if r.get('expert_joint') is None: skip[r.get('expert_phase')] = skip.get(r.get('expert_phase'), 0) + 1; continue
    ims = {c: np.asarray(Image.open(p).convert('RGB')) for c, p in r['frames'].items()}
    X.append(RR.dino_features(vit, nreg, ims, dev).astype(np.float16)); S.append(r['state']); E.append(r['expert_joint']); PH.append(r['expert_phase']); EP.append(r['episode']); ST.append(r['step'])
    INF.append(json.dumps(r.get('expert_info'))); HA.append(r.get('hold_arm', '')); OC.append(r.get('rollout_outcome', ''))
np.savez(out, X=np.stack(X), state=np.asarray(S, np.float32), expert=np.asarray(E, np.float32), phase=np.asarray(PH), episode=np.asarray(EP), step=np.asarray(ST), info=np.asarray(INF), hold_arm=np.asarray(HA), outcome=np.asarray(OC), source=lab)
print(json.dumps(dict(labels=lab, kept=len(X), skipped=skip, H=len(E[0]) if E else None)))
