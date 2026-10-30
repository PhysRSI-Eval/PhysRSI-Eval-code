"""插入类（C1）难点普查：用仿真真值给 astra_v1 中每个 C1 回合标出"插入漏斗"上最远到达的阶段与步数（rsiws-dc，用户 10-29 16:5xZ）。
目的：先把插入类的问题分清楚——卡在哪一步、是工具问题（感知/精度/执行）还是 harness 问题（送达/步数/Astra 不用）。
阶段（每步判断，取第一次满足的步数）：
  S1 抓起：某 rigid 物体比初始高 >= 3 cm 且离某臂 TCP <= 6 cm（被插件 = 第一个满足的物体）
  S2 近目标：被插件仍在手中，且与接收方水平距离 <= 10 cm
  S3 悬停：在手中、高于接收方顶部、水平距离 <= 3 cm
  S4 粗对准：在手中或刚松开、水平距离 <= 1 cm
  S5 原生成功（评测器）
掉落：被插件曾被抓起，后来离开所有 TCP 15 cm 且高度回到初始 +1 cm 内，且离接收方 > 10 cm。
接收方按名字关键词匹配（TARGET）；匹配不上的回合单独列出，需人工核对。  python -I c1_census.py <out_json>"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train_k2 import Tm, TCP

A = Path('/lizekai/RSIws/experiments/mva-20261029/astra_v1'); NAT = Path('/lizekai/RSIws/artifacts/native')
C1 = ['plug_in_charger', 'insert_key', 'insert_tubes', 'deposit_coin', 'fill_pen_holder']
TARGET = {'plug_in_charger': ['strip', 'socket', 'board', 'outlet'], 'insert_key': ['key_slot', 'lock'], 'insert_tubes': ['test_tube_slot', 'rack'],
          'deposit_coin': ['bank', 'slot', 'piggy', 'box'], 'fill_pen_holder': ['holder', 'cup', 'container']}
SKIP = ('camera_stand', 'Ground', 'Simple_Room', 'material')


def tcps(d, t):
    out = []
    for arm in ('left', 'right'):
        e = d['state/%s_ee_poses' % arm][min(t, int(d['T']) - 1)]; E = Tm(e[:3], e[3:7]); out.append(E[:3, :3] @ TCP + E[:3, 3])
    return out


def census(task, ep):
    d = np.load(A / 'arrays' / task / ('%s.npz' % ep)); tr = json.load(open(A / 'truth' / task / ('%s.json' % ep)))
    steps = sorted(int(k) for k in tr); o0 = tr[str(steps[0])]
    pos = lambda o, n: np.asarray(o[n]['position']) if n in o and 'position' in o[n] else None
    tgt = [n for n in o0 if any(k in n for k in TARGET[task]) and 'position' in o0[n] and not n.startswith(SKIP)]
    rig = [n for n, v in o0.items() if v.get('kind') == 'rigid' and 'position' in v and n not in tgt]
    rec = dict(task=task, episode=ep, target=tgt[0] if tgt else None, held=None, stage={}, drop_step=None)
    try:
        oc = json.loads((NAT / ep / 'evaluation_outcome.json').read_text()); rec['native_success'] = bool(oc.get('native_success')); rec['step_limit'] = oc.get('native_step_limit')
    except Exception:  # noqa: BLE001  旧回合没有 outcome 文件
        rec['native_success'] = None
    rec['T'] = int(d['T'])
    if not tgt: return rec
    z0 = {n: pos(o0, n)[2] for n in rig}; held = None; was_lifted = False
    for t in steps:
        o = tr[str(t)]; tp = pos(o, tgt[0]); P = tcps(d, t)
        if held is None:
            for n in rig:
                p = pos(o, n)
                if p is not None and p[2] - z0[n] >= 0.03 and min(np.linalg.norm(p - q) for q in P) <= 0.06: held = n; rec['held'] = n; rec['stage'].setdefault('S1_grasp', t); was_lifted = True; break
            continue
        p = pos(o, held)
        if p is None or tp is None: continue
        in_hand = min(np.linalg.norm(p - q) for q in P) <= 0.06; dxy = float(np.linalg.norm(p[:2] - tp[:2]))
        if in_hand and dxy <= 0.10: rec['stage'].setdefault('S2_near', t)
        if in_hand and dxy <= 0.03 and p[2] > tp[2]: rec['stage'].setdefault('S3_hover', t)
        if dxy <= 0.01: rec['stage'].setdefault('S4_align', t)
        if was_lifted and rec['drop_step'] is None and min(np.linalg.norm(p - q) for q in P) > 0.15 and p[2] - z0[held] < 0.01 and dxy > 0.10: rec['drop_step'] = t
    if rec['native_success']: rec['stage']['S5_success'] = rec['T']
    return rec


def main(out):
    rows = {}
    for l in open(A / 'MANIFEST.jsonl'):
        r = json.loads(l); rows[r['episode']] = r
    R = [census(r['task'], ep) for ep, r in rows.items() if r.get('status') == 'used' and r['task'] in C1]
    order = ['S1_grasp', 'S2_near', 'S3_hover', 'S4_align', 'S5_success']; summ = {}
    for task in C1:
        rr = [r for r in R if r['task'] == task]; ok = [r for r in rr if r['target']]
        f = {s: sum(s in r['stage'] for r in ok) for s in order}
        med = {s: (int(np.median([r['stage'][s] for r in ok if s in r['stage']])) if f[s] else None) for s in order}
        summ[task] = dict(episodes=len(rr), target_found=len(ok), targets=sorted({r['target'] for r in ok}), held=sorted({r['held'] for r in ok if r['held']}),
                          funnel=f, median_step=med, drops=sum(r['drop_step'] is not None for r in ok), step_limit=sorted({r.get('step_limit') for r in rr} - {None}))
        print(task, json.dumps(summ[task]))
    json.dump(dict(summary=summ, rows=R), open(out, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
