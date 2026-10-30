"""PhysWord-style atomic action-word labels from proprio + gripper only (2026-10-29, rsiws-dc taking over rsiws-ee; dev, no judgement).
原理：PhysWord（NVIDIA，2026）用 GPT 把示范切成原子动作段（approach/pick/move/place/insert/...），每段的动词作为"语义锚"，
再让该动词 token 的隐状态去对齐未来帧的潜变量。我们这里不调用 VLM，只用夹爪指令和末端位姿做通用切分（不写任务规则）：
  - 夹爪从开到闭的时刻 = pick 段中心；闭合后到再次张开之间 = 持握；张开时刻 = 释放段中心；
  - 释放动词从指令中的通用动词取（insert/plug -> insert，hand over 且另一臂此时闭合 -> handover，否则 place）；
  - 持握且腕部转动大于平移 -> rotate，持握平移 -> move，持握静止 -> hold；张开时移动 -> approach，张开不动 -> idle。
输出每臂每步一个词，供 MVA 的 PW 臂做 CE 监督；同时报告词分布和段数，用来判断标签是否够用。
  python -I label_words.py <arrays_dir> <out_json>"""
import json, sys
from pathlib import Path
import numpy as np

VOCAB = ['idle', 'approach', 'pick', 'hold', 'move', 'rotate', 'place', 'insert', 'handover', 'strike', 'push']
OPEN_T, HALF = 0.85, 6         # 全开为 1.0；低于 0.85 视为闭合（瓶子只合到约 0.69，插头约 0.36）；pick/释放段取事件前后各 6 步（25 Hz 下约 0.25 s）
V_MOVE, W_ROT = 0.02, 0.6      # 平移速度 m/s、腕部角速度 rad/s 阈值


def ang_speed(q, freq):
    # 四元数相邻帧夹角 -> 角速度；q 为 (T,4)，顺序不影响 |dot|
    d = np.abs(np.sum(q[1:] * q[:-1], axis=1)).clip(0, 1)
    return np.r_[0.0, 2 * np.arccos(d) * freq]


def release_verb(instr, other_closed):
    s = instr.lower()
    if ('hand' in s and 'over' in s) and other_closed: return 'handover'
    if 'insert' in s or 'plug' in s: return 'insert'
    if 'strike' in s or 'hit' in s: return 'strike'
    if 'push' in s: return 'push'
    return 'place'


def label_arm(d, arm, other):
    freq = int(d['freq']); T = int(d['T'])
    g = d['action/%s_ee_joint_states' % arm][:, 0]; go = d['action/%s_ee_joint_states' % other][:, 0]
    p = d['state/%s_ee_poses' % arm]
    v = np.r_[0.0, np.linalg.norm(np.diff(p[:, :3], axis=0), axis=1) * freq]; w = ang_speed(p[:, 3:7], freq)
    closed = g < OPEN_T; instr = str(d['instruction'])
    lab = np.array(['idle'] * T, dtype=object)
    lab[(~closed) & (v > V_MOVE)] = 'approach'
    hold = closed & ((v > V_MOVE) | (w > W_ROT))
    lab[hold & (w > W_ROT) & (w / np.maximum(v, 1e-6) > 10)] = 'rotate'
    lab[hold & (lab != 'rotate')] = 'move'
    lab[closed & ~hold] = 'hold'   # 闭合且静止：持物等待或另一臂在做事
    # 没有物体时一直闭合在推（push_T 的夹爪指令全程约 0）：闭合且从未张开过的移动记为 push
    if closed.all(): lab[(v > V_MOVE)] = 'push'
    ev = np.flatnonzero(np.diff(closed.astype(int)))
    for e in ev:
        sl = slice(max(0, e - HALF), min(T, e + HALF + 1))
        if closed[e + 1]: lab[sl] = 'pick'
        else: lab[sl] = release_verb(instr, bool(go[e] < OPEN_T))
    return lab, len(ev)


def main(src, out):
    rows, tot = [], {}
    for f in sorted(Path(src).glob('*/*.npz')):
        d = np.load(f)
        for arm, other in (('left', 'right'), ('right', 'left')):
            lab, nev = label_arm(d, arm, other)
            u, c = np.unique(lab, return_counts=True); dist = {k: int(n) for k, n in zip(u, c)}
            seg = int(1 + np.sum(lab[1:] != lab[:-1]))
            rows.append(dict(task=f.parent.name, ep=f.stem, arm=arm, T=int(d['T']), gripper_events=nev, segments=seg, dist=dist,
                             words=[str(x) for x in lab]))
            for k, n in dist.items(): tot[k] = tot.get(k, 0) + n
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(vocab=VOCAB, params=dict(OPEN_T=OPEN_T, HALF=HALF, V_MOVE=V_MOVE, W_ROT=W_ROT), totals=tot, rows=rows), open(out, 'w'))
    print('episodes x arms', len(rows), 'totals', tot)
    for r in rows: print(r['task'], r['ep'], r['arm'], 'T', r['T'], 'events', r['gripper_events'], 'segments', r['segments'], r['dist'])


if __name__ == '__main__':
    main(*sys.argv[1:3])
