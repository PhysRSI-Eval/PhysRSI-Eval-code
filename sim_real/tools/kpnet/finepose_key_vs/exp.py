"""finepose experiments B / C (reference-script arms) on the shared library (PLAN-BCDEF.md). Truth only for setup (dev) and analysis.
  B-ref : aligned held state (official grasp, keyway yaw, hover 45 mm) -> reference straight descent + turn through Env.move
  C-ref : same held state -> lift 30 mm + seeded perturbation (dx, dy in +-10 mm, dyaw in +-20 deg) by robot moves ->
          realign with target_hand(slot goal, in-hand G[perception]) -> reference descent + turn
  python exp.py --mode B-ref|C-ref --seed N [--perc truth] --gpu G --out DIR"""
import argparse, json, math, sys, time, traceback
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fp_lib as F  # noqa: E402


def perturb(env, seed, lift=30.0, xy=10.0, yaw=20.0):
    rng = np.random.default_rng(seed); dx, dy, dyw = rng.uniform(-xy, xy), rng.uniform(-xy, xy), rng.uniform(-yaw, yaw)
    for _ in range(int(lift // 5)): env.move(dz_mm=5, tag='perturb_lift')
    n = int(max(abs(dx), abs(dy)) // 5) + 1; m = int(abs(dyw) // 15) + 1
    for _ in range(n): env.move(dx_mm=dx / n, dy_mm=dy / n, tag='perturb_xy')
    for _ in range(m): env.move(dyaw_deg=dyw / m, tag='perturb_yaw')
    return dict(dx_mm=round(dx, 2), dy_mm=round(dy, 2), dyaw_deg=round(dyw, 2), lift_mm=lift)


def first_failure(env, res):
    ev = {e['tag']: e for e in env.events}; c = res.get('checks', {}).get('items', [0, 0, 0, 0])
    if res.get('native_success'): return None
    al = ev.get('realigned') or ev.get('held')
    if al and (max(abs(v) for v in al['key_vs_slot_xy_mm']) > 1.0 or abs(al['key_yaw_vs_keyway_deg']) > 5): return 'align'
    post = env.events[next((i for i, e in enumerate(env.events) if e['tag'] == 'held'), len(env.events)):]
    if max((e.get('insertion_depth_mm', -99) for e in post), default=-99) < 25: return 'descend'
    return 'turn'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', required=True); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--perc', default='truth'); ap.add_argument('--err', default='')  # TOL: dx_mm,dy_mm,dyaw_deg,tiltx_deg,tilty_deg (world)
    ap.add_argument('--gpu', type=int, required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
    res = dict(args=vars(a)); env = None; t0 = time.time()
    try:
        env = F.Env(a.out, a.gpu); ok = env.setup_held(); env.snap('held')
        if a.err:
            e_ = [float(v) for v in a.err.split(',')] + [0.0] * 5; E = np.eye(4); E[:3, 3] = [e_[0] / 1000, e_[1] / 1000, 0.0]
            E[:3, :3] = F.R.from_euler('zxy', [e_[2], e_[3], e_[4]], degrees=True).as_matrix(); env.goal_err = E; res['goal_err'] = e_[:5]
        if not ok: res['first_failure'] = 'setup'; raise StopIteration
        s0 = env.ad.packet['step_id']
        if a.mode == 'C-ref':
            res['perturbation'] = perturb(env, a.seed); env.snap('perturbed')
            G = env.G() if a.perc == 'truth' else None  # 'estimate' plugs in later (experiment F)
            L = F.target_hand(env.goal(0.045), G); res['realign'] = env.move_to(L, 'realigned', 150); env.snap('realigned')
        n = env.ref_insert(); res['insert_try'] = n; env.snap('descended'); env.ref_turn(); env.snap('turned')  # v3
        res['action_steps'] = int(env.ad.packet['step_id']) - s0; res['descend_moves'] = n
    except StopIteration:
        pass
    except Exception as e:  # noqa: BLE001
        res['error'] = '%s: %s' % (type(e).__name__, str(e)[:300]); res['trace'] = traceback.format_exc()[-1200:]
    finally:
        if env is not None:
            try:
                res['checks'] = env.checks(); res['terminal'] = env.ad.terminal_reason
                res['native_success'] = bool((env.ad.native_end or {}).get('success')) or res['checks'].get('all', 0) >= 1.0
            except Exception as e:  # noqa: BLE001
                res['check_error'] = str(e)[:200]; res['native_success'] = env.ad.terminal_reason == 'native_terminal'
            res['first_failure'] = res.get('first_failure') or (None if res.get('native_success') else ('engineering' if res.get('error') else first_failure(env, res)))
            res['final_truth'] = env.events[-1] if env.events else None; post = env.events[next((i for i, e in enumerate(env.events) if e['tag'] == 'held'), len(env.events)):]  # v2: only after the key is held
            res['max_depth_mm'] = max((e.get('insertion_depth_mm', -99) for e in post), default=None)
            res['max_inhand_shift_mm'] = max((e.get('inhand_shift_mm', 0) for e in env.events), default=None); env.close_sim()
        res['seconds'] = round(time.time() - t0); Path(a.out, 'result.json').write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps(dict(out=Path(a.out).name, success=res.get('native_success'), first_failure=res.get('first_failure'), max_depth=res.get('max_depth_mm'),
                              realign=res.get('realign'), error=res.get('error')), default=str), flush=True)


if __name__ == '__main__':
    main()
