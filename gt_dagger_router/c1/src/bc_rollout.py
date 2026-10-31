"""bc_rollout.py (C1 line, GT -> DAgger P2 + P3, 2026-10-31): roll out a BC "insert skill" policy in the simulator (zero quota, no Astra
online) from the plug handover states (loc1cov.py: replay Astra to its hold + 4.5 cm lift; --t 0 = the public layout start state), log
every chunk with frames, and at every policy-visited state write the PRIVILEGED EXPERT's corrective action chunk (DAgger relabel).

  python bc_rollout.py --policy POLICY.py [--ckpt PATH] [--budget 250] [--exec_n 8] [--H 16] src/loc1cov.py <loc1cov args>

Policy contract (POLICY.py, written by the tool line): `load(ckpt) -> obj` with
  obj.act(obs) -> dict(kind='joint', actions=(H, 14)) | dict(kind='eef', actions=(H, 16))
  obs = dict(images={'cam_high','cam_left_wrist','cam_right_wrist': HxWx3 uint8 RGB}, state=(14,) [left arm 6, left gripper, right arm 6,
             right gripper] (native state = official `state`), eef=(2, 7) link6 [x y z qw qx qy qz] per arm, instruction=str, step=int)
  joint: absolute native joint targets per tick (= the official `action/*_arm_joint_states` + `*_ee_joint_states`), executed with the
         simulator's raw action path (rpc replay, the path used to replay Astra's recorded actions);
  eef  : per tick [left link6 pos3 quat_wxyz4 grip1, right ...]; converted tick by tick to joint targets with the native EEF controller
         model (reach_model ArmFK.bounded_step = the simulator's own controller), then executed the same way.
The first exec_n actions of every chunk are executed, then the policy is queried again.

Expert label at every visited state (P3; truth = offline privileged labels only; never shown to the policy):
  truth CAD keypoints (prong tips / roots, hole entries) -> vs_exec.correction to the best outlet x pairing (reachable, smallest turn)
  phase 'align' (tips not within 0.5 mm / 1.5 deg / 2 mm of the hover 10 mm above the entries) -> target = that hover pose;
  phase 'insert' (aligned) -> target = the seated pose (tips 16.5 mm past the entries); phase 'release' (truth depth check passes) ->
  open the gripper; the other arm holds its joints. The chunk = H native-controller ticks toward the target from the measured joints
  (joint (H, 14) and eef (H, 16) forms). Written to <out>/labels.jsonl with the frame paths (same fields as obs + expert chunk).
Outcome per rollout: inserted (all 3 native checks) / budget / charger_dropped / ended, plus the step of the first failure signal."""
import importlib.util, json, math, os, runpy, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

C1 = Path(__file__).resolve().parent; V = '/lizekai/RSIws/experiments/macro-tools-20261027'
V14S = '/lizekai/RSIws/experiments/combined-online-20261026/plug-offline-eval/v14/src'
os.environ.setdefault('BC_GPU', sys.argv[sys.argv.index('--gpu') + 1] if '--gpu' in sys.argv else '0')   # tool line policy reads BC_GPU: put it on the job's GPU
OPT = dict(policy='', ckpt='', budget=250, exec_n=8, H=16, start=0, gripgate=0)
argv = sys.argv[1:]
while argv and argv[0].startswith('--') and argv[0][2:] in OPT:
    OPT[argv[0][2:]] = type(OPT[argv[0][2:]])(argv[1]); argv = argv[2:]
_po = subprocess.Popen


def _popen_c1(a_, **kw):   # server module pe_server_ext -> c1_server_ext (ops c1_frames / c1_contacts)
    a_ = [('c1_server_ext' if x == 'pe_server_ext' else x) for x in a_]
    if kw.get('env') is not None:   # c1/src (c1_server_ext) + v14/src (pe_server_ext) on the server path; loc1cov adds v14 itself, bc_start does not
        kw['env'] = dict(kw['env'], PYTHONPATH=os.pathsep.join([str(C1), V14S, kw['env'].get('PYTHONPATH', '')]))
    return _po(a_, **kw)


subprocess.Popen = _popen_c1
sys.path.insert(0, '/root/pe-1026/pkg'); sys.path.insert(0, '/lizekai/RSIws/experiments/combined-online-20261026/plug-offline-eval/v14/src')
from rsi import stage_tools as ST, combined_tools as CT  # noqa: E402
sp = importlib.util.spec_from_file_location('grip_cmd', V + '/pkg-v2/rsi/grip_cmd.py'); G = importlib.util.module_from_spec(sp); sp.loader.exec_module(G)
G.install(ST._Tool)
sys.path.insert(0, V + '/src'); sys.path.insert(0, str(C1))
import loc1cov_lib as LC  # noqa: E402
import vs_exec as VX  # noqa: E402
import v14_sim as _pe_sim  # noqa: E402
_Sim0 = _pe_sim.RA.Sim
_pe_sim.RA.Sim = lambda task, out, gpu, *a, **k: _Sim0(task, Path(out) if Path(out).name.startswith('c1-') else Path(out).with_name('c1-' + Path(out).name), gpu, *a, **k)

HELD = {'tip_l': (-0.0055, -0.0152, 0.0), 'tip_r': (0.0055, -0.0152, 0.0), 'root_l': (-0.0055, 0.0, 0.0), 'root_r': (0.0055, 0.0, 0.0)}
SPEC = dict(held_pts=['tip_l', 'tip_r'], held_axis=(['root_l', 'root_r'], ['tip_l', 'tip_r']), targets={k: ['o%d_hole_a' % k, 'o%d_hole_b' % k] for k in range(3)},
            pairings=[(0, 1), (1, 0)], axis=(0.0, 0.0, -1.0), hover_m=0.010, depth_m=0.0165)
RELEASE_DXY_MM = 3.0   # release only when all native checks pass, or depth passes with the tips within 3 mm (xy) of the hole entries
CHECKS = [['is_A_depth_in_B', dict(label_A='charger', label_B='socket', z_threshold=0.015)], ['is_A_in_B', dict(label_A='charger', label_B='socket')],
          ['is_axis_up', dict(label='charger', axis=[0, 1, 0], threshold=10)]]



def _ld(n, f):
    s = importlib.util.spec_from_file_location(n, f); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


RM = _ld('reach_model_v15', V + '/pkg-v15/rsi/reach_model.py')


class Rollout:
    def __init__(self, rec, co, tool, ad, rpc, objs, arm, out):
        self.rec, self.co, self.tool, self.ad, self.rpc, self.objs, self.arm, self.out = rec, co, tool, ad, rpc, objs, arm, Path(out)
        self.rm = RM.Reach(); self.lab = open(self.out / 'labels.jsonl', 'a'); self.log = []; self.s0 = None

    # ------------------------------------------------------------ simulator state (raw action path)
    def obs(self):
        o = self.rpc('observe'); fr = self.rpc('c1_frames', dir=str(self.out / 'frames'))
        from PIL import Image
        ims = {c: np.asarray(Image.open(p).convert('RGB')) for c, p in fr['paths'].items()}
        st = np.asarray(o['states'], float); eef = np.c_[np.asarray(o['eef_positions'], float), np.asarray(o['eef_quaternions_wxyz'], float)]
        if self.s0 is None: self.s0 = int(fr['step_id'])
        return dict(images=ims, state=st, eef=eef, instruction=o.get('instruction', ''), step=int(fr['step_id'])), fr['paths']

    def l6(self, eef, arm): i = CT.ARMS.index(arm); return CT.T_of(eef[i][:3], eef[i][3:])

    def eef_to_joint(self, state, acts):
        """(H, 16) link6 targets -> (H, 14) joint targets with the native EEF controller model, one bounded step per tick."""
        q = {a: np.asarray(state[7 * i:7 * i + 6], float) for i, a in enumerate(CT.ARMS)}; out = []
        for row in np.asarray(acts, float):
            a14 = []
            for i, a in enumerate(CT.ARMS):
                t = row[8 * i:8 * i + 8]; goal = dict(position=t[:3], quaternion_wxyz=t[3:7])
                q[a] = np.asarray(self.rm.fk[a].bounded_step(q[a], self.rm.lim[a], self.rm.root[a], goal), float); a14 += list(q[a]) + [float(t[7])]
            out.append(a14)
        return np.asarray(out)

    def execute(self, joint_chunk):
        rows = self.rpc('replay', actions=[[float(x) for x in a] for a in joint_chunk]); return rows[-1] if rows else {}

    # ------------------------------------------------------------ truth (labels / scoring only)
    def truth_world(self):
        o = self.objs()[LC.CH]; Tc = LC._T(o['position'], o['quaternion_wxyz']); P = {n: Tc[:3, :3] @ np.array(v) + Tc[:3, 3] for n, v in HELD.items()}
        gb = self.rpc('rgbhint_geom', bbox=['socket_0_2'])['bbox']['socket_0_2']; Ts = LC._T(gb['pose'][:3], gb['pose'][3:])
        for k, ox in enumerate(CT.OFFS):
            for n, dx in (('hole_a', -0.0055), ('hole_b', 0.0055)):
                p = Ts[:3, :3] @ np.array([ox + dx, CT.OUT_Y, 0.0]) + Ts[:3, 3]; p[2] = gb['z_max']; P['o%d_%s' % (k, n)] = p
        return P, Tc

    def checks(self): return [float(x) for x in (LC.RA_plain(self.rpc('rgbhint_check', checks=CHECKS)).get('items') or [0, 0, 0])]

    def expert(self, ob):
        """Privileged corrective chunk at this visited state: (phase, joint (H, 14), eef (H, 16), info)."""
        P, Tc = self.truth_world(); st = ob['state']; L = self.l6(ob['eef'], self.arm); i = CT.ARMS.index(self.arm)
        tcp = L[:3, 3] + L[:3, :3] @ CT.TCP
        if np.linalg.norm(Tc[:3, 3] - tcp) > 0.08: return 'charger_not_held', None, None, dict()
        q = np.asarray(st[7 * i:7 * i + 6], float); best = None
        for tid, names in SPEC['targets'].items():
            for pi, perm in enumerate(SPEC['pairings']):
                C, res = VX.correction(P, [P[n] for n in names], SPEC, perm)
                if C is None: continue
                Lh = C @ L; ok = self.rm.go(self.arm, q, Lh)[3]; key = (not ok, res['rot_deg'] + (20.0 if pi else 0.0))
                if best is None or key < best[0]: best = (key, tid, pi, Lh, res)
        _, tid, pi, Lh, res = best; ck = self.checks(); grip = float(st[7 * i + 6])
        if min(ck) >= 1.0 or (ck[0] >= 1.0 and res['dxy_mm'] <= RELEASE_DXY_MM):   # 10-31 fix (tool line gripper check): check[0] alone fired
            phase, goal, grip = 'release', L, 1.0                                 # 473 mm (median) from the socket -> taught 'let go' far away
        elif VX.converged(res, dict(tol_mm=0.5, tol_deg=1.5, tol_dz_mm=2.0, tol_tilt_deg=3.0)):
            phase = 'insert'; goal = Lh.copy(); goal[:3, 3] += np.asarray(SPEC['axis']) * (SPEC['hover_m'] + SPEC['depth_m'])
        else:
            phase, goal = 'align', Lh
            if np.linalg.norm(Lh[:2, 3] - L[:2, 3]) > 0.05:   # far: travel at a safe height first (as the VS executor's first move), not a straight line through the strip
                goal = Lh.copy(); goal[2, 3] = max(float(L[2, 3]), float(Lh[2, 3]) + 0.05); phase = 'align_transport'
        H = OPT['H']; e16 = []
        for k in range(H):   # eef form: the goal for the holding arm every tick, the other arm holds its link6
            row = []
            for j, a in enumerate(CT.ARMS):
                M = goal if a == self.arm else self.l6(ob['eef'], a); g = grip if a == self.arm else float(st[7 * j + 6])
                row += list(M[:3, 3]) + CT.q_of(M) + [g]
            e16.append(row)
        j14 = self.eef_to_joint(st, e16)
        return phase, j14, np.asarray(e16), dict(target=int(tid), pairing=int(pi), residual=res, checks=ck, reachable=bool(not best[0][0]))

    def diag(self):
        P, Tc = self.truth_world(); ck = self.checks()
        tips = (P['tip_l'] + P['tip_r']) / 2; hs = [(P['o%d_hole_a' % k] + P['o%d_hole_b' % k]) / 2 for k in range(3)]; h = min(hs, key=lambda x: np.linalg.norm(x[:2] - tips[:2]))
        return dict(tip_to_hole_xy_mm=round(float(np.linalg.norm(tips[:2] - h[:2])) * 1000, 1), tip_above_entry_mm=round(float(tips[2] - h[2]) * 1000, 1), checks=ck)


def run_bc(a, rec, co, tool, ad, rpc, objs, arm, rm, tru, SOCK, end):
    out = Path(a.out); log = dict(policy=OPT['policy'], ckpt=OPT['ckpt'], budget=OPT['budget'], exec_n=OPT['exec_n'], hold_arm=arm); rec['bc'] = log
    ro = Rollout(rec, co, tool, ad, rpc, objs, arm, out); chunks = log.setdefault('chunks', [])
    try:
        pol = _ld('bc_policy', OPT['policy']).load(OPT['ckpt'])
        while True:
            ob, paths = ro.obs(); used = ob['step'] - ro.s0
            ph, j14, e16, info = ro.expert(ob)
            ro.lab.write(json.dumps(dict(episode=rec['start']['episode'], layout=rec['start']['layout'], step=ob['step'], step_after_handover=used,
                                         frames=paths, state=ob['state'].tolist(), eef=ob['eef'].tolist(), instruction=ob['instruction'], hold_arm=arm,
                                         expert_phase=ph, expert_joint=None if j14 is None else j14.round(6).tolist(),
                                         expert_eef=None if e16 is None else e16.round(6).tolist(), expert_info=info, source='bc_rollout_dagger')) + '\n'); ro.lab.flush()
            dg = ro.diag(); row = dict(step=ob['step'], used=used, expert_phase=ph, **dg); chunks.append(row)
            if dg['checks'] and min(dg['checks']) >= 1.0: log['outcome'] = 'inserted'; break
            if ph == 'charger_not_held' and not OPT['start']: log['outcome'] = 'charger_dropped'; break   # --start 1 (public start states): no grasp expert, keep rolling
            if used >= OPT['budget']: log['outcome'] = 'budget'; break
            pr = pol.act(ob); acts = np.asarray(pr['actions'], float)
            jc = acts if pr.get('kind', 'joint') == 'joint' else ro.eef_to_joint(ob['state'], acts)
            if OPT['gripgate']:   # DIAGNOSTIC harness gate (10-31): the holding gripper may not open more than at the handover (until inserted = loop end)
                gi = 6 if arm == 'left' else 13; g0 = log.setdefault('grip_at_handover', float(ob['state'][gi])); jc = np.array(jc, float)
                n_over = int((jc[:, gi] > g0 + 1e-3).sum()); jc[:, gi] = np.minimum(jc[:, gi], g0); log['gate_overrides'] = log.get('gate_overrides', 0) + n_over
            r = ro.execute(jc[:max(1, min(OPT['exec_n'], len(jc)))]); row['policy_kind'] = pr.get('kind', 'joint')
            if r.get('ended'): log['outcome'] = 'native_end_success' if r.get('success') else 'native_end'; break
    except Exception:  # noqa: BLE001
        import traceback; log['outcome'] = 'error'; log['error'] = traceback.format_exc()[-1500:]
    finally:
        log['steps'] = (chunks[-1]['used'] if chunks else 0); log['first_failure_signal'] = next((c['step'] for c in chunks if c['expert_phase'] == 'charger_not_held'), None)
        log['native_success'] = bool(log.get('outcome') in ('inserted', 'native_end_success') and (rec.get('restore') or {}).get('step', 0) + log['steps'] <= 400)
        ro.lab.close(); rec['bc'] = log


if __name__ == '__main__':   # bc_start.py imports this module for the public start states
    LC.run = run_bc
    print('bc_rollout: BC policy rollout + DAgger expert labels; opts', OPT, flush=True)
    script = argv[0]; sys.argv = [script] + argv[1:]; sys.path.insert(0, str(Path(script).resolve().parent))
    runpy.run_path(script, run_name='__main__')
