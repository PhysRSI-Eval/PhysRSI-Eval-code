"""L1 tool registry (harness v2 RC). Every tool returns the uniform contract {value, uncertainty, abstain, steps_used, note}; one English
SKILL card per tool under ../skills/<name>/SKILL.md. Only tools that passed their pre-registered rung are registered; each entry carries its
evidence and scope. Deploy inputs only (RGB + camera calibration, proprioception, action history, object class name + CAD).

Context (Ctx): built per Astra turn from the native packet (or a stored replay frame): images (camera -> path or HxWx3), states (14),
eef {arm: (pos, quat_wxyz)}, head intrinsics/extrinsics (camera_meta), last commanded gripper openings (action history), step and
step limit; optional callbacks move_and_capture / return_observer (only for wrist-observer targets; they move an arm and are therefore
used only when Astra calls the tool with a step budget).
"""
import importlib.util, json, time, urllib.request, base64
from pathlib import Path
import numpy as np
from .contract import out, check

E = Path('/lizekai/RSIws/experiments'); HERE = Path(__file__).resolve().parent; SKILLS = HERE.parent / 'skills'
MT = E / 'macro-tools-20261027'
K_HEAD = [[288.13254096884566, 0.0, 320.0], [0.0, 288.13254096884566, 240.0], [0.0, 0.0, 1.0]]
T_HEAD = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.8660253435950835, -0.5000001052916332, -0.4099999964237213],
          [0.0, 0.5000001052916332, 0.8660253435950835, 1.3079999685287476], [0.0, 0.0, 0.0, 1.0]]   # static head camera (camera_meta)
_MOD = {}


def _load(name, path):
    if name not in _MOD:
        sp = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); _MOD[name] = m
    return _MOD[name]


def _post(url, obj, timeout=60):
    r = urllib.request.Request(url, data=json.dumps(obj).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())


def _b64img(x):
    if isinstance(x, (str, Path)): return base64.b64encode(Path(x).read_bytes()).decode()
    import io
    from PIL import Image
    b = io.BytesIO(); Image.fromarray(np.asarray(x, np.uint8)).save(b, format='PNG'); return base64.b64encode(b.getvalue()).decode()


class Ctx:
    def __init__(self, images, states, eef, step=0, max_steps=None, last_grip_cmd=None, head_K=None, head_T=None, cfg=None,
                 move_and_capture=None, return_observer=None):
        self.images, self.states, self.eef, self.step, self.max_steps = images, list(map(float, states)), eef, int(step), max_steps
        self.last_grip_cmd = last_grip_cmd or {}; self.head_K = head_K or K_HEAD; self.head_T = head_T or T_HEAD; self.cfg = cfg or {}
        self.move_and_capture, self.return_observer = move_and_capture, return_observer

    def joints(self, arm): i = ('left', 'right').index(arm); return np.asarray(self.states[7 * i:7 * i + 6], float)
    def grip(self, arm): return float(self.states[7 * ('left', 'right').index(arm) + 6])
    def head(self): return dict(rgb=self.images.get('cam_high'), intrinsics=self.head_K, extrinsics_cam_to_world=self.head_T)


# ------------------------------------------------------------------------------------------------ tools
LOC_TARGETS = ('outlets', 'keyhole', 'toolbox_slots', 'tube_holes', 'mahjong_tile', 'xylophone_keys')   # keyhole_head FAILED rung 1: not registered
# Layout audit 2026-10-28 (experiments/layout-audit-20261028/CONCLUSION-layout-audit.md). A target is registered only with this list.
LAYOUTS_VERIFIED = {
    'outlets': dict(native_layouts=18, heldout_rung1=[0, 1, 7, 14, 15], astra_states_accurate=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 17],
                    astra_states_no_answer=[5, 11, 15, 16], note='LOC1 trained on 11 layouts; wrist answers accurate wherever given; no answer (abstain) in 4 layouts'),
    'xylophone_keys': dict(native_layouts=18, heldout_rung1=[2, 9, 15], trained=[0, 1, 3, 4, 6, 7, 8, 10, 11, 13, 14, 16, 17], dev=[5, 12],
                           model='X-v2 all-layout (kpX_head_all.pt, PREREG-X-v2-alllayout PASS: held-out lateral median 0.84 mm, p95 3.73 mm, 100% within half-width)',
                           note='X-v1 (layout 0 only) is ~165 mm off in the other 17 layouts and is not used'),
    'keyhole': dict(native_layouts=1, verified=[0], perturbed_layouts='9 (native reset of generated layout files, no teleport): 8/9 answered, centre median 0.47 mm, '
                    'axis median 3.1 deg (rung-1 gates hold); degrades beyond the official yaw range: axis 15 deg at -35 deg, abstains at -48 deg'),
    'toolbox_slots': dict(native_layouts=1, verified=[0], perturbed_layouts='9: 9/9 within gate, slot max 2.5 mm (extrapolated up to 8 cm / 29 deg)'),
    'tube_holes': dict(native_layouts=1, verified=[0], perturbed_layouts='8 valid (1 generated layout physically unstable): 8/8, rack-point max 1.85 mm (up to 15 cm / 27 deg)'),
    'mahjong_tile': dict(native_layouts=1, verified=[0], perturbed_layouts='9: 9/9 within 8 mm / 10 deg, max 4.6 mm / 1.6 deg (wall moved up to 6 cm / 30 deg); arms homed'),
}
X_V2 = Path('/lizekai/RSIws/experiments/layout-audit-20261028/models/kpX_head_all.pt')


def _coarse_cb(ctx):
    url = ctx.cfg.get('locate_url')
    if not url: return None
    def cb(prompt):
        im = {c: _b64img(p) for c, p in ctx.images.items()}   # the SAM3 service needs all three cameras (fixed 10-29 00:0xZ)
        req = dict(images_b64=im, states=ctx.states, eef_positions=[list(ctx.eef[a][0]) for a in ('left', 'right')],
                   eef_quaternions_wxyz=[list(ctx.eef[a][1]) for a in ('left', 'right')], object=prompt, camera='cam_head')
        try: return _post(url, req, timeout=120)
        except Exception as e: return dict(found=False, error=type(e).__name__)  # noqa: BLE001
    return cb


def t_locate_mm(ctx, target, budget_steps=0, arms_homed=None, **kw):
    if target not in LOC_TARGETS and target not in ('outlet', 'power_strip', 'mahjong9', 'tile', 'tubes', 'toolbox', 'xylophone'):
        return out(note='target %r is not available (rung-1 passed targets: %s)' % (target, ', '.join(LOC_TARGETS)))
    L = _load('locate_v2', MT / 'pkg-v2/rsi/locate.py'); L.X_HEAD_CKPT = X_V2; kw.pop('extra_views', None)   # X-v2 head only (the wrist X model is layout-0 only)
    return L.locate_mm(target, obs=dict(head=ctx.head()), budget_steps=budget_steps, coarse=_coarse_cb(ctx), move_and_capture=ctx.move_and_capture,
                       return_observer=ctx.return_observer, arms_homed=arms_homed, device=ctx.cfg.get('device', 'cpu'), **kw)


def t_inhand_pose(ctx, arm, category='charger'):
    if category != 'charger':
        return out(note='in-hand pose is validated for the charger only (v3); other objects: not available')
    url = ctx.cfg.get('infer_url')
    if not url: return out(note='in-hand pose service not configured')
    req = dict(images_b64={'cam_high': _b64img(ctx.images['cam_high']), 'cam_%s_wrist' % arm: _b64img(ctx.images['cam_%s_wrist' % arm])}, states=ctx.states,
               eef_positions=[list(ctx.eef[a][0]) for a in ('left', 'right')], eef_quaternions_wxyz=[list(ctx.eef[a][1]) for a in ('left', 'right')],
               arm=arm, category=category, gripper_cmd=0.0)
    try:
        r = _post(url.rstrip('/') + '/inhand', req, timeout=60)
    except Exception as e:  # noqa: BLE001
        return out(note='in-hand pose service not reachable (%s)' % type(e).__name__)
    if 'position_m' not in r: return out(note=str(r.get('error', 'no estimate'))[:160])
    spread = r.get('member_spread_deg'); tau = float(ctx.cfg.get('spread_tau_deg', 22.795))
    val = dict(position_in_tcp_frame_m=[round(float(x), 4) for x in r['position_m']], quaternion_in_tcp_frame_wxyz=[round(float(x), 4) for x in r['quaternion_wxyz']])
    unc = dict(member_spread_mm=r.get('member_spread_mm'), member_spread_deg=spread, flip_caveat='about 1 frame in 8 can be a whole 180 deg turn of the charger')
    if isinstance(spread, (int, float)) and spread > tau:
        return out(value=dict(position_in_tcp_frame_m=val['position_in_tcp_frame_m']), uncertainty=unc, abstain=True,
                   note='estimator members disagree about the orientation (spread %.0f deg > %.0f): orientation not reported' % (spread, tau))
    return out(value=val, uncertainty=unc, abstain=False, note='in-hand pose v3 (charger), 0 control steps')


_IV4 = {}
def t_inhand_pose_v4(ctx, arm, category='charger'):
    """in-hand pose v4 (charger only; post-training line 10-29): service (cfg infer_url_v4, /inhand) or in-process (cfg inhand_v4_inprocess)."""
    if category != 'charger':
        return out(note='in-hand pose is validated for the charger only (v4); other objects: not available')
    i = ('left', 'right').index(arm); e = ctx.eef[arm]
    try:
        if ctx.cfg.get('infer_url_v4'):
            req = dict(images_b64={'cam_high': _b64img(ctx.images['cam_high']), 'cam_%s_wrist' % arm: _b64img(ctx.images['cam_%s_wrist' % arm])}, states=ctx.states,
                       eef_positions=[list(ctx.eef[a][0]) for a in ('left', 'right')], eef_quaternions_wxyz=[list(ctx.eef[a][1]) for a in ('left', 'right')], arm=arm, category=category)
            r = _post(ctx.cfg['infer_url_v4'].rstrip('/') + '/inhand', req, timeout=60); o = dict(value=r.get('value'), uncertainty=r.get('uncertainty'), abstain=r.get('abstain', True),
                                                                                                  flip_flag=r.get('flip_flag'), note=r.get('note', ''))
        elif ctx.cfg.get('inhand_v4_inprocess'):
            if 'm' not in _IV4:
                import importlib.util as _u
                sp = _u.spec_from_file_location('infer_v4', '/lizekai/RSIws/experiments/inhand-pose-20261026/v4/tool/infer_v4.py'); m = _u.module_from_spec(sp); sp.loader.exec_module(m)
                _IV4['m'] = m; _IV4['e'] = m.load(ctx.cfg.get('device', 'cuda'))
            o = _IV4['m'].infer(_IV4['e'], wrist=ctx.images['cam_%s_wrist' % arm], high=ctx.images['cam_high'], link6_pos=list(e[0]), link6_quat_wxyz=list(e[1]),
                                gripper_state=ctx.states[7 * i + 6], arm=arm, category=category)
        else:
            return out(note='in-hand pose v4 not configured')
    except Exception as ex:  # noqa: BLE001
        return out(note='in-hand pose v4 not reachable (%s)' % type(ex).__name__)
    unc = dict(o.get('uncertainty') or {}); unc['flip_flag'] = o.get('flip_flag')
    return out(value=o.get('value'), uncertainty=unc, abstain=bool(o.get('abstain')), note=o.get('note', ''))


_CR = {}
def t_check_reach(ctx, arm, target_pose, boxes=()):
    RC = _load('reach_cu_v2', MT / 'pkg-v2/rsi/reach_cu.py'); w = round(float(np.interp(ctx.grip(arm), [0.18, 0.25, 0.5, 0.75, 1.0], [0.0, 0.007, 0.034, 0.061, 0.088])), 3)
    cr = _CR.get(w) or _CR.setdefault(w, RC.CollisionReach(width=w))
    return RC.check_reach(cr, arm, ctx.joints(arm), target_pose, boxes=boxes)


def t_align_target(ctx, strip, inhand, hover_m=0.015):
    AT = _load('align_target', MT / 'pkg-v2/rsi/align_target.py'); e = ctx.eef
    arm = inhand.get('arm', 'left'); p, q = e[arm]; from scipy.spatial.transform import Rotation as _R
    L = np.eye(4); L[:3, :3] = _R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); L[:3, 3] = p
    return AT.align_target(strip, inhand, L, hover_m=hover_m)


_PO = {}
def t_predict_outcome(ctx, arm, link6_traj, gripper_cmds, object_class):
    PO = _load('predict_outcome_v3', E / 'outcome-w7-20261027/v3/tool/predict_outcome.py')
    m = _PO.get('m') or _PO.setdefault('m', PO.PredictOutcome())
    return m(arm, link6_traj, gripper_cmds, ctx.grip(arm), object_class)


def t_grasp_compatible(ctx, object_vertices, object_pose, arm='left', next_stage=None, support_top_z=None, mode='info'):
    if mode != 'info':
        return out(note='execute mode runs only on an explicit Astra call through the executor path (harness.call_executor); info mode returned nothing')
    GC = _load('grasp_compat2', MT / 'pkg-v2/rsi/grasp_compat2.py'); RM = _load('reach_model_v15', MT / 'pkg-v15/rsi/reach_model.py')
    rm = _MOD.get('_rm') or _MOD.setdefault('_rm', RM.Reach()); table_z = float(rm.root['left'][2, 3])
    r = GC.grasp_compatible(arm, ctx.joints(arm), ctx.grip(arm), object_vertices, np.asarray(object_pose, float), rm, None, support_top_z, table_z, next_stage,
                            last_command=ctx.last_grip_cmd.get(arm))
    r.pop('busy', None)
    if isinstance(r.get('value'), dict): r['value'].pop('_rows', None)
    r['note'] = (r.get('note') or '') + '. Scope: validated only for the make_kong replacement tile from idle-arm states (K-v2 pose 24/26 lifted); controller-only check in this call'
    return r


def _schema(props, req):
    return dict(type='object', properties=props, required=list(req))


REGISTRY = {
    'locate_mm': dict(fn=t_locate_mm, version='v2', mode='info (wrist targets move the free arm within budget_steps)',
                      evidence='rung 1 PASS (outlets 0.15 mm, keyhole 0.55, toolbox slots 0.21, tube holes 0.11, mahjong tile 0.19 mm; xylophone keys X-v2 0.84 mm lateral on held-out layouts); layouts verified per target in LAYOUTS_VERIFIED; '
                               'outlets on Astra holds: wrist layer 46/47 within 2 mm but rung-4 non-interference FAILED (an open gripper next to the strip dragged it) -> outlets listed with that scope note',
                      schema=_schema(dict(target=dict(type='string', enum=list(LOC_TARGETS)), budget_steps=dict(type='integer')), ['target'])),
    'inhand_pose': dict(fn=t_inhand_pose_v4, version='v4 (charger only; v3 kept as t_inhand_pose)', mode='info, 0 steps',
                        evidence='PREREG-IP-v4 charger PASS: held-out layouts 0.50 mm / 0.60 deg (coverage 97 %), Astra held-out layouts 0.36 / 0.45 (99 %); flip flag + abstain calibrated on DEV; parity tool/PARITY-v4-charger.json',
                        schema=_schema(dict(arm=dict(type='string', enum=['left', 'right']), category=dict(type='string')), ['arm'])),
    'check_reach': dict(fn=t_check_reach, version='v2', mode='info, 0 steps',
                        evidence='rung-4 style PREREG-reach-M3 PASS (unreachable recall 20/20, precision 20/22; clean subset 12/12, 12/14); fresh: "no" answers 26/26 correct with the controller-only term',
                        schema=_schema(dict(arm=dict(type='string'), target_pose=dict(type='object')), ['arm', 'target_pose'])),
    'predict_outcome': dict(fn=t_predict_outcome, version='v3 (model P)', mode='info, 0 steps, no images',
                            evidence='PREREG-W7-v3 PASS: slip balanced AUROC 0.79 on 1520 fresh samples (kinematic rule 0.50)',
                            schema=_schema(dict(arm=dict(type='string'), link6_traj=dict(type='array'), gripper_cmds=dict(type='array'), object_class=dict(type='string')),
                                           ['arm', 'link6_traj', 'gripper_cmds', 'object_class'])),
    'align_target': dict(fn=t_align_target, version='v1 (charger)', mode='info, 0 steps',
                         evidence='PREREG-align-target-v1 rung 1 PASS: on 48 answered Astra hold states (of 76 valid) the computed hover hand pose puts the charger within 2 mm / 2 deg of the true outlet axis in 44 (median 0.61 mm / 0.61 deg); rung 2 not informative (scripted execution 0/42 with both computed and truth targets)',
                         schema=_schema(dict(strip=dict(type='object'), inhand=dict(type='object'), hover_m=dict(type='number')), ['strip', 'inhand'])),
    'grasp_compatible': dict(fn=t_grasp_compatible, version='v1.1 (frozen tile scope)', mode='info, 0 steps; execute only on explicit call (step cap 150)',
                             evidence='make_kong tile: rung 2 truth 32/42, rung 3 K-v2 29/45 FAIL overall; idle-arm phase-1-end states 24/26 -> deployable in that scope, not general',
                             schema=_schema(dict(object=dict(type='string'), arm=dict(type='string'), mode=dict(type='string', enum=['info', 'execute'])), ['object'])),
}


def specs():
    """English tool list for the model (descriptions = first paragraph of each SKILL card + evidence + scope)."""
    S = []
    for n, e in REGISTRY.items():
        card = (SKILLS / n / 'SKILL.md'); desc = card.read_text().split('\n\n')[1].strip() if card.exists() else n
        S.append(dict(type='function', name=n, description='%s Mode: %s. Evidence: %s.' % (desc, e['mode'], e['evidence']), inputSchema=e['schema']))
    return S


CALLS = []


def call(name, ctx, **args):
    """Uniform call: unknown tool / exception -> abstain with an English note; every call is logged with steps_used and wall time."""
    t0 = time.time()
    if name not in REGISTRY:
        r = out(note='unknown tool %r; available: %s' % (name, ', '.join(REGISTRY)))
    else:
        try:
            r = REGISTRY[name]['fn'](ctx, **args)
        except Exception as e:  # noqa: BLE001
            r = out(note='internal error %s: %s' % (type(e).__name__, str(e)[:160]))
    v = check(r)
    if v: r = out(note='contract violation (%s); result withheld' % '; '.join(v))
    CALLS.append(dict(tool=name, step=ctx.step, steps_used=r['steps_used'], abstain=r['abstain'], wall_s=round(time.time() - t0, 3)))
    return r


# ------------------------------------------------------------------------------------------------ additions 10-29 12:xxZ (bottleneck line = harness-layer owner)
# Single integration point: every tool that passed a pre-registered rung is registered here with the uniform contract and an English SKILL card.
# 'listed': False keeps a tool registered (harness-internal use, e.g. router features) without showing it in Astra's tool list.

def _holding_arm(ctx):
    """Deploy signal (as gate v2 features): last gripper command closed and measured opening 0.02..0.9."""
    for arm in ('left', 'right'):
        c = ctx.last_grip_cmd.get(arm); g = ctx.grip(arm)
        if c is not None and float(c) < 0.5 and 0.02 < g < 0.9: return arm
    return None


def t_insert_charger(ctx, arm=None):
    """Executor (explicit call only; L3 EXECUTORS). Frozen rg3f deploy chain (macro-tools-20261027/rg3f, PREREG-rg3-fast PASS 24/60 within 400
    steps from Astra's own charger holds). First act = in-hand pose v4 at the lift, 0 steps: if it abstains, nothing moves (lift-abstain = no
    intervention). The chain itself runs through ctx.cfg['rg3f_backend'](ctx, arm, inhand) -> {inserted, steps_used, chain_end_step, phase_steps,
    failure}; a release without a deploy backend returns abstain with 0 steps."""
    arm = arm or _holding_arm(ctx)
    if arm not in ('left', 'right'):
        return out(note='no arm is holding an object now (gripper closed on something), so the charger insertion chain was not started; no motion was made')
    ih = t_inhand_pose_v4(ctx, arm)
    if ih['abstain']:
        return out(uncertainty=ih.get('uncertainty'), note='in-hand pose v4 abstained at this hold (%s); the insertion chain was not started; no motion was made' % (ih['note'][:120] or 'no estimate'))
    be = ctx.cfg.get('rg3f_backend')
    if be is None:
        return out(value=None, uncertainty=ih.get('uncertainty'), note='insertion executor backend is not configured in this release; no motion was made')
    r = be(ctx, arm, ih['value']) or {}
    v = dict(inserted=bool(r.get('inserted')), chain_end_step=r.get('chain_end_step'), phase_steps=r.get('phase_steps'), failure=r.get('failure'))
    return out(value=v, uncertainty=ih.get('uncertainty'), abstain=False, steps_used=int(r.get('steps_used') or 0),
               note='rg3f chain (put down, locate outlets, compatible re-grasp, transport, align, insert); %s' % ('all three insertion checks passed' if v['inserted'] else 'ended without insertion: %s' % v['failure']))


SSEG = Path('/lizekai/RSIws/experiments/smallseg-20261029')
# PASS groups only (CONCLUSION-SSEG-v1.md; results/JUDGE-SSEG-v2.json). key: v1 FAIL, v2 NOT JUDGEABLE -> not available.
SSEG_PASS = {('coin', 'head'): ('sseg-v2.pt', 'tiled', 'SSEG-v2 PASS: detection 0.803 (SAM3 0.017 in v1), median IoU 0.81, FP 0.0'),
             ('coin', 'wrist'): ('sseg-v2.pt', 'full', 'SSEG-v2 PASS: detection 0.892, median IoU 0.88, FP 0.002 (v1 also PASS 0.885; SAM3 0.215)'),
             ('mahjong', 'head'): ('sseg-v1.pt', 'full', 'SSEG-v1 PASS: detection 0.924, median IoU 0.94, FP 0.0'),
             ('mahjong', 'wrist'): ('sseg-v1.pt', 'full', 'SSEG-v1 PASS: detection 0.933, median IoU 0.98, FP 0.001')}
_SEG = {}


def _seg_net(ckpt, device):
    k = (ckpt, device)
    if k not in _SEG:
        import sys, torch
        sys.path.insert(0, str(SSEG / 'src')); T = _load('sseg_train_seg', SSEG / 'src/train_seg.py')
        net = T.Net(); ck = torch.load(SSEG / 'models' / ckpt, map_location='cpu'); net.load_state_dict(ck['state'], strict=False); net.eval().to(device); _SEG[k] = (T, net)
    return _SEG[k]


def t_segment_small(ctx, object, camera='cam_high', max_instances=30):
    """Fine segmentation of small / thin objects with the DINOv3 segmenter (frozen DINOv3-B/16 + FiLM head), for the validated groups only."""
    import torch, torch.nn.functional as F
    view = 'head' if camera == 'cam_high' else ('wrist' if camera in ('cam_left_wrist', 'cam_right_wrist') else None)
    g = SSEG_PASS.get((object, view))
    if g is None:
        return out(note='segment_small is validated for: %s; %s in %s is not available' % (', '.join('%s/%s' % k for k in SSEG_PASS), object, camera))
    ckpt, how, ev = g; dev = ctx.cfg.get('device', 'cpu'); T, net = _seg_net(ckpt, dev)
    from PIL import Image
    im = ctx.images.get(camera)
    img = Image.open(im).convert('RGB') if isinstance(im, (str, Path)) else Image.fromarray(np.asarray(im, np.uint8))
    x = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)[None].to(dev); H_, W_ = x.shape[-2:]; c = T.CLS.index(object)
    with torch.no_grad():
        if how == 'full':
            lg = F.interpolate(net(x), size=(H_, W_), mode='bilinear', align_corners=False)[0, c]
        else:   # SSEG-v2 head: 3x3 overlapping 320x240 tiles, each resized to 640x480, logits averaged back
            TW, TH = W_ // 2, H_ // 2; tiles = [(x0, y0) for y0 in (0, TH // 2, TH) for x0 in (0, TW // 2, TW)]
            xs = torch.cat([F.interpolate(x[:, :, y0:y0 + TH, x0:x0 + TW], size=(H_, W_), mode='bilinear', align_corners=False) for x0, y0 in tiles])
            lt = F.interpolate(net(xs), size=(TH, TW), mode='bilinear', align_corners=False)[:, c]
            acc = torch.zeros(H_, W_, device=dev); cnt = torch.zeros(H_, W_, device=dev)
            for k, (x0, y0) in enumerate(tiles): acc[y0:y0 + TH, x0:x0 + TW] += lt[k]; cnt[y0:y0 + TH, x0:x0 + TW] += 1
            lg = acc / cnt
    prob = torch.sigmoid(lg).float().cpu().numpy(); m = prob > 0.5
    from scipy import ndimage
    lab, n = ndimage.label(m); inst = []
    for i in range(1, n + 1):
        ys, xs_ = np.nonzero(lab == i)
        if len(ys) < 8: continue
        inst.append(dict(bbox_px=[int(xs_.min()), int(ys.min()), int(xs_.max()), int(ys.max())], centroid_px=[round(float(xs_.mean()), 1), round(float(ys.mean()), 1)],
                         area_px=int(len(ys)), mean_prob=round(float(prob[ys, xs_].mean()), 3)))
    inst = sorted(inst, key=lambda d: -d['area_px'])[:int(max_instances)]
    note = 'DINOv3 segmenter (%s, %s view). Pixel coordinates in the %s image (640x480).' % (ckpt.split('.')[0], view, camera)
    if object == 'mahjong': note += ' Adjacent tiles can merge into one region.'
    if not inst: return out(value=dict(n_instances=0, instances=[]), uncertainty=dict(evidence=ev), abstain=False, note=note + ' No instance found.')
    return out(value=dict(n_instances=len(inst), instances=inst), uncertainty=dict(evidence=ev, threshold=0.5), abstain=False, note=note)


# KPNet / KPT keypoint tools (post-training line factory). Contract for every keypoint tool registered through register_kp_tool:
#   value = {frame: 'world', keypoints: [{name, xyz_m | null, px: {camera: [u, v] | null}, visible: {camera: bool}}], object}
#   uncertainty = {reproj_px, triangulation_residual_mm, n_views}; abstain when fewer than two views see a keypoint or residual > its gate.
#   Registration requires: a pre-registered rung-1 PASS on held-out layouts (evidence string), the layouts verified, and an English SKILL card.
def register_kp_tool(name, fn, version, evidence, schema, mode='info, 0 steps unless budget_steps > 0 (moves the free arm to a second view)', listed=True):
    if not (SKILLS / name / 'SKILL.md').exists(): raise ValueError('SKILL card missing for %s' % name)
    REGISTRY[name] = dict(fn=fn, version=version, mode=mode, evidence=evidence, schema=schema, listed=listed)


REGISTRY['insert_charger'] = dict(fn=t_insert_charger, version='rg3f (frozen, macro-tools-20261027/rg3f)', mode='executor: runs only on an explicit call; 0 steps when in-hand pose v4 abstains',
                                  evidence='PREREG-rg3-fast PASS (offline counterfactual from Astra holds): inserted within the 400-step limit 24/60 (deploy estimates) = 24/60 (truth); on 52 states with Astra\'s own outcome the chain did better in 20, Astra in 1; needs the takeover by about step 95; plug_in_charger only',
                                  schema=_schema(dict(arm=dict(type='string', enum=['left', 'right'])), []))
REGISTRY['segment_small'] = dict(fn=t_segment_small, version='SSEG v1/v2 (DINOv3-B/16 + FiLM head; PASS groups only)', mode='info, 0 steps',
                                 evidence='PREREG-SSEG-v1/v2: coin head 0.80 / wrist 0.89, mahjong head 0.92 / wrist 0.93 detection on unseen Astra episodes (SAM3 coin 0.02 / 0.22); key not available',
                                 schema=_schema(dict(object=dict(type='string', enum=['coin', 'mahjong']), camera=dict(type='string', enum=['cam_high', 'cam_left_wrist', 'cam_right_wrist'])), ['object']))
REGISTRY['predict_outcome']['listed'] = False   # VALUE-CHECK-W7-v3: low value of information; registered for harness-internal use, not in Astra's list
REGISTRY['inhand_pose']['evidence'] += '; service :24820 (ip4-svc4) on main GPU1; a 180 deg flip flag and an abstain are part of the reply'
REGISTRY['locate_mm']['evidence'] += '; the hv2 release path for outlets (120-step cap, 0-step abstain) re-judged rung 4 PASS: 39/39 within 2 mm, strip moved 0 mm in 41/41 (r4loc), used online in hv2a 6/6'


def _specs_listed():
    S = []
    for n, e in REGISTRY.items():
        if e.get('listed', True) is False: continue
        card = (SKILLS / n / 'SKILL.md'); desc = card.read_text().split('\n\n')[1].strip() if card.exists() else n
        S.append(dict(type='function', name=n, description='%s Mode: %s. Evidence: %s.' % (desc, e['mode'], e['evidence']), inputSchema=e['schema']))
    return S


specs = _specs_listed   # noqa: F811  (tools with listed=False are callable by the harness but not shown to Astra)


# ------------------------------------------------------------------------------------------------ C3: check_held_small (10-29 14:xxZ)
HELD_PX = 25.0   # coin-deploy v2 --post_px: a held coin moves with the wrist camera; one left behind shifts ~25 px per cm of lift


def _coin_centroid(img, device):
    import torch, torch.nn.functional as F
    from PIL import Image
    T, net = _seg_net('sseg-v1.pt', device)   # the exact model and threshold of coin-deploy v2 (logit > 0, >= 20 px, coin channel)
    a = np.asarray(Image.open(img).convert('RGB') if isinstance(img, (str, Path)) else img, np.float32) / 255.0
    x = torch.from_numpy(a).permute(2, 0, 1)[None].to(device)
    with torch.no_grad(): m = (F.interpolate(net(x), size=(480, 640), mode='bilinear', align_corners=False)[0, T.CLS.index('coin')] > 0).cpu().numpy()
    ys, xs = np.nonzero(m)
    return None if len(xs) < 20 else [round(float(xs.mean()), 1), round(float(ys.mean()), 1), int(len(xs))]


def t_check_held_small(ctx, arm, object='coin', close_step=None):
    """Fact: is the small object still in the gripper after a lift? SSEG coin mask in this arm's wrist view at the close vs now (coin-deploy v2
    post-lift check: agreed with the truth lift flag 29/29, dev 20/20). Needs ctx.cfg['step_image'](step, camera) -> archived frame path and
    ctx.cfg['step_history']() -> per-step (states, executed actions) to find the last close of this arm when close_step is not given."""
    if object != 'coin': return out(note='check_held_small is validated for the coin only (mahjong and key: not available)')
    si, sh = ctx.cfg.get('step_image'), ctx.cfg.get('step_history')
    if close_step is None and callable(sh):
        S, A = sh(); i = 6 + 7 * ('left', 'right').index(arm)
        cl = [k for k in range(1, len(A)) if A[k][i] < 0.5 <= A[k - 1][i]]
        if cl:   # first step after the close where the measured finger opening settles (|change| < 0.005), within 10 steps
            k0 = cl[-1]; close_step = next((k for k in range(k0 + 1, min(k0 + 11, len(S))) if abs(S[k][i] - S[k - 1][i]) < 0.005), min(k0 + 5, len(S) - 1))
    if close_step is None or not callable(si): return out(note='no close of this arm found in the step history (or no frame source configured); nothing checked')
    cam = 'cam_%s_wrist' % arm; dev = ctx.cfg.get('device', 'cpu')
    ref = next((si(k, cam) for k in range(int(close_step), int(close_step) + 11) if Path(si(k, cam)).exists()), None)   # frames are archived at decision points only
    if ref is None: return out(note='no archived wrist frame within 10 steps after the close; nothing checked')
    m0 = _coin_centroid(ref, dev); m1 = _coin_centroid(ctx.images[cam], dev)
    if m0 is None:
        return out(value=dict(held='unsure', close_step=int(close_step), coin_at_close=None, coin_now=m1), abstain=False,
                   note='the coin was not segmented in the wrist view at the close, so held cannot be judged')
    shift = None if m1 is None else round(float(np.hypot(m1[0] - m0[0], m1[1] - m0[1])), 1)
    held = 'yes' if (shift is not None and shift <= HELD_PX) else 'no'
    return out(value=dict(held=held, shift_px=shift, close_step=int(close_step), coin_at_close=m0, coin_now=m1), uncertainty=dict(threshold_px=HELD_PX), abstain=False,
               note='SSEG coin mask in the %s wrist view: %s' % (arm, 'moved with the camera (held)' if held == 'yes' else ('not found now' if m1 is None else 'shifted %.0f px (left behind)' % shift)))


REGISTRY['check_held_small'] = dict(fn=t_check_held_small, version='SSEG-v1 coin post-lift check (coin-deploy v2)', mode='info, 0 steps',
                                    evidence='coin-deploy v2 (pre-registered batch, deploy inputs): after the lift, held yes/no agreed with the truth lift flag in 29/29 rows (21 yes, 8 no); dev 20/20; coin only; validated after a lift of at least 3 cm',
                                    schema=_schema(dict(arm=dict(type='string', enum=['left', 'right']), object=dict(type='string', enum=['coin']), close_step=dict(type='integer')), ['arm']))
# 14:3xZ dispatcher: not used anywhere until the exact deploy path (reference frame rule) is re-checked against truth on the coin v2 rows.
# coin v2 deleted its per-job frames (coin_deploy.py rmtree(obsdir)), so the re-check needs a re-run with frame capture -> hidden until then.
REGISTRY['check_held_small']['listed'] = False
