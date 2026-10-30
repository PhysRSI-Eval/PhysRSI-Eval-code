"""L4 validator smoke on synthetic packages (make_kong stored episode frames as evidence). python3 smoke/test_experience.py"""
import copy, glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rsi_v2 import experience as L4
TOOLS = {'locate_mm', 'inhand_pose', 'check_reach', 'predict_outcome', 'grasp_compatible'}
od = sorted(glob.glob('/lizekai/RSIws/artifacts/native/ctx-6ea789960e3149f9-2-5f76f5f1/*/observations'))[0]
frames = {int(Path(p).stem) for p in glob.glob(od + '/*.npz')}
good = dict(schema='rsi-exp-v1', task='make_kong', episode='ctx-6ea789960e3149f9-2-5f76f5f1', outcome=dict(native_success=True, steps=540),
            action_ledger=[dict(purpose='take the replacement tile off the wall', contact_mode='top pinch across the tile width', step_span=[480, 520],
                                method='called locate_mm(mahjong_tile) with arms home, then grasp_compatible info, then my own pinch',
                                tool_calls=[dict(tool='locate_mm', inputs=dict(target='mahjong_tile'), returned=dict(tile_xyz='tile_xyz'), used_next='yes')],
                                outcome='tile lifted', observed_effect='tile rose with the hand', failure_mechanism=None, next_adjustment=None,
                                reusable_lesson='open the gripper only to the tile width plus about tile_margin before descending; the neighbours block a fully open hand',
                                evidence_steps=[480, 500, 520])],
            keyframes=[dict(step=500, camera='cam_high', why='pinch closed on the tile')],
            binding=dict(episode_local_values=[dict(name='tile_xyz', value=[-0.3997, -0.0501, 0.8309], unit='m', recompute=dict(method='tool', tool='locate_mm', args=dict(target='mahjong_tile'))),
                                               dict(name='tile_margin', value=0.002, unit='m', recompute=dict(method='cad', args=dict(object='mahjong9'))),
                                               dict(name='gripper_max_opening', value=0.088, unit='m', recompute=dict(method='robot', args=dict(constant='gripper_max_opening')))],
                         general_facts=['A fully open gripper (gripper_max_opening) lands on the neighbouring wall tiles.']),
            program_index=[], summary='Drew the replacement tile with a narrow top pinch.')
bad = copy.deepcopy(good)
bad['action_ledger'][0]['reusable_lesson'] = 'move the hand to x=-0.3997, y=-0.0501 and close'      # hard-coded coordinates, no binding
bad['action_ledger'][0]['tool_calls'][0]['tool'] = 'magic_locator'; bad['action_ledger'][0]['step_span'] = [480, 900]; bad['keyframes'][0]['step'] = 9999
def repair(pkg, errors):   # stands in for Astra's one repair turn: rewrite the lesson with a binding, fix the tool name / span / keyframe
    pkg['action_ledger'][0]['reusable_lesson'] = 'move the hand above tile_xyz and close'; pkg['action_ledger'][0]['tool_calls'][0]['tool'] = 'locate_mm'
    pkg['action_ledger'][0]['step_span'] = [480, 520]; pkg['keyframes'][0]['step'] = 500; return pkg
res = dict(valid=L4.finalize(good, TOOLS, None, frames), bad_repaired=L4.finalize(bad, TOOLS, repair, frames), bad_no_repair=L4.finalize(bad, TOOLS, None, frames))
out = {k: dict(status=v['status'], errors_first=v['errors_first'], errors_final=v['errors_final']) for k, v in res.items()}
print(json.dumps(out, indent=1))
assert out['valid']['status'] == 'valid' and out['bad_repaired']['status'] == 'valid' and out['bad_no_repair']['status'] == 'invalid_no_repair'
assert len(out['bad_no_repair']['errors_first']) >= 4
print('PASS'); Path(__file__).resolve().parent.parent.joinpath('out', 'SMOKE-L4.json').write_text(json.dumps(out, indent=1))
