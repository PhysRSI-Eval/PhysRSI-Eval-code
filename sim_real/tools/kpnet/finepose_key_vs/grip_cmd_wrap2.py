"""INFRA ERRATUM wrapper (capability line, 2026-10-27 18:1xZ): gripper command persistence for stage_tools._Tool, without editing frozen code.
Bug (TAKEOVER-BUG-AUDIT.md): _Tool.line / act(..., tool.opening(arm)) re-send the MEASURED finger joint (current_state[arm*7+6]) as
gripper_opening; the offline simulator honours the continuous value (replay_server_ext.action_with_opening), so a held object loses its
squeeze (Astra's 0.0, or set_gripper's object width - 4 mm) on the first move.
Fix: per arm, remember the last EXPLICIT gripper command and return it from _Tool.opening():
  - initial value = Astra's executed gripper command at step --t of --episode (when the wrapped script has those args), else measured;
  - updated whenever act() is called with an opening that differs from the measured value (set_gripper, explicit open / close);
  - re-sends of tool.opening(arm) therefore keep the commanded squeeze; an explicit release (opening 1.0) is remembered as open.
Usage (same args as the wrapped script):  python grip_cmd_wrap.py src/<script>.py <script args...>"""
import runpy, sys
from pathlib import Path
# v2 (18:4xZ): package-aware. The wrapped script's own rsi package is preloaded (macro-tools pkg if the script inserts it, else
# regrasp env/pkg; their combined_tools.py differ), so the patched ST is the very module object the script uses.
_src = Path(sys.argv[1]).read_text()
_pkg = '/lizekai/RSIws/experiments/macro-tools-20261027/pkg' if 'macro-tools-20261027/pkg' in _src else '/lizekai/RSIws/experiments/regrasp-20261027/env/pkg'
sys.path.insert(0, '/lizekai/RSIws/experiments/regrasp-20261027/src'); sys.path.insert(0, _pkg)
from rsi import stage_tools as ST  # noqa: E402

CMD = {}
_meas = ST._Tool.opening
_act = ST._Tool.act


def opening(self, arm):
    return CMD[arm] if arm in CMD else _meas(self, arm)


def act(self, arm, pos, quat, opening_, reason, steps=5):
    if abs(float(opening_) - _meas(self, arm)) > 1e-6: CMD[arm] = float(opening_)
    for a in ST.ARMS:  # the other arm is re-sent by act() itself from self.opening(): seed it so it keeps its command too
        if a != arm and a not in CMD: CMD[a] = _meas(self, a)
    return _act(self, arm, pos, quat, opening_, reason, steps)


ST._Tool.opening = opening; ST._Tool.act = act
script = sys.argv[1]; args = sys.argv[2:]
if '--episode' in args and '--t' in args:
    import rg_sim as _rs  # noqa: E402
    _task, _lay, _acts = _rs.RA.recorded(args[args.index('--episode') + 1]); _c = _acts[int(args[args.index('--t') + 1]) - 1]['action']
    CMD.update(left=float(_c[6]), right=float(_c[13]))
print('grip_cmd_wrap2: pkg', _pkg, 'initial commands', dict(CMD), flush=True)
sys.argv = [script] + args; sys.path.insert(0, str(Path(script).resolve().parent))
runpy.run_path(script, run_name='__main__')
