"""L2 decision-point status block (harness v2 RC) = the v1.7b / Barrel-3 monitor_hub generalised: the hub knows no task rules; any monitor
that passed rung 1 registers as a plugin {name, fn(ctx) -> {answer: yes|no|unsure, p, text}, evidence}. Configuration per episode:
  {"monitors": [names], "schedule": [steps], "min_path_rad": 0.5}
Delivery (as v1.7b, which delivered 4/4): at the first decision point at/after each scheduled control step, after >= min_path_rad of arm
motion; facts only: each monitor's current camera reading as is ('confirmed'), the gripper closures Astra itself issued ('issued'), steps
used/left. The harness never calls an issued command 'done'. 0 control steps. Never raises into the episode.
Plugins shipped: kong_p2 (make_kong replacement tile taken; W8 P2, new-data answered accuracy 0.99). Monitors that failed rung 1 (P1, P3)
are not registered."""
import importlib.util
from pathlib import Path
import numpy as np

HEAD = '=== STATUS (automatic, from the harness; RGB-only estimates, privileged=false) ==='
TAIL = '=== END OF STATUS ==='
ARM = np.r_[0:6, 7:13]
PLUGINS = {}


def register(name, fn, evidence, tasks):
    PLUGINS[name] = dict(fn=fn, evidence=evidence, tasks=list(tasks))


def _kong_p2():
    sp = importlib.util.spec_from_file_location('mh17b', '/lizekai/RSIws/experiments/online-mk-v17b-20261028/pkg/rsi/monitor_hub.py')
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    def fn(ctx):
        return m.kong_p2(dict(images=[dict(camera='cam_high', path=str(ctx.images['cam_high']))]))
    return fn


register('kong_p2', _kong_p2(), 'W8 P2 (task_progress): new-data coverage 0.83, answered accuracy 0.99, false yes 0 (this check was not pre-registered); online v1.7b delivered 4/4', ['make_kong'])


class Hub:
    def __init__(self, cfg):
        self.names = [n for n in (cfg.get('monitors') or []) if n in PLUGINS]
        self.pending = sorted(int(s) for s in (cfg.get('schedule') or (150, 250, 350, 450))); self.min_path = float(cfg.get('min_path_rad', 0.5))
        self.path = 0.0; self.last_q = None; self.closed = {'left': False, 'right': False}; self.closures = {'left': 0, 'right': 0}; self.delivered = []

    def note_call(self, args):
        for arm in ('left', 'right'):
            c = (((args or {}).get('target') or {}).get(arm) or {}).get('gripper_closed')
            if c is None: continue
            if c and not self.closed[arm]: self.closures[arm] += 1
            self.closed[arm] = bool(c)

    def after(self, ctx, args):
        try:
            self.note_call(args); q = np.asarray(ctx.states, float)
            if self.last_q is not None: self.path += float(np.abs(q[ARM] - self.last_q[ARM]).max())
            self.last_q = q; due = [s for s in self.pending if ctx.step >= s]
            if not due or self.path < self.min_path or not self.names: return None
            self.pending = [s for s in self.pending if s > ctx.step]; mx = int(ctx.max_steps or 600); lines = [HEAD]; conf = {}
            for n in self.names:
                r = PLUGINS[n]['fn'](ctx); conf[n] = dict(answer=r['answer'], p=r.get('p'))
                lines.append('Confirmed from the head camera at step %d: %s.' % (ctx.step, r['text']))
            lines.append('Issued by you so far: %d gripper closures (left %d, right %d).' % (sum(self.closures.values()), self.closures['left'], self.closures['right']))
            lines.append('Control steps used: %d of %d (%d left).' % (ctx.step, mx, max(mx - ctx.step, 0))); lines.append(TAIL)
            rec = dict(step=ctx.step, confirmed=conf, issued=dict(self.closures), text='\n'.join(lines)); self.delivered.append(rec); return rec
        except Exception as e:  # noqa: BLE001
            self.delivered.append(dict(step=ctx.step, error='%s: %s' % (type(e).__name__, str(e)[:160]))); return None
