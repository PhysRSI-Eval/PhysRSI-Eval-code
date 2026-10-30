"""Glue for the harness v2 RC on top of the original RSI harness (skill library empty, Astra iterates by itself over K rounds; this module only
adds tools, facts and the experience package). Per Astra turn:
  tool call  -> L3 allow_call (executor guard) -> L1 registry.call -> reply (uniform contract, English)
  after act  -> L2 hub status block (scheduled) + L3 fine-stage availability fact (once) -> extra inputText items (facts only)
  episode end (native success) -> L4 experience.finalize (validator + one repair)
Token accounting: added unprompted text per turn, estimated as ceil(chars / 4) (no tokenizer installed on these hosts; stated as an estimate)."""
import json, math
from . import registry as L1, monitors as L2, surfacing as L3, experience as L4, policy as L3P


def tok(text): return int(math.ceil(len(text) / 4.0))


class Harness:
    def __init__(self, cfg=None):
        self.cfg = cfg or {}; self.hub = L2.Hub(self.cfg.get('monitor_hub') or {}); self.gate = L3.Gate(self.cfg.get('min_gap_steps', 25))
        self.turns = []; self.prev_grip = None; self.z_close = {}
        # L3 learned policy (policy.py): cfg 'l3_policy' = {name, mode: log|offer|auto, executor, auto_approved}; absent -> no policy
        pc = self.cfg.get('l3_policy') or {}; self.pol = L3P.load(pc.get('name', 'gate_v2')) if pc else None; self.pol_cfg = pc
        self.pol_mode = pc.get('mode', 'log') if pc.get('mode') != 'auto' or pc.get('auto_approved') is True else 'offer'
        if self.pol is not None and getattr(self.pol, 'a', {}).get('dedup_refit_required'): self.pol_mode = 'log'   # duplicate-state training rows: log only until refit
        if self.pol is not None and self.pol_mode not in getattr(self.pol, 'modes_allowed', ('log', 'offer', 'auto')): self.pol_mode = 'offer'   # router v0: log/offer only
        self.pol_log = [] if not (pc.get('mode') == 'auto' and self.pol_mode == 'offer') else [dict(event='auto downgraded to offer (auto_approved is not true)')]
        self.S, self.A = [], []; self.offered_exec = False; self.auto_done = False

    def tool_list(self):
        return L1.specs()

    def on_call(self, name, args, ctx, explicit=True):
        ok, why = L3.allow_call(name, args, explicit)
        if not ok: return L1.out(note=why)
        return L1.call(name, ctx, **(args or {}))

    def _fine_stage(self, ctx):
        """Deploy signal: an arm's gripper closed on something (measured 0.05..0.9) and the TCP rose >= 3 cm since the close."""
        for i, arm in enumerate(('left', 'right')):
            g = ctx.grip(arm); z = float(ctx.eef[arm][0][2])
            if 0.05 < g < 0.9:
                self.z_close.setdefault(arm, z)
                if z - self.z_close[arm] >= 0.03: return True
            else:
                self.z_close.pop(arm, None)
        return False

    def _policy(self, ctx, args):
        """Run the L3 policy once per turn (deploy inputs only). Returns extra unprompted lines (offer mode) or [] ; auto mode calls the executor."""
        if self.pol is None: return []
        self.S.append(list(ctx.states)); a = [1.0] * 14
        for i, arm in enumerate(('left', 'right')):
            c = (((args or {}).get('target') or {}).get(arm) or {}).get('gripper_closed')
            prev = self.A[-1][6 + 7 * i] if self.A else 1.0
            a[6 + 7 * i] = prev if c is None else (0.0 if c else 1.0)
        self.A.append(a)
        # gate v2 was trained on PER-STEP archived states / executed actions; online, cfg 'step_history' (callable -> (S, A)) must give that from
        # the native action log. The per-turn fallback (self.S / self.A) under-counts travel and closes within a turn and is for offline smoke only.
        sh = self.cfg.get('step_history'); S_, A_ = sh() if callable(sh) else (self.S, self.A)
        obs = dict(task=self.cfg.get('task'), step=ctx.step, step_limit=int(self.cfg.get('native_step_limit') or ctx.max_steps or 400), states_hist=S_, cmd_hist=A_,
                   executor=self.pol_cfg.get('executor'), images=ctx.images, tool_facts=self.cfg.get('tool_facts') or {})
        d = self.pol.decide(obs); self.pol_log.append(dict(step=ctx.step, **{k: d[k] for k in ('decision', 'p', 'executor')}))
        if d['decision'] != 'intervene' or self.pol_mode == 'log': return []
        if self.pol_mode == 'offer':
            if self.offered_exec: return []
            self.offered_exec = True
            return self.gate.unprompted(ctx.step, [L3.availability_fact(d['executor'], 'the result of an automatic put-down, re-grasp, transport and insertion of the held object, or an abstain with no motion')])
        if self.pol_mode == 'auto' and not self.auto_done:
            self.auto_done = True; r = self.on_call(d['executor'], {}, ctx, explicit=L3.POLICY_TOKEN); self.pol_log[-1]['executor_result'] = dict(abstain=r['abstain'], steps_used=r['steps_used'])
        return []

    def after_act(self, ctx, args=None):
        items = []; rec = self.hub.after(ctx, args)
        if rec: items.append(rec['text'])
        items += self._policy(ctx, args)
        items += self.gate.on_fine_stage(ctx.step, self._fine_stage(ctx))
        text = '\n'.join(items); self.turns.append(dict(step=ctx.step, added_tokens=tok(text) if text else 0, n_items=len(items)))
        return [dict(type='inputText', text=t) for t in items]

    def experience(self, pkg, repair=None, frames_available=None):
        return L4.finalize(pkg, set(L1.REGISTRY), repair, frames_available)
