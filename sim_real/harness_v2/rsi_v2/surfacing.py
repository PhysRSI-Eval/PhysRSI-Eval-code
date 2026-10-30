"""L3 surfacing rules (harness v2 RC). What the harness may put in front of Astra without being asked, and how.

R1 Tools are always listed (L1 specs); a tool's RESULT reaches Astra only as the reply to Astra's own call.
R2 Unprompted text = facts only: L2 status blocks and at most one one-line availability fact per trigger ("tool X is available; it returns
   Y"), never an instruction. Every unprompted line passes the advice filter below; a line that fails is dropped and logged.
R3 Executor modes (grasp_compatible execute, align_and_insert, regrasp) never run automatically; they run only on an explicit Astra call with
   a step cap. Evidence that taking over / steering hurts: plug M3 deploy takeover 3/52 vs Astra's own continuation 24/48 on the same states;
   coin c3 (rsiws-ee TAKEOVER-v1: even a truth-scripted takeover from Astra's failed states succeeded within the 300-step limit in 4/44);
   Barrel-3 b3 (TSTAVE phase 1 completed 0/5 vs CTRL 3/6: unprompted progress text displaced Astra's own work).
R4 check_hold is not surfaced (v1.8: 2/2 false alarms on real Astra closes); it may stay callable.
R5 Rate limit: at most one unprompted block per 25 control steps; unprompted text <= 2 % of the episode's text tokens (smoke: 0.0-0.4 %;
   the largest single turn carries one status block of about 90 tokens).
Triggers (deploy signals only): 'entering a fine stage' (gripper closed + lift detected from proprioception) -> availability fact for
inhand_pose / check_reach; others are L2 monitors.
"""
import re

ADVICE = re.compile(r"\b(you should|should|must|try to|try |we recommend|recommend|consider|instead|better to|please|regrasp it|put it down|"
                    r"do not|don't|avoid|make sure|next,? (?:call|move)|go to)\b", re.I)
NOT_SURFACED = {'check_hold'}
EXECUTORS = {'grasp_compatible': 'execute', 'align_and_insert': '*', 'regrasp': '*', 'insert_charger': '*'}
POLICY_TOKEN = 'l3_policy'   # an executor call authorised by an approved L3 policy (policy.py, mode auto) carries explicit=POLICY_TOKEN


def is_fact(line):
    return ADVICE.search(line) is None


def filter_unprompted(lines, log):
    out = []
    for l in lines:
        if is_fact(l): out.append(l)
        else: log.append(dict(dropped=l[:200], reason='advice wording'))
    return out


def availability_fact(tool, returns):
    return 'Tool %s is available; it returns %s.' % (tool, returns)


def allow_call(name, args, explicit):
    """Executor guard: an executor mode needs explicit=True (an Astra tool call), never an automatic trigger."""
    if name in NOT_SURFACED and not explicit: return False, '%s is not surfaced automatically' % name
    m = EXECUTORS.get(name)
    if m and (m == '*' or (args or {}).get('mode') == m) and explicit is not True and explicit != POLICY_TOKEN:
        return False, 'executor mode of %s runs only on an explicit call' % name
    return True, ''


class Gate:
    def __init__(self, min_gap_steps=25):
        self.last = -10 ** 9; self.gap = min_gap_steps; self.log = []; self.offered = set()

    def unprompted(self, step, lines):
        if step - self.last < self.gap: return []
        keep = filter_unprompted(lines, self.log)
        if keep: self.last = step
        return keep

    def on_fine_stage(self, step, gripper_closed_and_lifted):
        """One availability fact per episode when the fine stage starts (deploy signal from proprioception)."""
        if not gripper_closed_and_lifted or 'fine' in self.offered: return []
        self.offered.add('fine')
        return self.unprompted(step, [availability_fact('inhand_pose', 'the held object pose in the gripper (position, orientation, uncertainty)'),
                                      availability_fact('check_reach', 'whether a given end-effector pose is reachable from the current joints, and the nearest reachable shift')])
