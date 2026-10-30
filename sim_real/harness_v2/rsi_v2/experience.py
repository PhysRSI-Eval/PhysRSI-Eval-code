"""L4 experience package (harness v2 RC), RoboSkill-style ("Explore, Execute, Evolve"; ideas re-implemented, nothing reused).
After a server-authoritative native success, the SAME Astra session (no more stepping) writes one package (English). The harness validates
STRUCTURE and REFERENCES (not semantics), returns the error list once for one repair attempt, then freezes the package read-only.
Promotion into the skill library is a separate, stricter gate (offline replay of the recompute methods on held-out Astra states).

Schema rsi-exp-v1:
{ "schema": "rsi-exp-v1", "task": str, "episode": str, "outcome": {"native_success": bool, "steps": int},
  "action_ledger": [ { "purpose": str, "contact_mode": str, "step_span": [int, int], "method": str,
                       "tool_calls": [ {"tool": str (registry name), "inputs": {...}, "returned": {...}, "used_next": "yes"|"no"|"unknown"} ],
                       "outcome": str, "observed_effect": str, "failure_mechanism": str|null, "next_adjustment": str|null,
                       "reusable_lesson": str, "evidence_steps": [int] } ],
  "keyframes": [ {"step": int, "camera": str, "why": str} ],
  "binding": { "episode_local_values": [ {"name": str, "value": any, "unit": str,
                                          "recompute": {"method": "tool"|"proprio"|"cad"|"robot", "tool": str (if tool), "args": {...}} } ],
               "general_facts": [str] },
  "program_index": [ {"module": str, "sha256": str, "ran": bool} ],
  "summary": str }
Binding rule (our generalisation contract applied to experience): lessons and general facts contain NO numeric literals of positions or
sizes (a decimal with 2+ decimals, or a value with m/mm/cm/deg/rad); they use the NAME of an episode_local_values entry, and every entry has a
recompute method from OUR deploy-legal sources: a registry tool (locate_mm, inhand_pose, check_reach, predict_outcome, grasp_compatible),
proprioception, the CAD model, or a robot constant.
"""
import copy, hashlib, json, re
from pathlib import Path

REQ_TOP = ('schema', 'task', 'episode', 'outcome', 'action_ledger', 'keyframes', 'binding', 'program_index', 'summary')
REQ_LEDGER = ('purpose', 'contact_mode', 'step_span', 'method', 'tool_calls', 'outcome', 'observed_effect', 'failure_mechanism',
              'next_adjustment', 'reusable_lesson', 'evidence_steps')
NUM = re.compile(r'(?<![\w.])-?\d+\.\d{2,}|\b\d+(?:\.\d+)?\s?(?:mm|cm|m|deg|rad)\b')


def validate(pkg, tools, frames_available=None):
    """-> list of English error strings (empty = valid). tools: registry names; frames_available: set of stored step ids or None."""
    E = []
    if not isinstance(pkg, dict): return ['package is not a JSON object']
    for k in REQ_TOP:
        if k not in pkg: E.append('missing top-level key "%s"' % k)
    if E: return E
    if pkg['schema'] != 'rsi-exp-v1': E.append('schema must be "rsi-exp-v1"')
    steps = (pkg.get('outcome') or {}).get('steps')
    if not isinstance(steps, int): E.append('outcome.steps must be an integer')
    if not isinstance((pkg.get('outcome') or {}).get('native_success'), bool): E.append('outcome.native_success must be true/false')
    bound = {v.get('name') for v in (pkg['binding'].get('episode_local_values') or []) if isinstance(v, dict)}
    bound_vals = set()
    for i, v in enumerate(pkg['binding'].get('episode_local_values') or []):
        rc = (v or {}).get('recompute') or {}; bound_vals.add(str(v.get('value')))
        if rc.get('method') not in ('tool', 'proprio', 'cad', 'robot'): E.append('binding value %d (%s): recompute.method must be tool, proprio, cad or robot' % (i, v.get('name')))
        elif rc['method'] == 'tool' and rc.get('tool') not in tools: E.append('binding value %d (%s): recompute tool %r is not a registry tool' % (i, v.get('name'), rc.get('tool')))
    for i, a in enumerate(pkg['action_ledger']):
        for k in REQ_LEDGER:
            if k not in a: E.append('action %d: missing "%s"' % (i, k))
        sp = a.get('step_span')
        if not (isinstance(sp, list) and len(sp) == 2 and all(isinstance(x, int) for x in sp) and 0 <= sp[0] <= sp[1] <= (steps or 10 ** 9)):
            E.append('action %d: step_span must be [start, end] within 0..outcome.steps' % i)
        for j, c in enumerate(a.get('tool_calls') or []):
            if c.get('tool') not in tools: E.append('action %d call %d: tool %r is not in the registry' % (i, j, c.get('tool')))
            if c.get('used_next') not in ('yes', 'no', 'unknown'): E.append('action %d call %d: used_next must be yes, no or unknown' % (i, j))
        if frames_available is not None:
            miss = [s for s in a.get('evidence_steps') or [] if s not in frames_available]
            if miss: E.append('action %d: evidence steps %s have no stored frame' % (i, miss[:5]))
        for m in NUM.findall(a.get('reusable_lesson') or ''):
            E.append('action %d: the lesson contains the number "%s"; write the name of a binding entry instead (episode_local_values with a recompute method)' % (i, m.strip()))
    for t in pkg['binding'].get('general_facts') or []:
        for m in NUM.findall(t):
            E.append('general fact contains the number "%s"; write the name of a binding entry instead' % m.strip())
    for k in pkg.get('keyframes') or []:
        if frames_available is not None and k.get('step') not in frames_available: E.append('keyframe step %s has no stored frame' % k.get('step'))
    for p in pkg.get('program_index') or []:
        f = Path(p.get('module', ''))
        if f.exists() and hashlib.sha256(f.read_bytes()).hexdigest() != p.get('sha256'): E.append('program_index: %s sha256 does not match the file that ran' % p.get('module'))
    try:
        json.dumps(pkg).encode('ascii')
    except UnicodeEncodeError:
        E.append('package must be English (ASCII) text')
    return E


def finalize(pkg, tools, repair=None, frames_available=None):
    """Validate; if errors and a repair callable is given, call it ONCE with (package copy, errors) and validate again; then freeze.
    -> dict(status='valid'|'invalid_after_repair'|'invalid_no_repair', errors_first, errors_final, package, sha256)."""
    e1 = validate(pkg, tools, frames_available); final = pkg; e2 = e1; status = 'valid'
    if e1:
        if repair is None:
            status = 'invalid_no_repair'
        else:
            final = repair(copy.deepcopy(pkg), list(e1)); e2 = validate(final, tools, frames_available); status = 'valid' if not e2 else 'invalid_after_repair'
    blob = json.dumps(final, sort_keys=True)
    return dict(status=status, errors_first=e1, errors_final=e2, package=json.loads(blob), sha256=hashlib.sha256(blob.encode()).hexdigest(), frozen=True)
