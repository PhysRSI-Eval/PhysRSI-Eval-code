"""C1 census summary (2026-10-29): classify the 'hover unreachable' truth rows from census v1 (out/rgcensus) and v2 (out/rgcensus2).
  python census_summary.py -> c1/out/CENSUS-SUMMARY.json (+ printed table)
Classes (v2 rows; v1 rows get the stage only):
  charger_lost            truth in-hand change > 80 mm before the hover (hold / transport failure, not reach)
  reached_this_run        the frozen chain reached the hover in this rerun (not reproducible as unreachable)
  alt_reaches             some in-sim alternative reached the hover; split by whether the first stall had an external contact:
                            reachable_by_different_approach (no external contact) / contact_avoided_by_different_approach
  unreachable_with_hold   no alternative reached (sub-label = external contact bodies at the first stall, or 'none')
External contact = an overlap hit of the charger / gripper colliders with a body that is not the same gripper or the charger."""
import glob, json, re, sys
from collections import Counter
V = '/lizekai/RSIws/experiments/macro-tools-20261027'


def body(path):
    p = path.lower()
    for k in ('table', 'socket', 'charger'):
        if k in p: return k
    m = re.search(r'(left|right)', p)
    return ('arm_' + m.group(1)) if m else p.split('/')[-2] if '/' in p else p


def ext_contacts(d, hold_arm):
    out = set()
    for q, hits in (d.get('contacts') or {}).items():
        qb = body(q)
        for h in hits:
            hb = body(h)
            if hb == qb: continue
            if {qb, hb} == {'charger', 'arm_' + hold_arm}: continue   # the hold itself
            out.add('%s~%s' % (qb, hb))
    return sorted(out)


def v1_row(f):
    r = json.load(open(f)); ch = r.get('chain') or {}; c = (ch.get('insert') or {}).get('census') or {}
    tr = ch.get('transport') or {}
    lost = ((tr.get('inhand_change_truth') or {}).get('mm') or 0) > 80
    stage = 'charger_lost' if lost else ('prehover_unreached' if (tr.get('err_mm') or 0) > 20 else ('hover_reached' if c.get('reached') else 'hover_stall_after_prehover'))
    return dict(ep=f.split('/')[-2][-8:], arm=ch.get('arm'), grasp=(ch.get('grasp') or {}).get('family'), transport_err_mm=tr.get('err_mm'), hover_err_mm=c.get('hover_err_mm'),
                stage=stage, chain_fail=ch.get('fail'))


def v2_row(f):
    r = json.load(open(f)); ch = r.get('chain') or {}; c2 = r.get('census2') or {}; out = c2.get('out') or {}; diag = c2.get('diag') or []
    arm = ch.get('arm') or out.get('arm'); first = diag[0] if diag else {}
    lost = any(d.get('charger_lost') for d in diag if not d.get('label', '').startswith('alt')) or out.get('class_hint') == 'charger_lost'
    tries = out.get('tries') or []; alt_ok = [t for t in tries if t.get('hover_reached')]
    ext = ext_contacts(first, arm or '') if first else []
    if ch.get('fail'): cls = 'chain_failed_before:' + str(ch.get('fail'))
    elif lost: cls = 'charger_lost'
    elif out.get('reached'): cls = 'reached_this_run'
    elif alt_ok: cls = 'contact_avoided_by_different_approach' if ext else 'reachable_by_different_approach'
    elif out: cls = 'unreachable_with_hold'
    else: cls = 'no_census'
    return dict(ep=f.split('/')[-2][-8:], arm=arm, grasp=(ch.get('grasp') or {}).get('family'), cls=cls, first_stall=first.get('label'), ext_contacts=ext,
                charger_zmin_above_table_mm=first.get('charger_zmin_above_table_mm'), tcp_above_table_mm=first.get('tcp_above_table_mm'),
                model_from_stall=first.get('model_from_stall'), cu_low=sorted((first.get('cu_clear_mm') or {}).items(), key=lambda kv: min(kv[1].values()))[:2],
                tries=[dict((k, t.get(k)) for k in ('outlet', 'sym', 'same_as_failed', 'hover_reached', 'hover_err_mm', 'inserted', 'steps')) for t in tries],
                inserted_by_alt=any(t.get('inserted') for t in tries), native_success=r.get('native_success'), diag_errors=[d.get('diag_error') for d in diag if d.get('diag_error')][:1])


if __name__ == '__main__':
    v1 = [v1_row(f) for f in sorted(glob.glob(V + '/out/rgcensus/*/result.json'))]
    v2 = [v2_row(f) for f in sorted(glob.glob(V + '/out/rgcensus2/*/result.json'))]
    res = dict(v1_counts=Counter(x['stage'] for x in v1), v2_counts=Counter(x['cls'] for x in v2), v1=v1, v2=v2)
    json.dump(res, open(V + '/c1/out/CENSUS-SUMMARY.json', 'w'), indent=1, default=str)
    print('v1', len(v1), dict(res['v1_counts'])); print('v2', len(v2), dict(res['v2_counts']))
    for x in v2: print(x['ep'], x['cls'], x['first_stall'], x['ext_contacts'], [(t['outlet'], t['sym'], t['hover_reached'], t['inserted']) for t in x['tries']])
