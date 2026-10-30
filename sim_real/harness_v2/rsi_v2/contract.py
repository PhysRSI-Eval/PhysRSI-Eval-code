"""Uniform L1 return: {"value", "uncertainty", "abstain", "steps_used", "note"} (exactly these keys; note is English)."""
KEYS = ('value', 'uncertainty', 'abstain', 'steps_used', 'note')


def out(value=None, uncertainty=None, abstain=True, steps_used=0, note=''):
    return dict(value=value, uncertainty=uncertainty, abstain=bool(abstain), steps_used=int(steps_used), note=str(note))


def check(r):
    """Return a list of contract violations (empty = ok)."""
    errs = []
    if not isinstance(r, dict): return ['not a dict']
    if tuple(sorted(r)) != tuple(sorted(KEYS)): errs.append('keys %s' % sorted(r))
    if not isinstance(r.get('abstain'), bool): errs.append('abstain not bool')
    if not isinstance(r.get('steps_used'), int) or r['steps_used'] < 0: errs.append('steps_used not a non-negative int')
    if not isinstance(r.get('note'), str): errs.append('note not str')
    if r.get('abstain') is False and r.get('value') is None: errs.append('value missing while abstain=false')
    try:
        r.get('note', '').encode('ascii')
    except UnicodeEncodeError:
        errs.append('note not plain English/ASCII')
    return errs
