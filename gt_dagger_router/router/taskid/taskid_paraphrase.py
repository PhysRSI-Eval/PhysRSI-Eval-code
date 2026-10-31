"""Task ID robustness to instruction wording (dispatcher 10-31): the archive has ONE instruction per task, so text-only = lookup.
Train exactly as taskid_text.py (archive, v2 grouping, 5-fold CV, original instructions, vocab from train rows); test the held-out rows
with reworded instructions of 5 kinds per task (official demo instructions are identical to the archive ones for all 9 demo tasks):
  template  'Please <instr>' / 'Could you <instr>?' / 'Your task: <instr>' (rotated per row)
  synonym   word / phrase swaps (place<->put, pick up->grab, into->in, block->cube, ...)
  order     clause order changed ('A, then B' -> 'B after you A'; else the last prepositional phrase moved to the front)
  terse     articles / fillers dropped, lower case ('plug charger into power strip')
  content   *_by_language tasks: colours / categories / order words permuted (the deploy-time variation of those tasks); other tasks:
            synonym + template combined.
Variants: text_only vs text+image.  -> RESULT-taskid-paraphrase.json"""
import itertools, json, re, collections
import numpy as np
import taskid_text as TT
SYN = [('pick up', 'grab'), ('place', 'put'), ('put', 'place'), ('insert', 'slide'), ('into', 'in'), ('onto', 'on'), ('blocks', 'cubes'), ('block', 'cube'),
       ('cup', 'mug'), ('robot arm', 'arm'), ('reset', 'return'), ('stack', 'pile'), ('pour', 'empty'), ('push', 'shove'), ('arrange', 'order'), ('hang', 'hook'),
       ('swap', 'exchange'), ('objects', 'items'), ('basket', 'bin'), ('tower', 'stack'), ('strike', 'hit'), ('fold', 'crease'), ('plug', 'connect'), ('lift', 'raise')]
def lc(s): return s[0].lower() + s[1:]
def strip(s): return s.strip().rstrip('.').strip()
def template(s, k): return ['Please %s.', 'Could you %s?', 'Your task: %s.'][k % 3] % lc(strip(s))
def synonym(s):
    out = s
    for i, (a, b) in enumerate(SYN): out = re.sub(r'(?i)\b%s\b' % re.escape(a), '\x01%d\x02' % i, out)   # placeholders: no re-swap of swapped words
    return re.sub(r'\x01(\d+)\x02', lambda m: SYN[int(m.group(1))][1], out)
def order(s):
    s0 = strip(s); m = re.match(r'(.*), then (.*)', s0)
    if m: return '%s after you %s.' % (m.group(2)[0].upper() + m.group(2)[1:], lc(m.group(1)))
    m = re.match(r'(.*) ((?:into|onto|on|in|to|with|from|by) .*)$', s0)
    return '%s, %s.' % (m.group(2)[0].upper() + m.group(2)[1:], lc(m.group(1))) if m else 'Now: ' + lc(s0) + '.'
def terse(s): return ' '.join(w for w in TT.toks(s) if w not in {'the', 'a', 'an', 'all', 'then', 'and', 'each', 'it', 'them', 'with', 'one', 'by'})
WORDS = [['blue', 'yellow', 'orange', 'red', 'green', 'violet', 'turquoise', 'white', 'black', 'brown'], ['car', 'watch', 'wooden_toy'], ['left', 'middle', 'right'], ['first', 'second', 'third']]
def content(s, task, k):
    if 'language' not in task: return template(synonym(s), k + 1)
    out = s
    for grp in WORDS:
        found = [w for w in re.findall(r'\b(%s)\b' % '|'.join(grp), out)]
        if len(set(found)) < 2: continue
        perms = [p for p in itertools.permutations(sorted(set(found), key=found.index)) if list(p) != sorted(set(found), key=found.index)]; mp = dict(zip(sorted(set(found), key=found.index), perms[k % len(perms)]))
        out = re.sub(r'\b(%s)\b' % '|'.join(map(re.escape, mp)), lambda m: mp[m.group(1)], out)
    return out
KINDS = dict(template=lambda s, t, k: template(s, k), synonym=lambda s, t, k: synonym(s), order=lambda s, t, k: order(s), terse=lambda s, t, k: terse(s), content=content)
examples = {t: {kn: f(TT.TA[i], t, 0) for kn, f in KINDS.items()} for t, i in ((t, int(np.where(TT.task == t)[0][0])) for t in TT.TASKS)}
res = dict(examples=examples, results={})
for name, ut, ui in (('text_only', 1, 0), ('text+image', 1, 1)):
    res['results'][name] = {}
    for kn, f in [('original', None)] + list(KINDS.items()):
        p = np.full(len(TT.YA), -1)
        for k in range(5):
            tr, te = np.where(TT.FO != k)[0], np.where(TT.FO == k)[0]
            p[te] = TT.fit_pred(TT.XA, TT.TA, TT.YA, tr, te, ut, ui, None if f is None else [f(TT.TA[r], TT.task[r], int(r)) for r in te])
        r = TT.report(TT.YA, p); r.pop('per_task'); r['tasks_below_0.9'] = {TT.TASKS[c]: round(float((p[TT.YA == c] == c).mean()), 3) for c in range(len(TT.TASKS)) if (TT.YA == c).any() and (p[TT.YA == c] == c).mean() < 0.9}
        res['results'][name][kn] = r; print(name, kn, r['acc'], r['balanced_acc'], len(r['tasks_below_0.9']), flush=True)
json.dump(res, open(TT.H / 'RESULT-taskid-paraphrase.json', 'w'), indent=1, default=str)
