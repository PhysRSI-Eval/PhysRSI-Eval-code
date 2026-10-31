"""C1 (10-31 11:xxZ): job lists for the re-run of the expert demonstrations with the fixed recorder (tag er0/er1/er2 = perturb 0/1/2)
and the 1-state recorder smoke (tag ert). Built from the ed* lists (same states, same args) by re-tagging only."""
import json
D = '/lizekai/RSIws/experiments/macro-tools-20261027/c1/'


def retag(j, old, new):
    j = dict(j); a = list(j['args']); a[a.index('--tag') + 1] = new; j['args'] = a
    j['name'] = new.upper() + '-' + j['name'].split('-', 1)[1]; j['scratch'] = j['scratch'].replace('-%s-' % old, '-%s-' % new); return j


smoke = json.load(open(D + 'jobs-eds-smoke.json'))[:1]
json.dump([retag(smoke[0], 'eds', 'ert')], open(D + 'jobs-ert-smoke.json', 'w'), indent=0)
for k in (0, 1, 2):
    for h in ('aux', 'h1', 'h2'):
        J = json.load(open(D + 'jobs-ed%d-%s.json' % (k, h)))
        json.dump([retag(j, 'ed%d' % k, 'er%d' % k) for j in J], open(D + 'jobs-er%d-%s.json' % (k, h), 'w'), indent=0)
print('ok', json.load(open(D + 'jobs-ert-smoke.json'))[0]['scratch'])
