"""Per-step history for L3 policies (gap b). The native server already writes one file per control step while the episode runs:
<native output>/<episode>/<inner>/action_%06d.json = {executed_action[14], obs.states[14], step_id, ...} (robodojo_server/session.py; the adapter
knows this directory as NativeAdapter.output). The gate v2 features were computed from exactly these files (gate/build_dataset_v2.py), so the
online policy reads the same per-step source instead of per-turn snapshots. Incremental: only new files are parsed on each call.
  sl = StepLog(ad.output); S, A = sl()      # harness cfg: step_history=sl"""
import json
from pathlib import Path


class StepLog:
    def __init__(self, output_dir):
        self.root = Path(output_dir); self.inner = None; self.k = 0; self.S = []; self.A = []

    def _find(self):
        if self.inner is None:
            c = [p for p in self.root.iterdir() if p.is_dir() and (p / 'action_000000.json').exists()] if self.root.is_dir() else []
            if len(c) == 1: self.inner = c[0]
        return self.inner

    def __call__(self):
        d = self._find()
        if d is None: return self.S, self.A
        while True:
            f = d / ('action_%06d.json' % self.k)
            if not f.exists(): break
            try:
                r = json.loads(f.read_text())
            except (ValueError, OSError):   # the server may be writing this file now; read it on the next call
                break
            st = (r.get('obs') or {}).get('states')
            if st: self.S.append(st)
            self.A.append(r['executed_action']); self.k += 1
        return self.S, self.A
