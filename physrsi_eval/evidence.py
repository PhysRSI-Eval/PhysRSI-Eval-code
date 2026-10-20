"""m1_development: read-only evidence of this chain's earlier rounds, as H<k>/<file>.

Per round k (only that task's own chain; Context rounds and other tasks are never visible):
  H<k>/result.json             native grade of round k
  H<k>/failure-feedback.json   the outcome facts returned at the end of round k
  H<k>/reasoning-budget.json   measured reasoning time of round k
  H<k>/history.json            acknowledged actions
  H<k>/request_<n>.json        observation packets delivered to the agent
  H<k>/observations/<n>/<cam>.png
  H<k>/NOTES.md                r1_notes of round k
  H<k>/workspace/...           the committed workspace after the revision of round k
"""
import base64
import json
from pathlib import Path

from . import protocol as P
from .sim import jpeg_url
from .util import read_json


def round_dir(chain, k):
    return Path(chain) / f'K{k:02d}'


class ChainEvidence:
    def __init__(self, chain):
        self.chain = Path(chain)

    def committed(self, below):
        """[(k, receipt)] of committed rounds k < below, ascending."""
        rows = []
        for k in range(1, below):
            receipt = read_json(round_dir(self.chain, k) / 'client' / 'receipt.json')
            if receipt and (round_dir(self.chain, k) / 'snapshot').is_dir():
                rows.append((k, receipt))
        return rows

    def prior_outcomes(self, k):
        return [P.terminal_feedback(receipt, live=False) for _, receipt in self.committed(k)[-P.PRIOR_OUTCOMES:]]

    def entries(self, below, current=None):
        files = {}
        rounds = [(k, round_dir(self.chain, k), receipt, True) for k, receipt in self.committed(below)]
        if current is not None:  # the just-graded round, during its own revision (not yet committed)
            k, host = current
            rounds.append((k, round_dir(self.chain, k), host.receipt, False))
        for k, folder, receipt, committed in rounds:
            prefix, client = f'H{k}/', folder / 'client'
            files[prefix + 'result.json'] = ('json', client / 'result.json')
            files[prefix + 'failure-feedback.json'] = ('value', P.terminal_feedback(receipt, live=False))
            for name, kind, source in (('reasoning-budget.json', 'json', folder / 'agent' / 'reasoning-budget.json'),
                                       ('history.json', 'json', client / 'history.json'),
                                       ('NOTES.md', 'notes', client / 'episode-notes.json')):
                if source.is_file():
                    files[prefix + name] = (kind, source)
            for path in sorted(client.glob('request_*.json')):
                files[prefix + path.name] = ('json', path)
            for path in sorted(client.glob('observations/*/*.png')):
                files[prefix + path.relative_to(client).as_posix()] = ('png', path)
            if committed:
                snapshot = folder / 'snapshot'
                for path in sorted(snapshot.rglob('*')):
                    if path.is_file() and not path.is_symlink():
                        files[prefix + 'workspace/' + path.relative_to(snapshot).as_posix()] = ('workspace', path)
        return files

    def tool(self, name, below, current=None):
        entries = self.entries(below, current)
        if name == '':
            return dict(success=True, contentItems=[dict(type='inputText', text=json.dumps(sorted(entries)))])
        if name not in entries:
            return self._refuse('DEVELOPMENT_PATH_NOT_LISTED', name)
        kind, source = entries[name]
        if kind == 'value':
            return dict(success=True, contentItems=[dict(type='inputText', text=json.dumps(source, ensure_ascii=False))])
        if not source.is_file():
            return self._refuse('DEVELOPMENT_FILE_UNAVAILABLE', name)
        if source.stat().st_size > P.READ_LIMIT_BYTES:
            return dict(success=False, contentItems=[dict(type='inputText', text=json.dumps(dict(
                error='WORKSPACE_FILE_TOO_LARGE', path=name, size_bytes=source.stat().st_size,
                limit_bytes=P.READ_LIMIT_BYTES)))])
        raw = source.read_bytes()
        if kind == 'png' or (kind == 'workspace' and raw.startswith(b'\x89PNG\r\n\x1a\n')):
            if kind == 'workspace':
                url = 'data:image/png;base64,' + base64.b64encode(raw).decode()
            else:
                from PIL import Image
                import io
                url = jpeg_url(Image.open(io.BytesIO(raw)))
            return dict(success=True, contentItems=[dict(type='inputImage', imageUrl=url)])
        try:
            decoded = raw.decode('utf-8')
        except UnicodeDecodeError:
            return self._refuse('DEVELOPMENT_FILE_UNAVAILABLE', name)
        if kind == 'notes':
            decoded = (json.loads(decoded) or {}).get('text', '')
        return dict(success=True, contentItems=[dict(type='inputText', text=decoded)])

    @staticmethod
    def _refuse(error, name):
        return dict(success=False, contentItems=[dict(type='inputText', text=json.dumps(dict(error=error, path=name)))])
