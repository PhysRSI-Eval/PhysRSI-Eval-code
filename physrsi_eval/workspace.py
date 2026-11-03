"""Domain-labelled success transfer and per-reset workspace bindings (never fake real success)."""
import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

from .util import read_json, write_json, sha256


def file_hashes(root):
    root = Path(root)
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('workspace symlinks are not transferable')
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}


def portability_audit(root):
    """Heuristic inventory, not proof of portability; export requires explicit file classifications."""
    rows = []
    for name in file_hashes(root):
        text = (Path(root) / name).read_text(errors='replace')
        paths = sorted(set(re.findall(r'/(?:lizekai|home|root|opt|tmp|workspace)/[^\s\"\'<>]+', text)))
        coordinates = bool(re.search(r'(?:xyz|position|pose|quaternion|joints|T_obj|outlets_xyz|center_xyz)', text, re.I))
        rows.append(dict(path=name, absolute_paths=paths, spatial_content=coordinates,
                         needs_review=bool(paths or coordinates)))
    return rows


def initialize_environment(workspace, cfg, task):
    folder = Path(workspace) / 'memory' / 'environment'
    # Environmental observations never survive a new reset as active bindings.
    if folder.exists():
        shutil.rmtree(folder)
    profile = cfg.get('real', {}).get('profile', {}) if cfg.get('backend') == 'real' else {}
    write_json(folder / 'deployment.json', dict(backend=cfg.get('backend', 'sim'), task=task['task_id'],
               robot=profile.get('robot', 'RoboDojo dual ARX X5'), frame=profile.get('frame', 'environment_origin'),
               calibration_id=profile.get('calibration_id', 'sim-native'), layout_id=cfg.get('layout_id', 0),
               hardware=cfg.get('hardware', {}), schema='physrsi-environment-v1'))
    write_json(folder / 'scene.json', dict(episode_id=None, status='unbound: observe after reset',
               objects={}, support_surfaces={}, collision_regions={}, camera_bindings={},
               instruction='Re-estimate scene positions from this reset; archived coordinates are source evidence only.'))


def bind_reset(workspace, reset):
    if workspace:
        path = Path(workspace) / 'memory/environment/scene.json'
        if path.exists():
            scene = read_json(path)
            scene.update(episode_id=reset['episode_id'], status='reset acquired; scene still requires observation')
            write_json(path, scene)


def export_success(round_dir, destination, domain, classifications):
    """Export one successful committed round; human-reviewed classifications decide active files."""
    if domain not in ('sim', 'real'):
        raise ValueError('source domain must be sim or real')
    source, dest = Path(round_dir), Path(destination)
    receipt = read_json(source / 'client/receipt.json')
    result = read_json(source / 'client/result.json')
    reset = read_json(source / 'client/reset.json')
    if not receipt or receipt.get('task_id') != 'plug_in_charger' or receipt.get('native_success') is not True or not result or result.get('native_success') is not True or result.get('valid') is not True or not reset or not reset.get('episode_id'):
        raise ValueError('requires original native-success receipt/result/reset; hypothetical success is not exportable')
    deployment = read_json(source / 'workspace/memory/environment/deployment.json', {})
    if deployment.get('backend') != domain:
        raise ValueError('source domain does not match recorded deployment')
    if list((source / 'client').glob('pending-*.json')):
        raise ValueError('unresolved physical request in source round')
    snapshot = source / 'snapshot'
    hashes = file_hashes(snapshot)
    if not hashes:
        raise ValueError('committed workspace snapshot required')
    if set(classifications) != set(hashes):
        raise ValueError('classify every snapshot file: portable / episode_local / environment_bound / software_bound')
    allowed = {'portable', 'episode_local', 'environment_bound', 'software_bound'}
    if any(v not in allowed for v in classifications.values()):
        raise ValueError('invalid portability classification')
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True)
    shutil.copytree(snapshot, dest / 'original-workspace')
    (dest / 'evidence').mkdir(exist_ok=True)
    for name in ('receipt.json', 'result.json', 'reset.json', 'history.json'):
        if (source / 'client' / name).is_file():
            shutil.copy2(source / 'client' / name, dest / 'evidence' / name)
    for name, category in classifications.items():
        if category == 'portable':
            if name.startswith(('memory/environment/', 'memory/library/')) or name == 'tools/mcp_client.py':
                raise ValueError('bindings, libraries and host helper cannot be active transferred files')
            target = dest / 'portable' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snapshot / name, target)
    manifest = dict(schema='physrsi-success-v1', task='plug_in_charger', source_domain=domain,
                    native_success=True, episode_id=reset['episode_id'], source_round=str(source.resolve()),
                    classifications=classifications, audit=portability_audit(snapshot), files=file_hashes(dest))
    write_json(dest / 'manifest.json', manifest)
    return manifest


def build_success_seed(bundles, destination, source_domain):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.seed-', dir=destination.parent) as staging:
        ready = Path(staging) / 'ready'
        _build_success_seed(bundles, ready, source_domain)
        ready.rename(destination)
    return destination


def _build_success_seed(bundles, destination, source_domain):
    dest = Path(destination)
    if dest.exists():
        raise FileExistsError(dest)
    checked = []
    for bundle in bundles:
        root = Path(bundle)
        m = read_json(root / 'manifest.json')
        if not m or m.get('task') != 'plug_in_charger' or m.get('native_success') is not True or m.get('source_domain') != source_domain:
            raise ValueError('success bundle task/domain/outcome mismatch')
        actual = file_hashes(root); actual.pop('manifest.json', None)
        if actual != m['files']:
            raise ValueError('success bundle content/hash mismatch')
        checked.append((root, m))
    if not checked:
        raise ValueError('experience group requires a verified success bundle')
    dest.mkdir(parents=True)
    loaded = []
    for i, (root, m) in enumerate(checked):
        # Full originals remain readable, explicitly under SOURCE EVIDENCE, not active tools/policies.
        library = dest / 'memory/library' / f'{source_domain}-{i:02d}'
        shutil.copytree(root, library)
        for name in file_hashes(root / 'portable'):
            src, target = root / 'portable' / name, dest / name
            if target.exists() and sha256(target) != sha256(src):
                raise ValueError(f'conflicting portable file {name}; merge explicitly before experiment')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
        loaded.append(dict(episode_id=m['episode_id'], source_domain=source_domain, manifest_sha256=sha256(root / 'manifest.json')))
    write_json(dest / 'memory/library/transfer.json', dict(sources=loaded, tool_grants=[],
               instruction='Original workspaces and coordinates are historical evidence, never current-scene commands. Tools described in source experience are NOT automatically available. Re-estimate geometry on this reset.'))
    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit'); a.add_argument('workspace')
    e = sub.add_parser('export'); e.add_argument('round'); e.add_argument('destination')
    e.add_argument('--domain', choices=['sim', 'real'], required=True)
    e.add_argument('--classifications', required=True, help='JSON path -> portability category')
    a = p.parse_args()
    result = portability_audit(a.workspace) if a.command == 'audit' else export_success(a.round, a.destination, a.domain, read_json(a.classifications))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
