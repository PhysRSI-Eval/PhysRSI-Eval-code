"""Three plug_in_charger conditions on sim/real. K0 is shared unseeded Context; K1..K17 vary by condition."""
import argparse
import copy
import json
import re
from pathlib import Path

from .util import read_json, write_json, sha256
from .workspace import build_success_seed, file_hashes

GROUPS = ('scratch', 'experience', 'tools')
HARDWARE = dict(arms='ARX X5 2023 (方舟无限)', wrist_cameras={'RealSense D345': 2, 'RealSense D405': 2},
                head_camera='Gemini D335L', camera_model_names='user-provided; verify serials/SDK mapping at integration')


def make_config(base, group, backend, source_domain=None):
    if group not in GROUPS or backend not in ('sim', 'real'):
        raise ValueError('unknown group/backend')
    cfg = copy.deepcopy(base)
    cfg['backend'] = backend
    cfg['hardware'] = HARDWARE if backend == 'real' else {'arms': 'RoboDojo dual ARX X5'}
    if group != 'tools':
        cfg.pop('plug_tools', None)
    elif not cfg.get('plug_tools'):
        raise ValueError('tools group requires plug_tools profile and inference services')
    if group == 'experience' and source_domain not in ('real', 'sim'):
        raise ValueError('experience group requires explicit real or sim source; never mix provenance')
    if group != 'experience' and source_domain is not None:
        raise ValueError('only experience group accepts a source domain')
    cfg['experiment'] = dict(task='plug_in_charger', group=group, target_domain=backend,
                            source_domain=source_domain, schedule='K0x3 + K1..K17',
                            transfer_order='real-to-real first; sim-to-real as separate follow-up',
                            seed_applies_to='K1 workspace only; K0 remains Context',
                            extra_tools=['insert_charger'] if group == 'tools' else [])
    return cfg


def preflight(cfg):
    """Read-only deployment checks. Never resets hardware or starts inference."""
    issues = []
    if cfg['backend'] == 'real':
        from .backends import validate_real_profile
        try:
            validate_real_profile(cfg['real']['profile'])
        except (ValueError, KeyError) as e:
            issues.append(str(e))
    else:
        for key in ('robodojo_root', 'robodojo_python'):
            if not Path(cfg.get(key, '')).exists():
                issues.append(f'{key} missing: {cfg.get(key)}')
    if cfg.get('plug_tools'):
        from .plug_pipeline import load_profile
        try:
            p = load_profile(cfg['plug_tools'], cfg['backend'])
            cfg['plug_tools']['profile_sha256'] = sha256(cfg['plug_tools']['profile'])
            if cfg['backend'] == 'real' and p['inference_calibration_id'] != cfg['real']['profile']['calibration_id']:
                issues.append('tool inference calibration differs from real deployment')
            urdf = Path(p['robot_kin']['urdf'])
            if not urdf.is_file() or sha256(urdf) != p['robot_kin']['urdf_sha256']:
                issues.append(f'URDF missing/hash mismatch: {urdf}')
        except (ValueError, KeyError, OSError) as e:
            issues.append(str(e))
    return issues


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['check', 'run'])
    p.add_argument('--config', required=True)
    p.add_argument('--group', choices=GROUPS, required=True)
    p.add_argument('--backend', choices=['sim', 'real'], required=True)
    p.add_argument('--source-domain', choices=['sim', 'real'])
    p.add_argument('--bundle', action='append', default=[])
    p.add_argument('--system', default='gpt-6-astra')
    p.add_argument('--name', default='c1')
    p.add_argument('--phase', choices=['k0', 'chain', 'all'], default='all')
    a = p.parse_args()
    cfg = make_config(read_json(a.config), a.group, a.backend, a.source_domain)
    if a.group != 'experience' and a.bundle:
        p.error('scratch/tools cannot receive an experience seed')
    if a.group == 'experience' and not a.bundle:
        p.error('experience requires --bundle with verified successful workspace')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.name):
        p.error('name must be a single safe directory component')
    issues = preflight(cfg)
    if a.command == 'check':
        print(json.dumps(dict(experiment=cfg['experiment'], issues=issues, physical_actions=0), indent=2))
        raise SystemExit(bool(issues))
    if issues:
        p.error('; '.join(issues))
    from .__main__ import load_system, bound_task
    from .loop import run_k0, run_chain
    system = load_system(a.system)
    task = bound_task('plug_in_charger', system)
    label = a.group + ('-from-' + a.source_domain if a.source_domain else '')
    root = Path(cfg.get('runs_dir', '/lizekai/RSIws/runs/plug-three-arm')) / a.backend / label / a.name
    identity = dict(config=cfg, system=system, bundles={str(Path(b).resolve()): file_hashes(b) for b in a.bundle})
    previous = read_json(root / 'experiment.json')
    if previous is not None and previous != identity:
        p.error('frozen experiment differs; choose a new --name')
    write_json(root / 'experiment.json', identity)
    seed = None
    if a.group == 'experience':
        seed = root / 'seed'
        if not seed.exists():
            build_success_seed(a.bundle, seed, a.source_domain)
        digest = file_hashes(seed)
        old = read_json(root / 'seed-hashes.json')
        if old is not None and old != digest:
            raise ValueError('seed changed after freezing')
        write_json(root / 'seed-hashes.json', digest)
    if a.phase in ('k0', 'all'):
        run_k0(cfg, system, task, root)
    if a.phase in ('chain', 'all'):
        run_chain(cfg, system, task, root / 'chain', seed=seed)
    print(root)


if __name__ == '__main__':
    main()
