"""Command line.

  python -m physrsi_eval check     [--smoke TASK]
  python -m physrsi_eval k0        --system S --task T            # 3 Context episodes, mean score
  python -m physrsi_eval chain     --system S --task T [--chain c1] [--horizon 17]
  python -m physrsi_eval transfer  --system S --target T --source TASK=CHAIN_DIR [...] [--chain t1]
  python -m physrsi_eval video     --chain CHAIN_DIR --out DIR
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_config(path):
    path = Path(path or REPO / 'configs' / 'local.json')
    if not path.is_file():
        sys.exit(f'{path} not found: run scripts/setup.sh, or copy configs/local.example.json and edit it')
    return json.loads(path.read_text())


def load_system(name):
    """A name from configs/systems/ or a path to a system JSON file."""
    path = Path(name) if name.endswith('.json') else REPO / 'configs' / 'systems' / f'{name}.json'
    if not path.is_file():
        sys.exit(f'unknown system {name}; available: ' + ', '.join(p.stem for p in (REPO / 'configs' / 'systems').glob('*.json')))
    return json.loads(path.read_text())


def bound_task(task_id, system):
    from .loop import load_task
    task = load_task(task_id)
    task['public_task']['robot_and_action_contract'].update(model=system['model'], effort=system['effort'])
    return task


def main(argv=None):
    parser = argparse.ArgumentParser(prog='physrsi_eval', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', help='local config (default: configs/local.json)')
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check')
    check.add_argument('--smoke', metavar='TASK')
    for name in ('k0', 'chain', 'transfer'):
        p = sub.add_parser(name)
        p.add_argument('--system', required=True)
        p.add_argument('--target' if name == 'transfer' else '--task', required=True)
        if name != 'k0':
            p.add_argument('--chain', default='c1')
            p.add_argument('--horizon', type=int, default=17)
        if name == 'transfer':
            p.add_argument('--source', action='append', required=True, metavar='TASK=CHAIN_DIR')
    video = sub.add_parser('video')
    video.add_argument('--chain', required=True)
    video.add_argument('--out', required=True)
    for p in sub.choices.values():
        p.add_argument('--config', default=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.command == 'video':
        from .video import render_chain
        for target, frames in render_chain(args.chain, args.out):
            print(f'{target}  {frames} frames')
        return
    cfg = load_config(args.config)
    runs = Path(cfg.get('runs_dir', REPO / 'runs'))
    if args.command == 'check':
        from .check import run_checks, smoke
        results = run_checks(cfg)
        for name, ok, detail in results:
            print(f'[{"ok" if ok else "FAIL"}] {name}' + (f'  {detail}' if detail else ''))
        if args.smoke:
            print(json.dumps(smoke(cfg, args.smoke, runs / 'smoke'), indent=1))
        sys.exit(0 if all(ok for _, ok, _ in results) else 1)
    system = load_system(args.system)
    from .loop import run_chain, run_k0
    if args.command == 'k0':
        print(json.dumps(run_k0(cfg, system, bound_task(args.task, system), runs / system['name'] / args.task), indent=1))
    elif args.command == 'chain':
        rows = run_chain(cfg, system, bound_task(args.task, system), runs / system['name'] / args.task / args.chain,
                         horizon=args.horizon)
        print(json.dumps(rows, indent=1))
    else:
        from .transfer import build_seed
        sources = dict(item.split('=', 1) for item in args.source)
        chain = runs / system['name'] / args.target / ('transfer-' + args.chain)
        seed = chain.parent / (chain.name + '-seed')
        if not seed.exists():
            build_seed(args.target, sources, seed)
        rows = run_chain(cfg, system, bound_task(args.target, system), chain, horizon=args.horizon, seed=seed)
        print(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
