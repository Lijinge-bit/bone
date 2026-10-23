"""Execute the PRD E1 frozen/full-finetuning pair on a real CUDA environment.

Run this script with .venv314-gpu's Python. No holdout evaluation or model
registration is performed. A failed preflight stops before any training.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import Settings
from app.training import atomic_json, sha256_file

INITIAL_SHA = '0a3eb7e2e11b427991d0d9e2ad2f219662a40c8d102147abb761fd788afbfd7b'


def run_suite(data, output, weights, seeds, python=None):
    data, output, weights = Path(data).resolve(), Path(output).resolve(), Path(weights).resolve()
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError('Provide distinct nonnegative integer seeds')
    if not weights.is_file() or sha256_file(weights) != INITIAL_SHA:
        raise ValueError('The expected V0.3 initialization weights are missing or changed')
    for name in ('audit.json', 'manifest.jsonl', 'rsna-bone-age-v2.zip'):
        if not (data / name).is_file():
            raise ValueError(f'Missing audited source file: {data / name}')
    suite = output / ('e1-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    suite.mkdir(parents=True, exist_ok=False)
    state = {'status': 'preflight', 'seeds': seeds, 'initial_checkpoint_sha256': INITIAL_SHA,
             'data': str(data), 'test_evaluated': False, 'model_registered': False, 'runs': []}
    status_path = suite / 'suite.json'
    atomic_json(status_path, state)
    executable = str(python or sys.executable)
    def execute(script, *arguments):
        subprocess.run([executable, str(ROOT / 'scripts' / script), *map(str, arguments)],
                       cwd=ROOT, check=True)
    try:
        execute('check_environment.py', '--device', 'cuda', '--output', suite / 'environment.json')
        report = json.loads((suite / 'environment.json').read_text())
        if report.get('status') != 'ready' or not str(report.get('resolved_device', '')).startswith('cuda:'):
            raise RuntimeError('The preflight did not verify real CUDA execution; training stopped')
        execute('prepare_training_data.py', '--data', data, '--size', 224)
        for seed in seeds:
            for arm in ('frozen', 'finetune'):
                run_dir = suite / f'{arm}-seed{seed}'
                state.update(status='training', current_run=str(run_dir))
                atomic_json(status_path, state)
                execute('train.py', '--config', ROOT / 'configs' / f'train_e1_4060_{arm}.json',
                        '--data', data, '--run-dir', run_dir, '--weights-path', weights, '--seed', seed)
                state['runs'].append({'arm': arm, 'seed': seed, 'run_dir': str(run_dir)})
                atomic_json(status_path, state)
        state.update(status='completed', current_run=None)
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                     error=f'{type(exc).__name__}: {exc}')
        atomic_json(status_path, state)
        raise
    atomic_json(status_path, state)
    return suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Settings.from_env().dataset)
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments')
    parser.add_argument('--weights', type=Path, default=Settings.from_env().storage / 'models/resnet50-frozen-cpu-224.pt')
    parser.add_argument('--seeds', nargs='+', type=int, default=[20261023])
    args = parser.parse_args()
    try:
        suite = run_suite(args.data, args.output, args.weights, args.seeds)
    except (Exception, KeyboardInterrupt) as exc:
        parser.exit(1, f'E1 stopped: {exc}\n')
    print(json.dumps({'suite': str(suite), 'status': 'completed', 'test_evaluated': False,
                      'model_registered': False}, indent=2))


if __name__ == '__main__':
    main()
