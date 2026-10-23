"""Compare real inference/Grad-CAM against the original Python 3.12 runtime.

Works with the six public samples in the full bundle. --bundle prepares only
checksum-verified model/sample files, so source CI also exercises real weights.
Optional --cache repeats the fixed holdout evaluation without retraining.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import Settings
from app.network import denormalize_target, normalize_image
from app.predictor import Predictor
from scripts.train import metrics


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def checked_write(path, data, expected_sha):
    require(hashlib.sha256(data).hexdigest() == expected_sha,
            f"Bundle checksum mismatch: {path.name}")
    if path.exists():
        require(path.read_bytes() == data, f"Refusing to replace existing file: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def prepare_bundle(bundle, storage, reference):
    # Fixed member names: no extractall, database copies, or arbitrary ZIP paths.
    with ZipFile(bundle) as archive:
        raw = archive.read('bone-atlas/var/models/active.json')
        meta = json.loads(raw)
        require(meta['model_id'] == reference['model_id'], 'Unexpected bundled model')
        name = meta['checkpoint']
        require(Path(name).name == name and '/' not in name and '\\' not in name,
                'Invalid checkpoint filename')
        require(meta['checkpoint_sha256'] == reference['checkpoint_sha256'],
                'Unexpected bundled checkpoint')
        checked_write(storage / 'models' / name,
                      archive.read('bone-atlas/var/models/' + name),
                      reference['checkpoint_sha256'])
        checked_write(storage / 'models/active.json', raw, hashlib.sha256(raw).hexdigest())
        for row in reference['samples']:
            name = row['image_sha256'] + '.png'
            checked_write(storage / 'images' / name,
                          archive.read('bone-atlas/var/images/' + name), row['image_sha256'])


def check_samples(predictor, storage, reference):
    meta = predictor.metadata()
    require(meta is not None, 'Real model missing; use the full research bundle')
    require(meta['model_id'] == reference['model_id'] and
            meta['checkpoint_sha256'] == reference['checkpoint_sha256'],
            'Reference belongs to a different checkpoint')
    results = []
    for row in reference['samples']:
        path = storage / 'images' / (row['image_sha256'] + '.png')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == row['image_sha256'],
                f"Public sample checksum mismatch: {row['sample']}")
        with Image.open(path) as image:
            value, heatmap, error = predictor.assess(
                image, row['sex'], meta['model_id'], meta['checkpoint_sha256'])
            repeated, _, repeat_error = predictor.assess(
                image, row['sex'], meta['model_id'], meta['checkpoint_sha256'])
            require(error is None and repeat_error is None and heatmap is not None,
                    f"Grad-CAM unavailable: {row['sample']} ({error})")
            require(np.isfinite(value) and 0 <= value <= 360, 'Invalid model output')
            difference = abs(value - row['prediction_months'])
            require(difference <= reference['prediction_tolerance_months'],
                    f"Prediction changed: {row['sample']}/{row['sex']}: {difference}")
            require(abs(value - repeated) <= 1e-6, 'Inference is not repeatable')
            require(heatmap.mode == 'RGBA' and heatmap.size == image.size == tuple(row['image_size']),
                    'Grad-CAM mode/geometry changed')
            rgba = np.asarray(heatmap)
            alpha = rgba[:, :, 3]
            require(0 < int(alpha.max()) <= 160 and np.ptp(alpha) > 0,
                    'Grad-CAM alpha is empty or constant')
            grid = np.asarray(heatmap.getchannel('A').resize(
                (16, 16), Image.Resampling.BILINEAR), dtype=np.int16).ravel()
            grid_delta = int(np.abs(grid - row['alpha_grid_16x16']).max())
            require(grid_delta <= reference['heatmap_grid_tolerance'],
                    'Grad-CAM spatial attention changed beyond tolerance')
            mean_delta = abs(float(alpha.mean()) - row['alpha_mean'])
            require(mean_delta <= reference['heatmap_alpha_mean_tolerance'],
                    'Grad-CAM opacity changed beyond tolerance')
            results.append({'sample': row['sample'], 'sex': row['sex'],
                'prediction_months': value, 'baseline_months': row['prediction_months'],
                'absolute_difference_months': difference, 'repeat_difference_months': abs(value-repeated),
                'heatmap_size': list(heatmap.size), 'heatmap_mode': heatmap.mode,
                'heatmap_pixels_identical': hashlib.sha256(rgba.tobytes()).hexdigest() == row['heatmap_sha256'],
                'heatmap_alpha_grid_max_difference': grid_delta, 'heatmap_alpha_mean_difference': mean_delta})
    return results


def check_holdout(predictor, cache_path):
    audit = json.loads((cache_path.parent / 'audit.json').read_text())
    sha = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    require(sha == audit['cache_sha256'], 'Cache checksum mismatch')
    with np.load(cache_path, allow_pickle=False) as cache:
        indices = np.flatnonzero(cache['split'] == 2)
        images, sexes = cache['images'][indices], cache['sex'][indices]
        truth, ids = cache['targets'][indices], cache['ids'][indices]
    with (predictor.directory / 'test-predictions.csv').open(newline='') as stream:
        baseline = list(csv.DictReader(stream))
    require(len(indices) == len(baseline) == predictor.metadata()['evaluation']['test_samples'],
            'Holdout sample count changed')
    require([str(i) for i in ids] == [r['source_image_id'] for r in baseline],
            'Holdout identities/order changed')
    require(np.array_equal(truth, [float(r['reference_months']) for r in baseline]) and
            np.array_equal(sexes, [1.0 if r['sex'] == 'M' else 0.0 for r in baseline]),
            'Holdout labels/sex changed')
    outputs = []
    with torch.inference_mode():
        for offset in range(0, len(indices), 96):
            x = normalize_image(torch.from_numpy(images[offset:offset+96]).unsqueeze(1))
            sex = torch.from_numpy(sexes[offset:offset+96]).unsqueeze(1)
            outputs.extend(denormalize_target(predictor.model(x, sex)).tolist())
    outputs = np.asarray(outputs)
    require(np.isfinite(outputs).all(), 'Non-finite holdout prediction')
    original = np.asarray([float(r['predicted_months']) for r in baseline])
    outside = (outputs < 0) | (outputs > 360)
    require(np.array_equal(outside, (original < 0) | (original > 360)),
            'New out-of-range predictions introduced by runtime upgrade')
    delta = np.abs(outputs - original)
    require(float(delta.max()) <= .001, 'Holdout predictions changed beyond .001 months')
    meta = predictor.metadata()
    for index in np.flatnonzero(outside):
        try:
            predictor.assess(Image.fromarray(images[index]), 'M' if sexes[index] else 'F',
                             meta['model_id'], meta['checkpoint_sha256'])
        except RuntimeError as exc:
            require('有效性检查' in str(exc), f'Unexpected rejection reason: {exc}')
        else:
            raise RuntimeError('Predictor accepted an out-of-range real model output')
    result = metrics(outputs, truth, sexes)
    require(abs(result['mae_months'] - predictor.metadata()['evaluation']['test']['mae_months']) <= .0001,
            'Holdout MAE changed')
    return {'cache_sha256': sha, 'comparison': 'Original rounded CSV, same fixed split; no model selection',
            'metrics_use_unclipped_predictions': True,
            'out_of_range_count': int(outside.sum()),
            'out_of_range_guard_verified': int(outside.sum()),
            'out_of_range_source_ids': [str(i) for i in ids[outside]],
            'out_of_range_policy': 'Existing model limitation, unchanged; app Predictor rejects outside 0..360 months',
            'max_prediction_difference_months': float(delta.max()), 'metrics': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage', type=Path, default=Settings.from_env().storage)
    parser.add_argument('--reference', type=Path, default=ROOT / 'docs/reports/runtime-reference.json')
    parser.add_argument('--bundle', type=Path, help='Prepare verified public model/images in --storage')
    parser.add_argument('--cache', type=Path, help='Optional audited cache-96.npz for all 1,260 holdout images')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    if args.bundle:
        prepare_bundle(args.bundle, args.storage, reference)
    predictor = Predictor(args.storage / 'models')
    samples = check_samples(predictor, args.storage, reference)
    report = {'status': 'passed', 'created_at': datetime.now(timezone.utc).isoformat(),
        'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                    'platform': platform.platform(), 'device': 'cpu', 'threads': torch.get_num_threads()},
        'checkpoint_sha256': reference['checkpoint_sha256'],
        'baseline_runtime': {'python': reference['python'], 'torch': reference['torch']},
        'retrained': False, 'sample_image_sex_pairs': len(samples),
        'max_prediction_difference_months': max(r['absolute_difference_months'] for r in samples),
        'tolerances': {k: v for k, v in reference.items() if 'tolerance' in k},
        'samples': samples, 'holdout': check_holdout(predictor, args.cache) if args.cache else None}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
