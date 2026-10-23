"""Export version-bound RSNA unlabeled predictions, with no accuracy claim."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from zipfile import ZipFile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import Settings
from app.imaging import letterbox, validate_png
from app.model_registry import atomic_write, canonical_sha, json_bytes, sha256
from app.network import build_model, denormalize_target, normalize_for_model
from app.predictor import Predictor
from app.training import resolve_device


def read_unlabeled_manifest(directory):
    path = Path(directory) / 'manifest.jsonl'
    audit_path = Path(directory) / 'audit.json'
    if not path.is_file() and not audit_path.is_file():
        return None
    if not path.is_file() or not audit_path.is_file():
        raise ValueError('Unlabeled source audit/manifest is incomplete')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if sha256(path) != audit.get('manifest_sha256'):
        raise ValueError('Unlabeled manifest checksum mismatch')
    records = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('kind') != 'unlabeled':
                continue
            key = row.get('source_id')
            if key in records or row.get('status') != 'valid':
                raise ValueError('Duplicate or invalid unlabeled manifest record')
            records[key] = row
    return records


def predict_archive(archive_path, models, output, *, candidate=False, device='cpu',
                    batch_size=16, table=ROOT / 'boneage-test-dataset.csv'):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    device = resolve_device(device)
    predictor = Predictor(models, metadata_filename='candidate.json' if candidate else 'active.json')
    meta = predictor.metadata()
    if meta is None:
        raise ValueError('No compatible registered/candidate model metadata')
    identity = canonical_sha(meta)
    raw = (Path(models) / meta['checkpoint']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta['checkpoint_sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    model = build_model(meta['architecture'], pretrained=False)
    model.load_state_dict(torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True), strict=True)
    del raw
    model.to(device).eval()
    with Path(table).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids = [row.get('Case ID', '') for row in rows]
    if not rows or len(ids) != len(set(ids)) or any(not value.isdecimal() for value in ids):
        raise ValueError('Missing, duplicate or unsafe unlabeled source ID')
    if any(row.get('Sex') not in ('M', 'F') for row in rows):
        raise ValueError('Unlabeled sex must be M/F')
    manifest = read_unlabeled_manifest(Path(archive_path).parent)
    if manifest is not None and set(manifest) != set(ids):
        raise ValueError('Unlabeled CSV and audited image IDs differ')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix='.unlabeled-', suffix='.csv', dir=output.parent)
    temporary = Path(temporary_name)
    valid_count = 0
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='') as stream, \
                ZipFile(archive_path) as archive, torch.inference_mode():
            writer = csv.writer(stream)
            writer.writerow(['Case ID', 'Sex', 'predicted_boneage_months', 'model_id', 'status'])
            for offset in range(0, len(rows), batch_size):
                group = rows[offset:offset + batch_size]
                images = []
                for row in group:
                    member = f"boneage-test-dataset/boneage-test-dataset/{row['Case ID']}.png"
                    image_data = archive.read(member)
                    if manifest is not None:
                        record = manifest[row['Case ID']]
                        if (record.get('sex') != row['Sex'] or record.get('archive_path') != member or
                                hashlib.sha256(image_data).hexdigest() != record.get('sha256')):
                            raise ValueError('Unlabeled source image/sex checksum mismatch')
                    array, _ = letterbox(validate_png(image_data), meta['image_size'])
                    images.append(torch.from_numpy(array).unsqueeze(0))
                tensor = normalize_for_model(torch.stack(images).to(device), meta['preprocess_version'])
                sex = torch.tensor([[float(row['Sex'] == 'M')] for row in group], device=device)
                values = denormalize_target(model(tensor, sex)).cpu().tolist()
                if len(values) != len(group):
                    raise RuntimeError('Model returned the wrong prediction count')
                for row, value in zip(group, values):
                    valid = bool(np.isfinite(value) and 0 <= value <= 360)
                    valid_count += int(valid)
                    writer.writerow([row['Case ID'], row['Sex'], round(value, 4) if valid else '',
                                     meta['model_id'], 'research_prediction' if valid else 'invalid_output'])
            stream.flush()
            os.fsync(stream.fileno())
        current = predictor.metadata()
        if current is None or canonical_sha(current) != identity:
            raise RuntimeError('Model metadata changed during export; previous export preserved')
        report = {'model_id': meta['model_id'], 'checkpoint_sha256': meta['checkpoint_sha256'],
                  'model_metadata_sha256': identity, 'architecture': meta['architecture'],
                  'preprocess_version': meta['preprocess_version'], 'image_size': meta['image_size'],
                  'rows': len(rows), 'valid_rows': valid_count, 'invalid_rows': len(rows) - valid_count,
                  'device': str(device), 'ground_truth_available': False, 'mae_computed': False,
                  'source_image_checksums_verified': manifest is not None,
                  'predictions_sha256': sha256(temporary)}
        atomic_write(output.with_suffix(output.suffix + '.metadata.json'), json_bytes(report))
        os.replace(temporary, output)
        return report
    finally:
        temporary.unlink(missing_ok=True)


def main():
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=settings.dataset / 'rsna-bone-age-v2.zip')
    parser.add_argument('--models', type=Path, default=settings.storage / 'models')
    parser.add_argument('--candidate', type=Path, help='Use candidate.json in this directory')
    parser.add_argument('--output', type=Path, default=settings.storage / 'unlabeled-predictions.csv')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    report = predict_archive(args.archive, args.candidate or args.models, args.output,
                             candidate=args.candidate is not None, device=args.device, batch_size=args.batch_size)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
