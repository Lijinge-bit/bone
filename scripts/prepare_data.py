"""Audit the original Kaggle archive and build a small CPU training cache.

This never substitutes image IDs for patient IDs. Original PNGs stay in the ZIP.
"""
import argparse
import csv
import hashlib
import io
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.database import now
from app.imaging import PREPROCESS_VERSION, letterbox, validate_png


def grouped_split(labels, groups, seed):
    """Stratify group representatives by sex/boneage; keep duplicates together."""
    bins = defaultdict(list)
    for group, indices in groups.items():
        sex = labels[indices[0]][1]
        age = labels[indices[0]][0]
        bins[(sex, min(int(age // 24), 9))].append(group)
    rng = np.random.default_rng(seed)
    split = np.zeros(len(labels), dtype=np.uint8)
    for key in sorted(bins):
        keys = np.array(sorted(bins[key]))
        rng.shuffle(keys)
        n = len(keys)
        nv = max(1, round(n * .1)) if n >= 10 else 0
        nt = max(1, round(n * .1)) if n >= 10 else 0
        for i, group in enumerate(keys):
            code = 2 if i < nt else 1 if i < nt + nv else 0
            for index in groups[group]:
                split[index] = code
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', type=Path, default=Path('/workspace/bone-data/rsna-bone-age-v2.zip'))
    ap.add_argument('--output', type=Path, default=Path('/workspace/bone-data'))
    ap.add_argument('--size', type=int, default=96)
    ap.add_argument('--seed', type=int, default=20261022)
    args = ap.parse_args()
    if not 64 <= args.size <= 256:
        ap.error('size must be 64..256')
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    with (root / 'boneage-training-dataset.csv').open() as f:
        train = list(csv.DictReader(f))
    with (root / 'boneage-test-dataset.csv').open() as f:
        test = list(csv.DictReader(f))
    start = time.monotonic()
    anomalies = []
    manifest = []
    labels, ids, tensors = [], [], []
    exact, near = defaultdict(list), defaultdict(list)
    fingerprints = {}
    match_count = {'train': 0, 'unlabeled': 0}
    dimensions = Counter()
    modes = Counter()
    print('Auditing original ZIP with per-file CRC and PNG decoding…', flush=True)
    with ZipFile(args.archive) as z:
        by_id = {'train': defaultdict(list), 'unlabeled': defaultdict(list)}
        png_entries = []
        for info in z.infolist():
            if not info.filename.lower().endswith('.png'):
                continue
            png_entries.append(info.filename)
            kind = 'unlabeled' if 'boneage-test-dataset/' in info.filename else 'train'
            by_id[kind][Path(info.filename).stem].append(info.filename)
        csv_hashes = {}
        for name in ('boneage-training-dataset.csv', 'boneage-test-dataset.csv'):
            if name in z.namelist():
                archive_hash = hashlib.sha256(z.read(name)).hexdigest()
                local_hash = hashlib.sha256((root / name).read_bytes()).hexdigest()
                csv_hashes[name] = {'sha256': local_hash, 'archive_matches': archive_hash == local_hash}
                if archive_hash != local_hash:
                    raise RuntimeError(f'CSV mismatch: {name}; inspect source before training')
        for kind, rows in (('train', train), ('unlabeled', test)):
            known = {r['id' if kind == 'train' else 'Case ID'] for r in rows}
            for extra in sorted(set(by_id[kind]) - known):
                anomalies.append({'kind': kind, 'id': extra, 'reason': 'image_without_csv_row'})
            for index, row in enumerate(rows):
                source_id = row['id' if kind == 'train' else 'Case ID']
                paths = by_id[kind].get(source_id, [])
                item = {'source_id': source_id, 'kind': kind, 'patient_id': None,
                        'boneage': int(row['boneage']) if kind == 'train' else None,
                        'sex': ('M' if row['male'] == 'True' else 'F') if kind == 'train' else row['Sex']}
                if len(paths) != 1:
                    item['status'] = 'missing' if not paths else 'ambiguous_path'
                    anomalies.append(item.copy()); manifest.append(item); continue
                info = z.getinfo(paths[0])
                if info.file_size > 30_000_000:
                    item['status'] = 'oversize'; anomalies.append(item.copy()); manifest.append(item); continue
                try:
                    data = z.read(paths[0])  # ZipFile verifies CRC on read.
                    image = validate_png(data)
                    array, _ = letterbox(image, args.size)
                    sha = hashlib.sha256(data).hexdigest()
                    # Exact decoded raster detects identical content with different PNG encodings.
                    raster_sha = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                    dh = image.resize((17, 16), Image.Resampling.BILINEAR)
                    pixels = np.asarray(dh)
                    # Conservative candidate bucket; no claim that all near duplicates were found.
                    difference_hash = np.packbits(pixels[:, 1:] > pixels[:, :-1]).tobytes().hex()
                    item.update({'status': 'valid', 'archive_path': paths[0], 'sha256': sha,
                                 'raster_sha256': raster_sha, 'dhash': difference_hash,
                                 'width': image.width, 'height': image.height, 'mode': image.mode})
                    match_count[kind] += 1
                    dimensions[f'{image.width}x{image.height}'] += 1; modes[image.mode] += 1
                    fingerprints.setdefault(raster_sha, []).append((kind, source_id))
                    if kind == 'train':
                        n = len(labels)
                        ids.append(source_id); labels.append((float(row['boneage']), 1.0 if row['male'] == 'True' else 0.0))
                        tensors.append(array)
                        exact[raster_sha].append(n); near[difference_hash].append(n)
                except Exception as exc:
                    item.update({'status': 'decode_failed', 'reason': str(exc)[:200]})
                    anomalies.append(item.copy())
                manifest.append(item)
                if (index + 1) % 1000 == 0:
                    print(kind, index + 1, '/', len(rows), 'elapsed_s', round(time.monotonic() - start), flush=True)
    # Union identical rasters and exact dHash candidate buckets before splitting.
    parent = list(range(len(labels)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for indices in list(exact.values()) + list(near.values()):
        first = find(indices[0])
        for index in indices[1:]: parent[find(index)] = first
    groups = defaultdict(list)
    for i in range(len(labels)): groups[str(find(i))].append(i)
    split = grouped_split(labels, groups, args.seed)
    index_by_id = {v: i for i, v in enumerate(ids)}
    for item in manifest:
        if item['kind'] == 'train' and item['source_id'] in index_by_id:
            i = index_by_id[item['source_id']]
            item['split'] = ['train', 'validation', 'test'][int(split[i])]
            item['duplicate_group'] = str(find(i))
        elif item['kind'] == 'unlabeled': item['split'] = 'unlabeled_prediction'
    label_array = np.asarray(labels, dtype=np.float32)
    cache_path = args.output / f'cache-{args.size}.npz'
    np.savez(cache_path, images=np.stack(tensors), targets=label_array[:, 0], sex=label_array[:, 1],
             ids=np.asarray(ids), split=split)
    manifest_file = args.output / 'manifest.jsonl'
    manifest_file.write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in manifest) + '\n')
    counts = {name: int((split == code).sum()) for code, name in enumerate(('train', 'validation', 'test'))}
    cross = [v for v in fingerprints.values() if len({kind for kind, _ in v}) > 1]
    audit = {
        'created_at': now(), 'source': 'kmader/rsna-bone-age', 'dataset_version': 2,
        'source_url': 'https://www.kaggle.com/datasets/kmader/rsna-bone-age',
        'license': 'Other (specified in description); research use; verify terms for other uses',
        'training_rows': len(train), 'unlabeled_rows': len(test), 'png_entries': len(png_entries),
        'matched_training': match_count['train'], 'matched_unlabeled': match_count['unlabeled'],
        'anomalies_count': len(anomalies), 'anomalies': anomalies[:100],
        'exact_duplicate_groups': sum(len(v) > 1 for v in exact.values()),
        'dhash_candidate_groups': sum(len(v) > 1 for v in near.values()),
        'cross_labeled_unlabeled_duplicate_groups': cross,
        'patient_isolation': 'UNVERIFIED: no patient identifiers or exam dates supplied',
        'near_duplicate_limit': 'Exact 256-bit dHash buckets grouped; not exhaustive visual similarity search',
        'label_unit': 'months (RSNA research convention); verify reference labeling standard before clinical use',
        'csv_checksums': csv_hashes, 'splits': counts, 'split_seed': args.seed,
        'grouping': 'decoded raster equality and exact dHash candidates; patient grouping unavailable',
        'image_size': args.size, 'preprocess_version': PREPROCESS_VERSION,
        'manifest_sha256': hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        'cache_sha256': hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        'modes': dict(modes), 'distinct_dimensions': len(dimensions),
        'elapsed_seconds': round(time.monotonic() - start),
    }
    (args.output / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k not in {'csv_checksums','anomalies'}}, ensure_ascii=False, indent=2), flush=True)
    print('Cache:', cache_path, flush=True)


if __name__ == '__main__': main()
