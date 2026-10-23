"""CPU diagnostic: frozen pretrained ResNet50 features, trained sex/regression head.

This is a separate head-only baseline, not GPU full fine-tuning. Only the fixed
train and validation partitions are extracted or used for selection. Features
are deterministic: no augmentation, eval-mode backbone, frozen BatchNorm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.network import build_model, denormalize_target, normalize_for_model, preprocess_version_for
from app.training import (atomic_json, atomic_torch_save, compact_reference, environment_info,
                          export_model_from_predictions, now, restore_rng, rng_state,
                          seed_everything, sha256_file, write_history)
from app.training_data import BoneAgeDataset, load_data_reference

ARCHITECTURE = 'resnet50-sex-v1'
WEIGHT_NAME = 'resnet50-11ad3fa6.pth'
WEIGHT_SHA256 = '11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca'
PROTOCOL = 'frozen_backbone_head_only'
CHECKPOINT_KIND = 'boneage-frozen-head-checkpoint-v1'


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def find_weights(weights_dir):
    directory = Path(weights_dir) if weights_dir else Path(os.environ.get('TORCH_HOME', Path.home() / '.cache/torch'))
    choices = [directory / WEIGHT_NAME, directory / 'hub/checkpoints' / WEIGHT_NAME,
               directory / 'checkpoints' / WEIGHT_NAME]
    for path in choices:
        if path.is_file():
            if sha256_file(path) != WEIGHT_SHA256:
                raise ValueError(f'Official pretrained weight SHA256 mismatch: {path}')
            return path.resolve()
    raise FileNotFoundError(f'Place checksum-verified {WEIGHT_NAME} in --weights-dir or TORCH_HOME; this tool never downloads')


def initialize_model(weights):
    model = build_model(ARCHITECTURE, pretrained=False).cpu()
    state = torch.load(weights, map_location='cpu', weights_only=True)
    model.features.load_state_dict({key: value for key, value in state.items()
                                    if not key.startswith('fc.')}, strict=True)
    for parameter in model.features.parameters():
        parameter.requires_grad_(False)
    model.features.eval()
    return model


def extract_split(model, dataset, directory, config, lineage):
    """Commit completed batches; a resume never reuses uncommitted rows."""
    directory.mkdir(parents=True, exist_ok=True)
    feature_path, progress_path = directory / 'features.npy', directory / 'progress.json'
    identity = {'schema_version': 1, 'split': dataset.split, 'samples': len(dataset),
                'source_ids': dataset.ids, 'lineage': lineage,
                'image_size': config['image_size'], 'preprocess_version': preprocess_version_for(ARCHITECTURE),
                'feature_module': 'pool(features(image))', 'feature_width': 2048,
                'augmentation': False, 'backbone_eval': True, 'batchnorm_frozen': True}
    fingerprint = canonical_sha(identity)
    completed = 0
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding='utf-8'))
        if progress.get('fingerprint') != fingerprint:
            raise ValueError(f'Frozen feature identity mismatch: {dataset.split}')
        completed = progress['completed_samples']
        if type(completed) is not int or not 0 <= completed <= len(dataset):
            raise ValueError('Invalid frozen feature progress')
        features = np.load(feature_path, mmap_mode='r+', allow_pickle=False)
        if features.shape != (len(dataset), 2048) or features.dtype != np.float32:
            raise ValueError('Invalid frozen feature array shape/dtype')
        if progress.get('status') == 'completed':
            if completed != len(dataset) or sha256_file(feature_path) != progress['feature_sha256']:
                raise ValueError('Completed frozen feature cache checksum mismatch')
            return {**identity, 'fingerprint': fingerprint, 'feature_sha256': progress['feature_sha256']}
    else:
        if feature_path.exists():
            raise ValueError(f'Untracked feature file exists: {feature_path}')
        features = np.lib.format.open_memmap(feature_path, mode='w+', dtype=np.float32,
                                             shape=(len(dataset), 2048))
        atomic_json(directory / 'identity.json', identity)
    started = time.monotonic()
    progress = {'fingerprint': fingerprint, 'completed_samples': completed,
                'status': 'extracting', 'updated_at': now()}
    atomic_json(progress_path, progress)
    # Subset uses only this dataset's already fixed train OR validation rows.
    remainder = torch.utils.data.Subset(dataset, range(completed, len(dataset)))
    loader = DataLoader(remainder, batch_size=config['feature_batch_size'], shuffle=False,
                        num_workers=0, pin_memory=False)
    model.eval()
    with torch.inference_mode():
        for images, _sex, _target, _ids in loader:
            normalized = normalize_for_model(images, preprocess_version_for(ARCHITECTURE))
            pooled = model.pool(model.features(normalized)).flatten(1)
            if not bool(torch.isfinite(pooled).all()):
                raise FloatingPointError('Non-finite pretrained features')
            count = len(images)
            features[completed:completed + count] = pooled.cpu().numpy()
            features.flush()
            completed += count
            progress.update(completed_samples=completed, updated_at=now(),
                            elapsed_seconds=round(time.monotonic() - started, 3))
            atomic_json(progress_path, progress)
            if completed % (config['feature_batch_size'] * 10) == 0 or completed == len(dataset):
                print(json.dumps({'stage': 'feature_extraction', 'split': dataset.split,
                                  'completed_samples': completed, 'samples': len(dataset),
                                  'elapsed_seconds': progress['elapsed_seconds']}), flush=True)
    features.flush()
    del features
    digest = sha256_file(feature_path)
    progress.update(status='completed', feature_sha256=digest, completed_samples=len(dataset), updated_at=now())
    atomic_json(progress_path, progress)
    return {**identity, 'fingerprint': fingerprint, 'feature_sha256': digest}


class FrozenFeatures(Dataset):
    def __init__(self, path, dataset):
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.float32 or array.shape != (len(dataset), 2048) or not np.isfinite(array).all():
            raise ValueError('Malformed/non-finite frozen features')
        self.features = torch.from_numpy(array).unsqueeze(-1).unsqueeze(-1)
        self.ids = dataset.ids
        # The source Dataset provides only the chosen partition's records.
        self.sex = torch.tensor([[1. if record['sex'] == 'M' else 0.] for record in dataset.records])
        self.targets = torch.tensor([float(record['boneage']) for record in dataset.records])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.features[index], self.sex[index], self.targets[index], self.ids[index]


@torch.inference_mode()
def predict_features(model, loader):
    model.eval()
    predictions, targets, sexes, ids = [], [], [], []
    for features, sex, target, source_ids in loader:
        output = denormalize_target(model.predict_from_features(features, sex))
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError('Non-finite head prediction')
        predictions.extend(output.tolist())
        targets.extend(target.tolist())
        sexes.extend(sex[:, 0].tolist())
        ids.extend(str(value) for value in source_ids)
    return np.asarray(predictions), np.asarray(targets), np.asarray(sexes), ids


def train_head(model, loaders, config, run_dir, reference, feature_reference, resume):
    parameters = list(model.sex_embedding.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config['head_lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    generator = loaders['generator']
    history, completed, best_epoch, best_mae, bad_epochs = [], 0, 0, float('inf'), 0
    checkpoint_dir = run_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    latest = checkpoint_dir / 'latest.pt'
    if resume and latest.is_file():
        state = torch.load(latest, map_location='cpu', weights_only=True)
        if (state.get('kind') != CHECKPOINT_KIND or state['config'] != config
                or state['data_reference'] != compact_reference(reference)
                or state['feature_reference'] != feature_reference):
            raise ValueError('Frozen head resume identity/config mismatch')
        model.sex_embedding.load_state_dict(state['sex_embedding'], strict=True)
        model.head.load_state_dict(state['head'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        completed, best_epoch, best_mae = state['epoch'], state['best_epoch'], state['best_mae']
        bad_epochs, history = state['bad_epochs'], state['history']
        restore_rng(state['rng'], generator)
    started = time.monotonic()
    patience = config['early_stopping_patience']
    for epoch in range(completed, config['epochs']):
        if bad_epochs >= patience:
            break
        model.train()
        model.features.eval()
        total, seen = 0., 0
        for features, sex, target, _ids in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            output = model.predict_from_features(features, sex)
            loss = nn.functional.smooth_l1_loss(output, (target - 120.) / 60., beta=.5)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Non-finite frozen head loss')
            loss.backward()
            norm = nn.utils.clip_grad_norm_(parameters, config['gradient_clip_norm'])
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError('Non-finite frozen head gradients')
            optimizer.step()
            total += float(loss.detach()) * len(features)
            seen += len(features)
        train = predict_features(model, loaders['train_eval'])
        validation = predict_features(model, loaders['validation'])
        train_mae = float(np.abs(train[0] - train[1]).mean())
        validation_mae = float(np.abs(validation[0] - validation[1]).mean())
        significant = validation_mae < best_mae - config['early_stopping_min_delta']
        improved = validation_mae < best_mae
        bad_epochs = 0 if significant else bad_epochs + 1
        if improved:
            best_epoch, best_mae = epoch + 1, validation_mae
        learning_rate = optimizer.param_groups[0]['lr']
        scheduler.step()
        completed = epoch + 1
        record = {'epoch': completed, 'phase': PROTOCOL, 'train_loss': total / seen,
                  'train_eval_mae_months': train_mae, 'validation_mae_months': validation_mae,
                  'learning_rates': {'head': learning_rate}, 'best_epoch': best_epoch,
                  'bad_epochs': bad_epochs, 'elapsed_seconds': round(time.monotonic() - started, 3), 'device': 'cpu'}
        history.append(record)
        state = {'kind': CHECKPOINT_KIND, 'config': config, 'data_reference': compact_reference(reference),
                 'feature_reference': feature_reference, 'sex_embedding': model.sex_embedding.state_dict(),
                 'head': model.head.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'rng': rng_state(generator), 'epoch': completed,
                 'best_epoch': best_epoch, 'best_mae': best_mae, 'bad_epochs': bad_epochs, 'history': history}
        if improved:
            atomic_torch_save(checkpoint_dir / 'best.pt', state)
        atomic_torch_save(latest, state)
        write_history(run_dir / 'history.jsonl', history)
        atomic_json(run_dir / 'status.json', {'status': 'training_head', 'training_protocol': PROTOCOL,
                    'run_dir': str(run_dir), 'completed_epochs': completed, 'best_epoch': best_epoch,
                    'best_validation_mae_months': best_mae, 'test_evaluated': False, 'updated_at': now()})
        print(json.dumps(record), flush=True)
    best = torch.load(checkpoint_dir / 'best.pt', map_location='cpu', weights_only=True)
    model.sex_embedding.load_state_dict(best['sex_embedding'], strict=True)
    model.head.load_state_dict(best['head'], strict=True)
    status = 'early_stopped' if completed < config['epochs'] else 'completed'
    report = export_model_from_predictions(model, run_dir, config, reference,
                predict_features(model, loaders['train_eval']), predict_features(model, loaders['validation']),
                best_epoch, best_mae, completed, status)
    atomic_json(run_dir / 'status.json', {'status': status, 'training_protocol': PROTOCOL,
                'run_dir': str(run_dir), 'completed_epochs': completed, 'best_epoch': best_epoch,
                'best_validation_mae_months': best_mae, 'test_evaluated': False,
                'candidate': str(run_dir / 'export/candidate.json'), 'finished_at': now()})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=128, help='Regression head minibatch size')
    parser.add_argument('--feature-batch-size', type=int, default=16)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--image-size', type=int, choices=(224, 384, 512), default=224)
    parser.add_argument('--weights-dir', type=Path)
    parser.add_argument('--resume', action='store_true', help='Resume committed extraction batches and head epochs')
    parser.add_argument('--extract-only', action='store_true', help='Stop once train/validation features are cached')
    args = parser.parse_args(argv)
    for key in ('epochs', 'batch_size', 'feature_batch_size', 'threads'):
        if getattr(args, key) < 1:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    run_dir = args.run_dir.expanduser().resolve()
    config = {'architecture': ARCHITECTURE, 'device': 'cpu', 'image_size': args.image_size,
              'epochs': args.epochs, 'batch_size': args.batch_size, 'feature_batch_size': args.feature_batch_size,
              'threads': args.threads, 'seed': args.seed, 'head_lr': 3e-4, 'weight_decay': 5e-4,
              'gradient_clip_norm': 5., 'early_stopping_patience': 10, 'early_stopping_min_delta': .01,
              'augmentation': False, 'bn_strategy': 'freeze_running_stats', 'amp': False,
              'training_protocol': PROTOCOL, 'scheduler': 'cosine_annealing', 'minimum_lr': 1e-6,
              'pretrained_weights': WEIGHT_NAME, 'pretrained_weights_sha256': WEIGHT_SHA256,
              'feature_module': 'pool(features(image))', 'feature_width': 2048,
              'limitations': ['Frozen pretrained backbone with trained head only; not full fine-tuning',
                              'No image augmentation; fixed train/validation selection only']}
    if args.resume:
        if json.loads((run_dir / 'config.json').read_text(encoding='utf-8')) != config:
            raise ValueError('Resume configuration differs from the original frozen baseline')
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(run_dir / 'config.json', config)
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    atomic_json(run_dir / 'environment.json', {**environment_info('cpu'), 'resolved_device': 'cpu',
                'amp_enabled': False, 'training_protocol': PROTOCOL})
    atomic_json(run_dir / 'status.json', {'status': 'initializing', 'training_protocol': PROTOCOL,
                'run_dir': str(run_dir), 'test_evaluated': False, 'started_at': now()})
    try:
        reference = load_data_reference(args.data)
        compact = compact_reference(reference)
        reference_path = run_dir / 'data-reference.json'
        if args.resume and reference_path.exists() and json.loads(reference_path.read_text(encoding='utf-8')) != compact:
            raise ValueError('Resume audited data reference differs')
        atomic_json(reference_path, compact)
        weights = find_weights(args.weights_dir)
        model = initialize_model(weights)
        datasets = {split: BoneAgeDataset(args.data, split, args.image_size, reference=reference)
                    for split in ('train', 'validation')}
        if any(len(dataset) == 0 for dataset in datasets.values()):
            raise ValueError('Nonempty train and validation partitions are required')
        lineage = {'data_reference': compact, 'pretrained_weights_sha256': WEIGHT_SHA256}
        feature_reference = {}
        for split, dataset in datasets.items():
            atomic_json(run_dir / 'status.json', {'status': 'extracting_features', 'split': split,
                        'training_protocol': PROTOCOL, 'run_dir': str(run_dir), 'test_evaluated': False,
                        'updated_at': now()})
            feature_reference[split] = extract_split(model, dataset, run_dir / 'features' / split, config, lineage)
        atomic_json(run_dir / 'feature-reference.json', feature_reference)
        if args.extract_only:
            atomic_json(run_dir / 'status.json', {'status': 'features_completed', 'training_protocol': PROTOCOL,
                        'run_dir': str(run_dir), 'test_evaluated': False, 'finished_at': now()})
            return
        frozen = {split: FrozenFeatures(run_dir / 'features' / split / 'features.npy', dataset)
                  for split, dataset in datasets.items()}
        # DataLoader iterator setup also consumes RNG. Cached/resumed extraction
        # must leave a fresh head-training run at the same random boundary.
        seed_everything(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
        loaders = {'train': DataLoader(frozen['train'], batch_size=args.batch_size, shuffle=True, generator=generator),
                   'train_eval': DataLoader(frozen['train'], batch_size=args.batch_size, shuffle=False),
                   'validation': DataLoader(frozen['validation'], batch_size=args.batch_size, shuffle=False),
                   'generator': generator}
        report = train_head(model, loaders, config, run_dir, reference, feature_reference, args.resume)
        print(json.dumps({'status': 'finished', 'candidate': str(run_dir / 'export'),
                          'best_validation_mae_months': report['best_validation_mae_months'],
                          'training_protocol': PROTOCOL, 'test_evaluated': False}), flush=True)
    except BaseException as exc:
        atomic_json(run_dir / 'status.json', {'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                    'training_protocol': PROTOCOL, 'error': f'{type(exc).__name__}: {exc}',
                    'recovery': 'Repeat the identical command with --resume', 'test_evaluated': False,
                    'finished_at': now()})
        raise


if __name__ == '__main__':
    main()
