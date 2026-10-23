"""CPU experiment: adapt the final ResNet residual block using fixed FP32 prefixes.

Only train/validation images are cached. The first two layer4 blocks and all
earlier image layers remain frozen; this is not full-backbone or GPU training.
The exported network is the unchanged deployment architecture.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from app.network import build_model, denormalize_target, normalize_for_model
from app.predictor import Predictor
from app.training import (atomic_json, atomic_torch_save, compact_reference, cpu_tree,
                          environment_info, export_model_from_predictions, now,
                          restore_rng, rng_state, seed_everything, sha256_file, write_history)
from app.training_data import BoneAgeDataset, load_data_reference

FEATURE_SHAPE = (2048, 7, 7)
CHECKPOINT_KIND = 'boneage-resnet-tail-checkpoint-v1'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def initial_model(directory, metadata_filename='candidate.json'):
    directory = Path(directory).resolve()
    metadata = Predictor(directory, metadata_filename).metadata()
    if metadata is None or metadata['architecture'] != 'resnet50-sex-v1' or metadata['image_size'] != 224:
        raise ValueError('A valid, complete ResNet50-sex 224 candidate is required')
    raw = (directory / metadata['checkpoint']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata['checkpoint_sha256']:
        raise ValueError('Initialization checkpoint SHA256 mismatch')
    model = build_model(metadata['architecture'], pretrained=False)
    model.load_state_dict(torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True), strict=True)
    model.to(memory_format=torch.channels_last)
    model.eval()
    lineage = {'model_id': metadata['model_id'], 'checkpoint_sha256': metadata['checkpoint_sha256'],
               'metadata_sha256': sha256_file(directory / metadata_filename),
               'architecture': metadata['architecture'], 'image_size': 224,
               'preprocess_version': metadata['preprocess_version']}
    return model, lineage


def prefix_forward(model, images):
    return model.features.layer4[:2](model.features[:7](images))


def tail_forward(model, features, sex):
    return model.predict_from_features(model.features.layer4[2](features), sex)


def cache_identity(dataset, reference, lineage):
    if dataset.split not in ('train', 'validation'):
        raise ValueError('Tail training must never cache the holdout')
    return {'schema': 'boneage-resnet-prefix-fp32-v1', 'split': dataset.split,
            'source_ids': dataset.ids, 'samples': len(dataset), 'shape': [len(dataset), *FEATURE_SHAPE],
            'dtype': 'float32', 'precision': 'fp32', 'memory_format': 'channels_last',
            'module': 'features through layer4.1, before layer4.2',
            'augmentation': False, 'batchnorm': 'eval',
            'initialization': lineage, 'data_reference': compact_reference(reference)}


def prepare_split(model, dataset, directory, reference, lineage, batch_size=16):
    """Resume at committed batches; completed caches require a full SHA check."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identity = cache_identity(dataset, reference, lineage)
    identity_sha = fingerprint(identity)
    path, progress_path = directory / 'features.npy', directory / 'progress.json'
    completed = 0
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if progress.get('identity_sha256') != identity_sha:
            raise ValueError('Prefix cache initialization/data identity mismatch')
        completed = progress['completed_samples']
        if type(completed) is not int or not 0 <= completed <= len(dataset):
            raise ValueError('Invalid prefix extraction progress')
        features = np.load(path, mmap_mode='r+', allow_pickle=False)
        if features.dtype != np.float32 or list(features.shape) != identity['shape']:
            raise ValueError('Prefix shape/dtype mismatch')
        if progress['status'] == 'completed':
            if completed != len(dataset) or sha256_file(path) != progress['feature_sha256']:
                raise ValueError('Completed prefix cache checksum mismatch')
            return {**identity, 'identity_sha256': identity_sha, 'feature_sha256': progress['feature_sha256']}
    else:
        if path.exists():
            raise ValueError('Untracked prefix features already exist')
        features = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                            shape=tuple(identity['shape']))
        atomic_json(directory / 'identity.json', identity)
    started = time.monotonic()
    progress = {'identity_sha256': identity_sha, 'completed_samples': completed,
                'status': 'extracting', 'updated_at': now()}
    atomic_json(progress_path, progress)
    loader = DataLoader(Subset(dataset, range(completed, len(dataset))), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    model.eval()
    with torch.inference_mode():
        for images, _sex, _target, _ids in loader:
            inputs = normalize_for_model(images, lineage['preprocess_version']).contiguous(memory_format=torch.channels_last)
            values = prefix_forward(model, inputs)
            if not bool(torch.isfinite(values).all()) or tuple(values.shape[1:]) != FEATURE_SHAPE:
                raise FloatingPointError('Non-finite or malformed prefix features')
            features[completed:completed + len(images)] = values.numpy()
            features.flush()
            completed += len(images)
            progress.update(completed_samples=completed, updated_at=now(),
                            elapsed_seconds=round(time.monotonic() - started, 3))
            atomic_json(progress_path, progress)
            if completed % (batch_size * 10) == 0 or completed == len(dataset):
                print(json.dumps({'stage': 'prefix_cache', 'split': dataset.split,
                                  'completed_samples': completed, 'samples': len(dataset),
                                  'elapsed_seconds': progress['elapsed_seconds']}), flush=True)
    features.flush()
    del features
    digest = sha256_file(path)
    progress.update(status='completed', feature_sha256=digest, updated_at=now())
    atomic_json(progress_path, progress)
    return {**identity, 'identity_sha256': identity_sha, 'feature_sha256': digest}


def prepare_cache(initial, metadata_filename, data, cache, batch_size=16, threads=4):
    torch.set_num_threads(threads)
    model, lineage = initial_model(initial, metadata_filename)
    reference = load_data_reference(data)
    verify_reference(lineage, initial, metadata_filename, reference)
    output = {}
    for split in ('train', 'validation'):
        dataset = BoneAgeDataset(data, split, 224, reference)
        output[split] = prepare_split(model, dataset, Path(cache) / split, reference, lineage, batch_size)
    atomic_json(Path(cache) / 'reference.json', output)
    return output


def verify_reference(lineage, initial, metadata_filename, reference):
    metadata = json.loads((Path(initial) / metadata_filename).read_text())
    for name in ('manifest_sha256', 'split_sha256', 'audit_sha256', 'source_sha256'):
        if metadata['data_reference'][name] != reference[name]:
            raise ValueError(f'Initialization and training data differ: {name}')


class PrefixDataset(Dataset):
    def __init__(self, directory, dataset, reference, lineage):
        directory = Path(directory)
        self.path = directory / 'features.npy'
        identity = cache_identity(dataset, reference, lineage)
        progress = json.loads((directory / 'progress.json').read_text())
        if (progress.get('status') != 'completed' or progress.get('completed_samples') != len(dataset)
                or progress.get('identity_sha256') != fingerprint(identity)
                or sha256_file(self.path) != progress.get('feature_sha256')):
            raise ValueError('Incomplete, changed or incompatible prefix cache')
        self.array = np.load(self.path, mmap_mode='r', allow_pickle=False)
        if self.array.dtype != np.float32 or list(self.array.shape) != identity['shape']:
            raise ValueError('Invalid prefix cache shape/dtype')
        self.records, self.ids = dataset.records, dataset.ids
        self.proof = {'identity_sha256': fingerprint(identity), 'feature_sha256': progress['feature_sha256']}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        row = self.records[index]
        return (torch.from_numpy(self.array[index].copy()), torch.tensor([1. if row['sex'] == 'M' else 0.]),
                torch.tensor(float(row['boneage'])), self.ids[index])


def set_phase(model, adapt):
    model.train()
    for parameter in model.features.parameters():
        parameter.requires_grad_(False)
    if adapt:
        for parameter in model.features.layer4[2].parameters():
            parameter.requires_grad_(True)
    # Even trainable BN affine parameters use frozen running statistics.
    model.features.eval()
    return 'last_residual_block' if adapt else 'frozen_backbone_head_only'


@torch.inference_mode()
def predictions(model, loader):
    model.eval()
    values, targets, sexes, ids = [], [], [], []
    for features, sex, target, identifiers in loader:
        features = features.contiguous(memory_format=torch.channels_last)
        output = denormalize_target(tail_forward(model, features, sex))
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError('Non-finite tail prediction')
        values.extend(output.tolist()); targets.extend(target.tolist())
        sexes.extend(sex[:, 0].tolist()); ids.extend(identifiers)
    return np.asarray(values), np.asarray(targets), np.asarray(sexes), ids


def train_epoch(model, loader, optimizer, config, epoch):
    adapt = config['adapt_tail'] and epoch >= config['freeze_backbone_epochs']
    phase = set_phase(model, adapt)
    total, seen = 0., 0
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    for features, sex, target, _ids in loader:
        optimizer.zero_grad(set_to_none=True)
        features = features.contiguous(memory_format=torch.channels_last)
        output = tail_forward(model, features, sex)
        target = (target - 120.) / 60.
        loss = (nn.functional.l1_loss(output, target) if config['loss'] == 'l1' else
                nn.functional.smooth_l1_loss(output, target, beta=config['loss_beta']))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Non-finite tail loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip_norm'], error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach()) * len(target); seen += len(target)
        if seen % (config['batch_size'] * 200) == 0:
            print(json.dumps({'stage': 'tail_epoch', 'epoch': epoch + 1, 'phase': phase,
                              'completed_samples': seen, 'samples': len(loader.dataset)}), flush=True)
    if seen != len(loader.dataset):
        raise RuntimeError('Tail training omitted samples')
    return total / seen, phase


def tail_state(model):
    return cpu_tree({k: v for k, v in model.state_dict().items()
                     if k.startswith(('features.layer4.2.', 'head.', 'sex_embedding.'))})


def restore_tail(model, state):
    if set(state) != set(tail_state(model)):
        raise ValueError('Tail checkpoint has missing or unexpected keys')
    merged = model.state_dict()
    merged.update(state)
    model.load_state_dict(merged, strict=True)


def changed_parameters(model, initial, metadata_filename):
    """Prove what changed in the selected exported checkpoint, not just the loop."""
    original, _lineage = initial_model(initial, metadata_filename)
    before = original.state_dict()
    changed = [key for key, value in model.state_dict().items() if not torch.equal(value.cpu(), before[key])]
    outside = [key for key in changed if not key.startswith(('features.layer4.2.', 'head.', 'sex_embedding.'))]
    statistics = [key for key in changed if key.endswith(('running_mean', 'running_var', 'num_batches_tracked'))]
    if outside or statistics:
        raise RuntimeError('Selected model changed a frozen prefix or BatchNorm statistics')
    return {'changed_state_keys': changed, 'frozen_prefix_unchanged': True,
            'batchnorm_running_statistics_unchanged': True,
            'image_block_actually_changed': any(k.startswith('features.layer4.2.') for k in changed),
            'sex_or_regression_head_changed': any(k.startswith(('sex_embedding.', 'head.')) for k in changed)}


def run_experiment(initial, metadata_filename, data, cache, run_dir, config, resume=False):
    validate_config(config)
    run_dir, cache = Path(run_dir).resolve(), Path(cache).resolve()
    if resume:
        if json.loads((run_dir / 'config.json').read_text()) != config:
            raise ValueError('Tail resume configuration mismatch')
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(run_dir / 'config.json', config)
    status = {'status': 'initializing', 'started_at': now(), 'test_evaluated': False}
    atomic_json(run_dir / 'status.json', status)
    try:
        torch.set_num_threads(config['threads'])
        seed_everything(config['seed'])
        model, lineage = initial_model(initial, metadata_filename)
        reference = load_data_reference(data)
        verify_reference(lineage, initial, metadata_filename, reference)
        compact = compact_reference(reference)
        datasets = {split: PrefixDataset(cache / split, BoneAgeDataset(data, split, 224, reference), reference, lineage)
                    for split in ('train', 'validation')}
        proof = {split: d.proof for split, d in datasets.items()}
        atomic_json(run_dir / 'initialization.json', {'kind': 'complete_local_state_dict', **lineage})
        atomic_json(run_dir / 'data-reference.json', compact)
        atomic_json(run_dir / 'feature-reference.json', proof)
        code_proof = {name: sha256_file(Path(__file__).resolve().parents[1] / name) for name in
                      ('app/tail_training.py', 'app/network.py', 'app/imaging.py', 'app/training_data.py', 'app/training.py')}
        atomic_json(run_dir / 'code-sha256.json', code_proof)
        atomic_json(run_dir / 'environment.json', {**environment_info('cpu'), 'resolved_device': 'cpu',
            'precision': 'fp32', 'memory_format': 'channels_last', 'protocol': config['training_protocol']})
        seed_everything(config['seed'])
        generator = torch.Generator().manual_seed(config['seed'])
        loaders = {'train': DataLoader(datasets['train'], batch_size=config['batch_size'], shuffle=True, generator=generator),
                   'train_eval': DataLoader(datasets['train'], batch_size=config['batch_size'], shuffle=False),
                   'validation': DataLoader(datasets['validation'], batch_size=config['batch_size'], shuffle=False)}
        optimizer = torch.optim.AdamW([
            {'params': model.features.layer4[2].parameters(), 'lr': config['backbone_lr'], 'name': 'last_block'},
            {'params': list(model.head.parameters()) + list(model.sex_embedding.parameters()),
             'lr': config['head_lr'], 'name': 'head'}], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['minimum_lr'])
        checkpoints = run_dir / 'checkpoints'
        checkpoints.mkdir(exist_ok=True)
        completed, best_epoch, bad_epochs = 0, 0, 0
        history = []
        if resume:
            saved = torch.load(checkpoints / 'latest.pt', weights_only=True, map_location='cpu')
            if (saved.get('kind') != CHECKPOINT_KIND or saved['config'] != config
                    or saved['data_reference'] != compact or saved['features'] != proof or saved['initialization'] != lineage):
                raise ValueError('Tail resume data/initialization/configuration mismatch')
            if saved.get('code_sha256') != code_proof:
                raise ValueError('Tail resume training code differs from its checkpoint')
            restore_tail(model, saved['model'])
            optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
            restore_rng(saved['rng'], generator)
            completed, best_epoch, bad_epochs = saved['epoch'], saved['best_epoch'], saved['bad_epochs']
            best_mae, stopping_best, best = saved['best_mae'], saved['stopping_best'], saved['best_model']
            history = saved['history']
        else:
            baseline = predictions(model, loaders['validation'])
            best_mae = float(np.abs(baseline[0] - baseline[1]).mean())
            stopping_best = best_mae
            best = tail_state(model)
            atomic_json(run_dir / 'initial-validation.json', {'mae_months': best_mae, 'n': len(baseline[0]),
                        'checkpoint_sha256': lineage['checkpoint_sha256'], 'cached_prefix': True, 'epoch': 0})
            # Evaluation iterator creation consumes RNG; reset consistently for both arms.
            seed_everything(config['seed']); generator.manual_seed(config['seed'])
        def payload():
            return {'kind': CHECKPOINT_KIND, 'config': config, 'data_reference': compact,
                'features': proof, 'initialization': lineage, 'code_sha256': code_proof,
                'model': tail_state(model), 'best_model': best,
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'rng': rng_state(generator), 'epoch': completed, 'best_epoch': best_epoch,
                'best_mae': best_mae, 'stopping_best': stopping_best, 'bad_epochs': bad_epochs, 'history': history}
        if not resume:
            atomic_torch_save(checkpoints / 'latest.pt', payload())
        start = time.monotonic()
        for epoch in range(completed, config['epochs']):
            if bad_epochs >= config['early_stopping_patience']:
                break
            loss, phase = train_epoch(model, loaders['train'], optimizer, config, epoch)
            train = predictions(model, loaders['train_eval']); validation = predictions(model, loaders['validation'])
            train_mae = float(np.abs(train[0] - train[1]).mean())
            val_mae = float(np.abs(validation[0] - validation[1]).mean())
            significant = val_mae < stopping_best - config['early_stopping_min_delta']
            bad_epochs = 0 if significant else bad_epochs + 1
            if significant:
                stopping_best = val_mae
            if val_mae < best_mae:
                best_mae, best_epoch, best = val_mae, epoch + 1, tail_state(model)
            learning_rates = {g['name']: g['lr'] for g in optimizer.param_groups}
            scheduler.step(); completed = epoch + 1
            record = {'epoch': completed, 'phase': phase, 'train_loss': loss,
                'train_eval_mae_months': train_mae, 'validation_mae_months': val_mae,
                'best_epoch': best_epoch, 'bad_epochs': bad_epochs, 'learning_rates': learning_rates,
                'elapsed_seconds': round(time.monotonic() - start, 3), 'device': 'cpu'}
            history.append(record)
            atomic_torch_save(checkpoints / 'latest.pt', payload())
            write_history(run_dir / 'history.jsonl', history)
            status.update(status='training', completed_epochs=completed, best_epoch=best_epoch,
                          best_validation_mae_months=best_mae)
            atomic_json(run_dir / 'status.json', status)
            print(json.dumps(record), flush=True)
        restore_tail(model, best)
        parameter_proof = changed_parameters(model, initial, metadata_filename)
        if not config['adapt_tail'] and parameter_proof['image_block_actually_changed']:
            raise RuntimeError('Frozen control changed image features')
        atomic_json(run_dir / 'parameter-changes.json', parameter_proof)
        final = 'completed' if completed == config['epochs'] else 'early_stopped'
        report = export_model_from_predictions(model, run_dir, config, reference,
            predictions(model, loaders['train_eval']), predictions(model, loaders['validation']),
            best_epoch, best_mae, completed, final)
        status.update(status=final, finished_at=now(), candidate=str(run_dir / 'export'),
                      best_epoch=best_epoch, best_validation_mae_months=best_mae, completed_epochs=completed)
        atomic_json(run_dir / 'status.json', status)
        return report
    except BaseException as exc:
        status.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      error=f'{type(exc).__name__}: {exc}', finished_at=now())
        atomic_json(run_dir / 'status.json', status)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'train'))
    parser.add_argument('--initial-model', type=Path, required=True)
    parser.add_argument('--metadata', default='candidate.json')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.threads < 1:
        parser.error('batch-size and threads must be positive')
    if args.mode == 'prepare':
        prepare_cache(args.initial_model, args.metadata, args.data, args.cache, args.batch_size, args.threads)
    else:
        if args.config is None or args.run_dir is None:
            parser.error('train requires --config and --run-dir')
        config = json.loads(args.config.read_text())
        validate_config(config)
        run_experiment(args.initial_model, args.metadata, args.data, args.cache, args.run_dir, config, args.resume)


def validate_config(config):
    if config.get('architecture') != 'resnet50-sex-v1' or config.get('image_size') != 224:
        raise ValueError('Tail experiment requires ResNet50-sex 224')
    if config.get('device') != 'cpu' or config.get('amp') is not False or config.get('augmentation') is not False:
        raise ValueError('Cached tail experiments require CPU FP32 without image augmentation')
    if config.get('bn_strategy') != 'freeze_running_stats':
        raise ValueError('Cached tail experiments require frozen BatchNorm statistics')
    if type(config.get('adapt_tail')) is not bool:
        raise ValueError('adapt_tail must be a boolean')
    for key in ('epochs', 'batch_size', 'threads', 'early_stopping_patience'):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f'{key} must be positive')
    for key in ('seed', 'freeze_backbone_epochs'):
        if type(config.get(key)) is not int or config[key] < 0:
            raise ValueError(f'{key} must be nonnegative')
    for key in ('backbone_lr', 'head_lr', 'gradient_clip_norm', 'loss_beta'):
        if not np.isfinite(config.get(key, float('nan'))) or config[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if config.get('loss') not in ('smooth_l1', 'l1'):
        raise ValueError('Unknown loss')
    for key in ('minimum_lr', 'weight_decay', 'early_stopping_min_delta'):
        if not np.isfinite(config.get(key, float('nan'))) or config[key] < 0:
            raise ValueError(f'Invalid {key}')
    if config['minimum_lr'] > min(config['head_lr'], config['backbone_lr']):
        raise ValueError('minimum_lr exceeds group learning rate')
    expected = 'cpu_last_residual_block_finetuning' if config['adapt_tail'] else 'cpu_matched_frozen_control'
    if config.get('training_protocol') != expected:
        raise ValueError('Training protocol does not describe the actual trainable layers')
