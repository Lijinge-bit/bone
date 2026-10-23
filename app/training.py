"""Reproducible candidate training with an untouched final test split.

Resume uses completed epoch boundaries, restoring optimizer, cosine schedule,
scaler, random generators and early stopping. Checkpoints use safe tensor loads.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import uuid
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from app.metrics import metrics
from app.network import build_model, denormalize_target, normalize_for_model, preprocess_version_for
from app.training_data import BoneAgeDataset, load_data_reference
from app.config import Settings

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ('boneage-small-cnn-v1', 'resnet50-sex-v1')
ALIASES = {'cnn': ARCHITECTURES[0], 'small-cnn': ARCHITECTURES[0], 'resnet50': ARCHITECTURES[1]}
CHECKPOINT_SCHEMA = 2


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(config_path=None, overrides=None):
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    architecture = ALIASES.get(overrides.get('architecture'), overrides.get('architecture'))
    default = 'train_cnn60_cpu.json' if architecture == ARCHITECTURES[0] else 'train_resnet50_4060.json'
    config = json.loads((Path(config_path) if config_path else ROOT / 'configs' / default).read_text(encoding='utf-8'))
    config.update(overrides)
    config.setdefault('loss', 'smooth_l1')
    config.setdefault('loss_beta', .5)
    config['architecture'] = ALIASES.get(config['architecture'], config['architecture'])
    if config['architecture'] not in ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {config['architecture']}")
    if config['loss'] not in ('smooth_l1', 'l1'):
        raise ValueError('loss must be smooth_l1 or l1')
    if not math.isfinite(config['loss_beta']) or config['loss_beta'] <= 0:
        raise ValueError('loss_beta must be finite and positive')
    for name in ('pretrained', 'amp', 'augmentation'):
        if type(config[name]) is not bool:
            raise ValueError(f'{name} must be a JSON boolean')
    for name in ('amp', 'pretrained', 'augmentation'):
        if not isinstance(config[name], bool):
            raise ValueError(f'{name} must be a boolean')
    if config['image_size'] not in (96, 224, 384, 512):
        raise ValueError('image_size must be one of 96, 224, 384 or 512')
    for name in ('image_size', 'epochs', 'batch_size', 'threads', 'gradient_accumulation_steps'):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    for name in ('num_workers', 'freeze_backbone_epochs', 'early_stopping_patience', 'seed'):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 0:
            raise ValueError(f'{name} must be a non-negative integer')
    for name in ('backbone_lr', 'head_lr', 'gradient_clip_norm'):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f'{name} must be finite and positive')
    for name in ('weight_decay', 'minimum_lr', 'early_stopping_min_delta'):
        if not math.isfinite(config[name]) or config[name] < 0:
            raise ValueError(f'{name} must be finite and non-negative')
    if config['minimum_lr'] > min(config['head_lr'], config['backbone_lr']):
        raise ValueError('minimum_lr must not exceed either group learning rate')
    if config['bn_strategy'] not in ('freeze_running_stats', 'update_running_stats'):
        raise ValueError('Unknown bn_strategy')
    if config['architecture'] == ARCHITECTURES[0] and config['freeze_backbone_epochs']:
        raise ValueError('The baseline CNN trains the full model from epoch 1')
    if str(config['device']) == 'cpu' and config['amp']:
        raise ValueError('AMP requires CUDA; explicitly set amp=false for CPU')
    return config


def resolve_device(requested):
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ValueError('device must be cpu, cuda or cuda:<index>') from exc
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only explicit cpu and cuda devices are supported')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable. Install CUDA-enabled PyTorch '
                               'and an NVIDIA driver, or explicitly use --device cpu --no-amp.')
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise RuntimeError(f'CUDA device index {index} is unavailable')
        device = torch.device('cuda', index)
    return device


def environment_info(requested_device='cpu'):
    try:
        torchvision = importlib.metadata.version('torchvision')
    except importlib.metadata.PackageNotFoundError:
        torchvision = None
    available = torch.cuda.is_available()
    driver = None
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                    capture_output=True, text=True, timeout=10, check=False)
            if result.returncode == 0:
                driver = sorted(set(result.stdout.strip().splitlines()))
        except (OSError, subprocess.TimeoutExpired):
            pass
    devices = []
    if available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append({'index': index, 'name': properties.name, 'total_memory_bytes': properties.total_memory})
    return {'created_at': now(), 'python': platform.python_version(), 'python_executable': sys.executable,
            'platform': platform.platform(), 'torch': str(torch.__version__), 'torchvision': torchvision,
            'numpy': np.__version__, 'requested_device': str(requested_device), 'cuda_available': available,
            'nvidia_driver_versions': driver,
            'torch_cuda_build': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'cuda_devices': devices, 'threads': torch.get_num_threads()}


def validate_resume_config(saved, current):
    # Early candidate runs used this exact hard-coded loss; adding explicit
    # configuration preserves those checkpoints without allowing a loss change.
    saved = {'loss': 'smooth_l1', 'loss_beta': .5, **saved}
    current = {'loss': 'smooth_l1', 'loss_beta': .5, **current}
    if saved != current:
        different = sorted(key for key in set(saved) | set(current) if saved.get(key) != current.get(key))
        raise ValueError('Resume configuration mismatch: ' + ', '.join(different))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state(generator):
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])],
            'torch': torch.get_rng_state(), 'loader': generator.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'])
    generator.set_state(state['loader'])
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume CUDA random-generator topology differs')
        torch.cuda.set_rng_state_all(state['cuda'])


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def atomic_torch_save(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(cpu_tree(payload), stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_checkpoint(path):
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('schema_version') != CHECKPOINT_SCHEMA:
        raise ValueError('Unsupported checkpoint; resume requires checkpoints/latest.pt or best.pt')
    for key in ('config', 'data_reference', 'model', 'optimizer', 'scheduler', 'scaler', 'epoch',
                'best_epoch', 'best_mae', 'bad_epochs', 'rng', 'history', 'phase'):
        if key not in payload:
            raise ValueError(f'Training checkpoint is missing {key}')
    return payload


def phase_for(config, completed_epoch):
    return 'head_only' if completed_epoch < config['freeze_backbone_epochs'] else 'finetune'


def apply_phase(model, config, completed_epoch):
    phase = phase_for(config, completed_epoch)
    for parameter in model.features.parameters():
        parameter.requires_grad_(phase != 'head_only')
    if phase == 'head_only':
        model.features.eval()
    if config['bn_strategy'] == 'freeze_running_stats':
        for module in model.features.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    return phase


def build_optimizer(model, config):
    backbone = list(model.features.parameters())
    backbone_ids = {id(parameter) for parameter in backbone}
    head = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    if not backbone or not head:
        raise ValueError('Model must expose nonempty features and head parameter groups')
    return torch.optim.AdamW([
        {'params': backbone, 'lr': config['backbone_lr'], 'name': 'backbone'},
        {'params': head, 'lr': config['head_lr'], 'name': 'head'},
    ], weight_decay=config['weight_decay'])


def divide_gradients(parameters, sample_count):
    """Scale summed losses by actual samples, including a short accumulation tail."""
    if sample_count < 1:
        raise ValueError('Cannot normalize an empty accumulation window')
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(sample_count)


def normalized_inputs(images, config, augment=False):
    version = preprocess_version_for(config['architecture'])
    if augment and config['augmentation']:
        contrast = torch.empty((len(images), 1, 1, 1), device=images.device).uniform_(.9, 1.1)
        brightness = torch.empty((len(images), 1, 1, 1), device=images.device).uniform_(-.06, .06)
        if config['architecture'] == ARCHITECTURES[0]:
            return (normalize_for_model(images, version) * contrast + brightness).clamp(-1, 1)
        images = ((images.float() / 255 * contrast + brightness).clamp(0, 1) * 255).round().to(torch.uint8)
    return normalize_for_model(images, version)


def require_finite(value, message):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(message)


def train_epoch(model, loader, optimizer, scaler, device, config, completed_epoch):
    model.train()
    phase = apply_phase(model, config, completed_epoch)
    optimizer.zero_grad(set_to_none=True)
    total, seen, accumulated = 0.0, 0, 0
    amp_overflows, consecutive_overflows, optimizer_steps = 0, 0, 0
    amp = bool(config['amp'] and device.type == 'cuda')
    for index, (images, sexes, targets, _ids) in enumerate(loader):
        images, sexes, targets = images.to(device), sexes.to(device), targets.to(device)
        require_finite(sexes, 'Non-finite sex input')
        require_finite(targets, 'Non-finite target')
        with torch.autocast(device_type=device.type, enabled=amp):
            output = model(normalized_inputs(images, config, augment=True), sexes)
            require_finite(output, 'Non-finite training output')
            if config.get('loss', 'smooth_l1') == 'l1':
                loss = nn.functional.l1_loss(output, (targets - 120) / 60, reduction='sum')
            else:
                loss = nn.functional.smooth_l1_loss(output, (targets - 120) / 60,
                                                  beta=config.get('loss_beta', .5), reduction='sum')
        require_finite(loss, 'Non-finite training loss')
        scaler.scale(loss).backward()
        accumulated += len(images)
        seen += len(images)
        total += float(loss.detach())
        if (index + 1) % config['gradient_accumulation_steps'] == 0 or index + 1 == len(loader):
            scaler.unscale_(optimizer)
            divide_gradients(model.parameters(), accumulated)
            finite_gradients = torch.stack([torch.isfinite(parameter.grad).all()
                                           for parameter in model.parameters() if parameter.grad is not None]).all()
            if not bool(finite_gradients):
                if not amp:
                    raise FloatingPointError('Non-finite gradient; optimizer step cancelled')
                # A finite loss can still overflow in scaled FP16 backward. The
                # scaler skips this update and lowers its scale; never apply an
                # invalid gradient. Persistent failures indicate a broken run.
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                amp_overflows += 1
                consecutive_overflows += 1
                print(json.dumps({'event': 'amp_overflow', 'epoch': completed_epoch + 1,
                                  'batch': index + 1, 'optimizer_step_skipped': True,
                                  'previous_scale': old_scale, 'new_scale': scaler.get_scale()}), flush=True)
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                if consecutive_overflows >= 8:
                    raise FloatingPointError('AMP gradients remained non-finite after 8 scale reductions')
                continue
            consecutive_overflows = 0
            require_finite(nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm']), 'Non-finite gradient norm')
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
            if not bool(torch.stack([torch.isfinite(parameter).all() for parameter in model.parameters()]).all()):
                raise FloatingPointError('Non-finite parameter after optimizer update')
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
    if not seen:
        raise ValueError('Training split is empty')
    if not optimizer_steps:
        raise FloatingPointError('No finite optimizer update completed in this epoch')
    model._training_amp_overflow_steps = amp_overflows
    model._training_optimizer_steps = optimizer_steps
    return total / seen, phase


@torch.inference_mode()
def predict(model, loader, device, config):
    model.eval()
    predictions, labels, sexes, ids = [], [], [], []
    for images, sex, target, source_ids in loader:
        output = denormalize_target(model(normalized_inputs(images.to(device), config), sex.to(device)))
        require_finite(output, 'Non-finite evaluation prediction')
        predictions.extend(output.cpu().tolist())
        labels.extend(target.tolist())
        sexes.extend(sex[:, 0].tolist())
        ids.extend(str(value) for value in source_ids)
    if not predictions:
        raise ValueError('Evaluation split is empty')
    return np.asarray(predictions), np.asarray(labels), np.asarray(sexes), ids


def compact_reference(reference):
    audit = reference['audit']
    result = {key: reference[key] for key in ('manifest_sha256', 'split_sha256', 'audit_sha256', 'source_sha256') if key in reference}
    for key in ('cache_sha256', 'patient_isolation', 'grouping'):
        if key in audit:
            result[key] = audit[key]
    result['split_counts'] = audit.get('splits', {})
    if 'training_cache' in reference:
        result['training_cache'] = reference['training_cache']
    return result


def create_run_directory(output_root, architecture):
    root = Path(output_root or ROOT / 'var/runs')
    root.mkdir(parents=True, exist_ok=True)
    prefix = 'cnn' if architecture == ARCHITECTURES[0] else 'resnet50'
    for _ in range(10):
        path = root / (prefix + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
        try:
            path.mkdir()
            return path.resolve()
        except FileExistsError:
            continue
    raise RuntimeError('Could not allocate a unique run directory')


def write_history(path, history):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        for record in history:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
    os.replace(temporary, path)


def export_candidate(model, run_dir, config, reference, train_loader, validation_loader,
                     device, best_epoch, best_mae, completed_epochs, status, best_model_state=None):
    if best_model_state is None:
        best = load_checkpoint(run_dir / 'checkpoints/best.pt')
        if best['best_epoch'] != best_epoch or best['best_mae'] != best_mae:
            raise ValueError('Best checkpoint disagrees with the selected validation epoch')
        best_model_state = best['model']
    model.load_state_dict(best_model_state, strict=True)
    validation_predictions = predict(model, validation_loader, device, config)
    training_predictions = predict(model, train_loader, device, config)
    return export_model_from_predictions(model, run_dir, config, reference, training_predictions,
                                         validation_predictions, best_epoch, best_mae, completed_epochs, status)


def export_model_from_predictions(model, run_dir, config, reference, training_predictions,
                                  validation_predictions, best_epoch, best_mae, completed_epochs, status):
    """Export already selected weights and raw predictions without touching a holdout."""
    run_dir = Path(run_dir)
    predictions, labels, sexes, ids = validation_predictions
    validation_metrics = metrics(predictions, labels, sexes)
    tp, ty, ts, _ = training_predictions
    train_metrics = metrics(tp, ty, ts, bootstrap_samples=200)
    directory = run_dir / 'export'
    directory.mkdir(exist_ok=True)
    checkpoint = directory / 'model.pt'
    atomic_torch_save(checkpoint, model.state_dict())
    limitations = ['Research model; not clinically validated', 'Patient independence depends on the dataset audit',
                   'Final holdout has not been evaluated; validation was used for model selection'] + config.get('limitations', [])
    audit = reference['audit']
    report = {'created_at': now(), 'status': status, 'architecture': config['architecture'],
              'training_protocol': config.get('training_protocol', 'full_finetuning'),
              'training_samples': train_metrics['n'], 'validation_samples': validation_metrics['n'],
              'test_samples': None, 'epochs': completed_epochs, 'configured_epochs': config['epochs'],
              'best_epoch': best_epoch, 'best_validation_mae_months': best_mae,
              'training': train_metrics, 'validation': validation_metrics, 'test': None,
              'target_mae_months': 6, 'target_passed': False, 'patient_isolation': audit.get('patient_isolation'),
              'grouping': audit.get('grouping'), 'limitations': limitations, 'data_reference': compact_reference(reference),
              'selection_split': 'validation', 'metrics_use_unclipped_predictions': True,
              'preprocess_version': preprocess_version_for(config['architecture'])}
    validation_dir = run_dir / 'validation'
    validation_dir.mkdir(exist_ok=True)
    atomic_json(validation_dir / 'evaluation.json', report)
    with (validation_dir / 'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_image_id', 'reference_months', 'predicted_months', 'sex'])
        for source_id, truth, prediction, sex in zip(ids, labels, predictions, sexes):
            writer.writerow([source_id, float(truth), float(prediction), 'M' if sex else 'F'])
    resnet = config['architecture'] == ARCHITECTURES[1]
    metadata = {'schema_version': 2, 'model_id': run_dir.name, 'architecture': config['architecture'],
                'checkpoint': 'model.pt', 'checkpoint_sha256': sha256_file(checkpoint), 'image_size': config['image_size'],
                'preprocess_version': preprocess_version_for(config['architecture']), 'created_at': now(),
                'usage': 'research_only', 'input_channels': 3 if resnet else 1, 'sex_encoding': {'M': 1, 'F': 0},
                'target_normalization': {'offset': 120.0, 'scale': 60.0}, 'explanation_method': 'Grad-CAM',
                'gradcam_layer': 'features.layer4' if resnet else 'features',
                'explanation': 'Grad-CAM of predicted boneage; relative attention, not diagnostic segmentation',
                'data_reference': compact_reference(reference),
                'training_protocol': config.get('training_protocol', 'full_finetuning'),
                'evaluation': {key: report[key] for key in ('training_samples', 'validation_samples', 'test_samples',
                    'best_epoch', 'best_validation_mae_months', 'training', 'validation', 'test',
                    'target_mae_months', 'target_passed', 'patient_isolation', 'limitations')}}
    initialization_path = run_dir / 'initialization.json'
    if initialization_path.is_file():
        metadata['initialization'] = json.loads(initialization_path.read_text(encoding='utf-8'))
    elif config.get('pretrained_weights_sha256'):
        metadata['initialization'] = {'kind': 'frozen_official_imagenet',
            'weights': config['pretrained_weights'], 'sha256': config['pretrained_weights_sha256']}
    metadata['training_config_sha256'] = sha256_file(run_dir / 'config.json')
    atomic_json(directory / 'candidate.json', metadata)
    return report


def run_training(config, data_dir, output_root=None, run_dir=None, resume=None):
    device = resolve_device(config['device'])
    torch.set_num_threads(config['threads'])
    saved = load_checkpoint(resume) if resume else None
    if saved:
        validate_resume_config(saved['config'], config)
        inferred = Path(resume).resolve().parent.parent
        run_dir = Path(run_dir).resolve() if run_dir else inferred
        if run_dir != inferred:
            raise ValueError('Resume must use the checkpoint original run directory')
        validate_resume_config(json.loads((run_dir / 'config.json').read_text(encoding='utf-8')), config)
    elif run_dir:
        run_dir = Path(run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = create_run_directory(output_root, config['architecture'])
    checkpoints = run_dir / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    environment_path = run_dir / ('environment-resume-' + uuid.uuid4().hex[:8] + '.json' if saved else 'environment.json')
    atomic_json(environment_path, {**environment_info(config['device']),
                'resolved_device': str(device), 'amp_enabled': bool(config['amp'] and device.type == 'cuda')})
    status = {'status': 'initializing', 'run_dir': str(run_dir), 'device': str(device),
              'started_at': now(), 'completed_epochs': saved['epoch'] if saved else 0, 'test_evaluated': False}
    atomic_json(run_dir / 'status.json', status)
    try:
        seed_everything(config['seed'])
        reference = load_data_reference(Path(data_dir))
        datasets = {split: BoneAgeDataset(Path(data_dir), split, config['image_size'], reference=reference)
                    for split in ('train', 'validation')}
        cache_metadata = getattr(datasets['train'], 'cache_metadata', None)
        if cache_metadata is not None:
            reference['training_cache'] = cache_metadata
        data_reference = compact_reference(reference)
        if saved:
            expected_reference = data_reference.copy()
            # Early V0.3 runs recorded the audited 96px cache SHA directly.
            if 'training_cache' not in saved['data_reference']:
                expected_reference.pop('training_cache', None)
            if saved['data_reference'] != expected_reference:
                raise ValueError('Resume dataset manifest/split/audit/cache checksum mismatch')
        if not saved:
            atomic_json(run_dir / 'config.json', config)
            atomic_json(run_dir / 'data-reference.json', data_reference)
        if min(map(len, datasets.values())) < 1:
            raise ValueError('Nonempty independent train and validation splits are required')
        generator = torch.Generator().manual_seed(config['seed'])
        options = {'batch_size': config['batch_size'], 'num_workers': config['num_workers'], 'pin_memory': device.type == 'cuda'}
        train_loader = DataLoader(datasets['train'], shuffle=True, generator=generator, **options)
        train_eval_loader = DataLoader(datasets['train'], shuffle=False, **options)
        validation_loader = DataLoader(datasets['validation'], shuffle=False, **options)
        if not saved and config['weights_path']:
            # Hash and deserialize the same bytes, so a mutable warm-start file
            # cannot be swapped between integrity recording and loading.
            weight_path = Path(config['weights_path']).expanduser().resolve()
            raw_weights = weight_path.read_bytes()
            initialization = {'kind': 'local_state_dict', 'path': str(weight_path),
                              'sha256': hashlib.sha256(raw_weights).hexdigest()}
            model = build_model(config['architecture'], pretrained=False)
            model.load_state_dict(torch.load(io.BytesIO(raw_weights), map_location='cpu', weights_only=True), strict=True)
            del raw_weights
            atomic_json(run_dir / 'initialization.json', initialization)
        else:
            model = build_model(config['architecture'], pretrained=config['pretrained'] if not saved else False)
            if not saved:
                initialization = {'kind': 'official_imagenet' if config['pretrained'] else 'random',
                                  'seed': config['seed']}
                if config['pretrained']:
                    weight_path = Path(torch.hub.get_dir()) / 'checkpoints/resnet50-11ad3fa6.pth'
                    if not weight_path.is_file():
                        raise RuntimeError('Cannot identify the official pretrained weights for provenance')
                    initialization.update(weights='ResNet50_Weights.IMAGENET1K_V2',
                                          sha256=sha256_file(weight_path))
                atomic_json(run_dir / 'initialization.json', initialization)
        model = model.to(device)
        optimizer = build_optimizer(model, config)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['minimum_lr'])
        scaler = torch.amp.GradScaler('cuda', init_scale=1024,
                                     enabled=bool(config['amp'] and device.type == 'cuda'))
        history, completed_epoch, best_epoch, best_mae, bad_epochs = [], 0, 0, float('inf'), 0
        early_stopping_best, best_model_state = float('inf'), None
        if saved:
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            scaler.load_state_dict(saved['scaler'])
            completed_epoch, best_epoch = saved['epoch'], saved['best_epoch']
            best_mae, bad_epochs, history = saved['best_mae'], saved['bad_epochs'], saved['history']
            early_stopping_best = saved.get('early_stopping_best', best_mae)
            best_model_state = saved.get('best_model')
            if best_epoch and best_model_state is None:
                old_best = load_checkpoint(checkpoints / 'best.pt')
                if old_best['best_epoch'] != best_epoch or old_best['best_mae'] != best_mae:
                    raise ValueError('Legacy checkpoint best-state reference is inconsistent')
                best_model_state = old_best['model']
            if saved['phase'] != phase_for(config, completed_epoch):
                raise ValueError('Resume phase is inconsistent with its epoch')
            restore_rng(saved['rng'], generator)
            write_history(run_dir / 'history.jsonl', history)
        start = time.monotonic()

        def payload():
            return {'schema_version': CHECKPOINT_SCHEMA, 'config': config, 'data_reference': data_reference,
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(), 'epoch': completed_epoch, 'best_epoch': best_epoch,
                    'best_mae': best_mae, 'bad_epochs': bad_epochs, 'history': history,
                    'early_stopping_best': early_stopping_best, 'best_model': best_model_state,
                    'phase': phase_for(config, completed_epoch), 'rng': rng_state(generator)}

        if not saved:
            atomic_torch_save(checkpoints / 'latest.pt', payload())
            write_history(run_dir / 'history.jsonl', history)
        status.update(status='training', completed_epochs=completed_epoch)
        atomic_json(run_dir / 'status.json', status)
        patience = config['early_stopping_patience']
        stopped = bool(patience and bad_epochs >= patience)
        for epoch in range(completed_epoch, config['epochs']):
            if stopped:
                break
            learning_rates = {group['name']: group['lr'] for group in optimizer.param_groups}
            loss, phase = train_epoch(model, train_loader, optimizer, scaler, device, config, epoch)
            # Unaugmented eval-mode MAE on both splits is directly comparable.
            tp, ty, _ts, _ = predict(model, train_eval_loader, device, config)
            vp, vy, _vs, _ = predict(model, validation_loader, device, config)
            train_mae, validation_mae = float(np.abs(tp - ty).mean()), float(np.abs(vp - vy).mean())
            significant = validation_mae < early_stopping_best - config['early_stopping_min_delta']
            improved = validation_mae < best_mae
            bad_epochs = 0 if significant else bad_epochs + 1
            if significant:
                early_stopping_best = validation_mae
            if improved:
                best_mae, best_epoch = validation_mae, epoch + 1
                best_model_state = cpu_tree(model.state_dict())
            scheduler.step()
            completed_epoch = epoch + 1
            record = {'epoch': completed_epoch, 'phase': phase, 'train_loss': loss,
                      'amp_overflow_steps': getattr(model, '_training_amp_overflow_steps', 0),
                      'optimizer_steps': getattr(model, '_training_optimizer_steps', 0),
                      'train_eval_mae_months': train_mae, 'validation_mae_months': validation_mae,
                      'learning_rates': learning_rates, 'best_epoch': best_epoch, 'bad_epochs': bad_epochs,
                      'elapsed_seconds': round(time.monotonic() - start, 3), 'device': str(device)}
            if device.type == 'cuda':
                record['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
                record['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
            history.append(record)
            state = payload()
            if improved:
                atomic_torch_save(checkpoints / 'best.pt', state)
            atomic_torch_save(checkpoints / 'latest.pt', state)
            write_history(run_dir / 'history.jsonl', history)
            status.update(completed_epochs=completed_epoch, best_epoch=best_epoch, best_validation_mae_months=best_mae)
            atomic_json(run_dir / 'status.json', status)
            print(json.dumps(record, allow_nan=False), flush=True)
            stopped = bool(patience and bad_epochs >= patience)
        final_status = 'early_stopped' if stopped and completed_epoch < config['epochs'] else 'completed'
        report = export_candidate(model, run_dir, config, reference, train_eval_loader, validation_loader,
                                  device, best_epoch, best_mae, completed_epoch, final_status, best_model_state)
        status.update(status=final_status, finished_at=now(), completed_epochs=completed_epoch,
                      candidate=str(run_dir / 'export/candidate.json'), test_evaluated=False)
        atomic_json(run_dir / 'status.json', status)
        return {**status, 'evaluation': report}
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            reason = 'interrupted'
        elif isinstance(exc, torch.OutOfMemoryError) or 'out of memory' in str(exc).lower():
            reason = 'failed_oom'
        elif isinstance(exc, FloatingPointError):
            reason = 'failed_nonfinite'
        else:
            reason = 'failed'
        status.update(status=reason, finished_at=now(), error=f'{type(exc).__name__}: {exc}',
                      recovery='Resume checkpoints/latest.pt from the last completed epoch; no partial epoch was committed')
        atomic_json(run_dir / 'status.json', status)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--data', type=Path, default=Settings.from_env().dataset)
    paths = parser.add_mutually_exclusive_group()
    paths.add_argument('--output', type=Path, help='Parent for a unique run (default var/runs)')
    paths.add_argument('--run-dir', type=Path, help='Exact new directory; must not exist except when resuming')
    parser.add_argument('--resume', type=Path, help='RUN/checkpoints/latest.pt or best.pt')
    parser.add_argument('--device', help='Explicit cpu, cuda or cuda:<index>; no automatic fallback')
    parser.add_argument('--architecture', '--model', dest='architecture')
    for flag in ('epochs', 'batch-size', 'threads', 'seed', 'gradient-accumulation-steps', 'early-stopping-patience'):
        parser.add_argument('--' + flag, type=int)
    parser.add_argument('--weights-path')
    parser.add_argument('--loss', choices=('smooth_l1', 'l1'))
    parser.add_argument('--loss-beta', type=float)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    overrides = {key: value for key, value in vars(args).items()
                 if key not in ('config', 'data', 'output', 'run_dir', 'resume') and value is not None}
    try:
        if args.resume and args.config is None:
            config = load_checkpoint(args.resume)['config'].copy()
            config.update(overrides)
            config = load_config(overrides=config)
        else:
            config = load_config(args.config, overrides)
        result = run_training(config, args.data, output_root=args.output, run_dir=args.run_dir, resume=args.resume)
    except (Exception, KeyboardInterrupt) as exc:
        print(json.dumps({'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                          'error': f'{type(exc).__name__}: {exc}'}, ensure_ascii=False), file=sys.stderr, flush=True)
        raise SystemExit(130 if isinstance(exc, KeyboardInterrupt) else 1) from exc
    print(json.dumps({key: value for key, value in result.items() if key != 'evaluation'}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
