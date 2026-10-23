"""Report actual CPU/CUDA capability without weights downloads or dataset reads."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.training import atomic_json, environment_info, resolve_device
from app.network import build_model, normalize_for_model, preprocess_version_for


def smoke_test(device, architecture, batch_size, image_size, amp):
    """Exercise a complete real forward/backward/AdamW step, without pretrained downloads."""
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(architecture, pretrained=False).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    images = torch.full((batch_size, 1, image_size, image_size), 127, dtype=torch.uint8, device=device)
    sexes = torch.zeros((batch_size, 1), device=device)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024, enabled=amp)
    with torch.autocast(device_type=device.type, enabled=amp):
        predictions = model(normalize_for_model(images, preprocess_version_for(architecture)), sexes)
        loss = torch.nn.functional.smooth_l1_loss(predictions, torch.zeros_like(predictions), beta=.5)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Smoke-test loss is not finite')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    memory = {'peak_allocated_bytes': None, 'peak_reserved_bytes': None}
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        memory = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
    return {'status': 'passed', 'architecture': architecture, 'pretrained': False,
            'batch_size': batch_size, 'image_size': image_size, 'amp_enabled': amp,
            'loss': float(loss.detach()), 'gradient_norm': float(gradient_norm),
            'optimizer_step': True, **memory}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--architecture', '--model')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--image-size', type=int)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    report = environment_info(args.device)
    try:
        device = resolve_device(args.device)
        report['resolved_device'] = str(device)
        architecture = args.architecture or ('resnet50-sex-v1' if device.type == 'cuda' else 'boneage-small-cnn-v1')
        size = args.image_size or (224 if architecture == 'resnet50-sex-v1' else 96)
        amp = args.amp if args.amp is not None else device.type == 'cuda'
        if amp and device.type != 'cuda':
            raise ValueError('AMP preflight requires CUDA')
        if args.batch_size < 1 or size < 1:
            raise ValueError('batch-size and image-size must be positive')
        report['smoke_test'] = smoke_test(device, architecture, args.batch_size, size, amp)
        report['status'] = 'ready'
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        report.update(status='unavailable', error=str(exc))
    if args.output:
        atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['status'] == 'ready' else 1)


if __name__ == '__main__':
    main()
