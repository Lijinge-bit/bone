"""Small synthetic checks for training integrity; never open the research holdout."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app import training


class TinyDataset(Dataset):
    requested_splits = []

    def __init__(self, data_dir, split, image_size, reference=None):
        if split not in ('train', 'validation'):
            raise AssertionError('Training attempted to open the final holdout')
        self.requested_splits.append(split)
        self.split = split
        self.count = 5 if split == 'train' else 3

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return (torch.full((1, 8, 8), 30 + index * 35, dtype=torch.uint8),
                torch.tensor([float(index % 2)]), torch.tensor(100.0 + index * 7),
                f'{self.split}-{index}')


class TinyModel(nn.Module):
    def __init__(self, dropout=.25):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 2, 1), nn.SiLU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(3, 1))

    def forward(self, image, sex):
        features = self.features(image).mean((2, 3))
        return self.head(torch.cat((features, sex), 1)).squeeze(1)


REFERENCE = {'manifest_sha256': 'a' * 64, 'split_sha256': 'b' * 64,
             'audit_sha256': 'c' * 64, 'source_sha256': 'd' * 64,
             'records': [], 'audit': {'splits': {'train': 5, 'validation': 3, 'test': 1},
                                      'patient_isolation': 'unverified', 'grouping': 'synthetic'}}


class TrainingIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.config = training.load_config(overrides={'architecture': 'cnn', 'epochs': 3,
                    'batch_size': 2, 'threads': 1, 'augmentation': True})
        TinyDataset.requested_splits = []

    def synthetic_training(self):
        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(training, 'BoneAgeDataset', TinyDataset))
        stack.enter_context(patch.object(training, 'load_data_reference', return_value=copy.deepcopy(REFERENCE)))
        stack.enter_context(patch.object(training, 'build_model', side_effect=lambda *a, **k: TinyModel()))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        return stack

    def test_configuration_and_explicit_cuda_failure(self):
        self.assertEqual(training.resolve_device('cpu'), torch.device('cpu'))
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA was requested'):
                training.resolve_device('cuda')
        with self.assertRaisesRegex(ValueError, 'Unsupported architecture'):
            training.load_config(overrides={'architecture': 'imaginary'})
        with self.assertRaisesRegex(ValueError, 'AMP requires CUDA'):
            training.load_config(overrides={'architecture': 'cnn', 'amp': True})
        self.assertEqual(training.load_config(overrides={'architecture':'cnn','loss':'l1'})['loss'],'l1')
        with self.assertRaisesRegex(ValueError,'loss_beta'):
            training.load_config(overrides={'architecture':'cnn','loss_beta':0})
        with self.assertRaisesRegex(ValueError,'boolean'):
            training.load_config(overrides={'architecture':'cnn','amp':'false'})
        changed = {**self.config, 'epochs': 99}
        with self.assertRaisesRegex(ValueError, 'epochs'):
            training.validate_resume_config(self.config, changed)

    def test_accumulation_matches_actual_large_batches_including_single_sample_tail(self):
        torch.manual_seed(7)
        small = TinyModel(dropout=0)
        large = copy.deepcopy(small)
        dataset = TinyDataset(None, 'train', 96)
        config = {**self.config, 'augmentation': False, 'gradient_clip_norm': 1e6}
        for model, batch_size, accumulation in ((small, 2, 2), (large, 4, 1)):
            optimizer = torch.optim.SGD(model.parameters(), lr=.02)
            scaler = torch.amp.GradScaler('cuda', enabled=False)
            training.train_epoch(model, DataLoader(dataset, batch_size=batch_size), optimizer,
                scaler, torch.device('cpu'), {**config, 'gradient_accumulation_steps': accumulation}, 0)
        for actual, expected in zip(small.parameters(), large.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_freeze_boundary_and_batchnorm_policy(self):
        model = TinyModel()
        model.features = nn.Sequential(nn.Conv2d(1, 2, 1), nn.BatchNorm2d(2))
        config = training.load_config(overrides={'device': 'cpu', 'amp': False})
        model.train()
        self.assertEqual(training.apply_phase(model, config, 2), 'head_only')
        self.assertFalse(any(p.requires_grad for p in model.features.parameters()))
        self.assertFalse(model.features.training)
        model.train()
        self.assertEqual(training.apply_phase(model, config, 3), 'finetune')
        self.assertTrue(all(p.requires_grad for p in model.features.parameters()))
        self.assertFalse(model.features[1].training)
        model.train()
        training.apply_phase(model, {**config, 'bn_strategy': 'update_running_stats'}, 3)
        self.assertTrue(model.features[1].training)
        optimizer = training.build_optimizer(model, config)
        self.assertEqual([g['lr'] for g in optimizer.param_groups], [3e-5, 3e-4])

    def test_epoch_resume_restores_rng_optimizer_schedule_and_candidate_contract(self):
        with tempfile.TemporaryDirectory() as tmp, self.synthetic_training():
            root = Path(tmp)
            active = root / 'active.json'
            active.write_text('existing-model')
            complete = training.run_training(self.config, root, run_dir=root / 'complete')
            original_train_epoch = training.train_epoch

            def interrupt_second_epoch(*args, **kwargs):
                if args[-1] == 1:
                    raise KeyboardInterrupt('synthetic interruption')
                return original_train_epoch(*args, **kwargs)

            with patch.object(training, 'train_epoch', side_effect=interrupt_second_epoch):
                with self.assertRaises(KeyboardInterrupt):
                    training.run_training(self.config, root, run_dir=root / 'resumed')
            interrupted_status = json.loads((root / 'resumed/status.json').read_text())
            self.assertEqual(interrupted_status['status'], 'interrupted')
            resume_path = root / 'resumed/checkpoints/latest.pt'
            checkpoint = training.load_checkpoint(resume_path)
            self.assertEqual(checkpoint['epoch'], 1)
            resumed = training.run_training(self.config, root, resume=resume_path)
            self.assertEqual(resumed['status'], 'completed')
            a = training.load_checkpoint(root / 'complete/checkpoints/latest.pt')
            b = training.load_checkpoint(resume_path)
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
            self.assertEqual(a['scheduler'], b['scheduler'])
            self.assertEqual(a['bad_epochs'], b['bad_epochs'])
            self.assertEqual(a['early_stopping_best'], b['early_stopping_best'])
            self.assertEqual(a['scaler'], b['scaler'])
            self.assertEqual(a['phase'], b['phase'])
            for x, y in zip(a['history'], b['history']):
                self.assertEqual({k: v for k, v in x.items() if k != 'elapsed_seconds'},
                                 {k: v for k, v in y.items() if k != 'elapsed_seconds'})
            candidate = json.loads((root / 'resumed/export/candidate.json').read_text())
            self.assertEqual(candidate['schema_version'], 2)
            self.assertEqual(candidate['checkpoint'], 'model.pt')
            self.assertIsNone(candidate['evaluation']['test'])
            self.assertFalse(candidate['evaluation']['target_passed'])
            self.assertEqual(candidate['data_reference']['split_sha256'], REFERENCE['split_sha256'])
            self.assertEqual(candidate['evaluation']['validation']['n'], 3)
            self.assertEqual(active.read_text(), 'existing-model')
            self.assertEqual(set(TinyDataset.requested_splits), {'train', 'validation'})
            self.assertTrue((Path(complete['run_dir']) / 'validation/predictions.csv').is_file())

    def test_nonfinite_and_oom_have_explicit_failure_status_and_recoverable_boundary(self):
        for error, expected in ((FloatingPointError('bad gradient'), 'failed_nonfinite'),
                                (torch.OutOfMemoryError('synthetic OOM'), 'failed_oom')):
            with self.subTest(expected), tempfile.TemporaryDirectory() as tmp, self.synthetic_training():
                run_dir = Path(tmp) / 'run'
                with patch.object(training, 'train_epoch', side_effect=error):
                    with self.assertRaises(type(error)):
                        training.run_training(self.config, tmp, run_dir=run_dir)
                status = json.loads((run_dir / 'status.json').read_text())
                self.assertEqual(status['status'], expected)
                self.assertFalse(status['test_evaluated'])
                self.assertEqual(training.load_checkpoint(run_dir / 'checkpoints/latest.pt')['epoch'], 0)
                self.assertFalse((run_dir / 'export/candidate.json').exists())

    def test_early_stopping_state_and_resume_data_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp, self.synthetic_training():
            root = Path(tmp)
            config = {**self.config, 'epochs': 6, 'early_stopping_patience': 1,
                      'early_stopping_min_delta': 1000.0}
            result = training.run_training(config, root, run_dir=root / 'run')
            self.assertEqual(result['status'], 'early_stopped')
            self.assertEqual(result['completed_epochs'], 2)
            checkpoint_path = root / 'run/checkpoints/latest.pt'
            checkpoint = training.load_checkpoint(checkpoint_path)
            self.assertEqual(checkpoint['bad_epochs'], 1)
            self.assertEqual(checkpoint['early_stopping_best'], checkpoint['history'][0]['validation_mae_months'])
            changed = copy.deepcopy(REFERENCE)
            changed['split_sha256'] = 'e' * 64
            with patch.object(training, 'load_data_reference', return_value=changed):
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    training.run_training(config, root, resume=checkpoint_path)

    def test_unique_output_and_existing_explicit_run_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            one = training.create_run_directory(tmp, 'boneage-small-cnn-v1')
            two = training.create_run_directory(tmp, 'boneage-small-cnn-v1')
            self.assertNotEqual(one, two)
            sentinel = one / 'sentinel'
            sentinel.write_text('preserve')
            with self.assertRaises(FileExistsError):
                training.run_training(self.config, tmp, run_dir=one)
            self.assertEqual(sentinel.read_text(), 'preserve')

    def test_nonfinite_gradient_does_not_update_parameters(self):
        model = TinyModel(dropout=0)
        before = copy.deepcopy(model.state_dict())
        hook = next(model.parameters()).register_hook(lambda gradient: gradient * float('nan'))
        try:
            with self.assertRaisesRegex(FloatingPointError, 'Non-finite gradient'):
                training.train_epoch(model, DataLoader(TinyDataset(None, 'train', 96), batch_size=2),
                    torch.optim.SGD(model.parameters(), lr=.1), torch.amp.GradScaler('cuda', enabled=False),
                    torch.device('cpu'), {**self.config, 'augmentation': False}, 0)
        finally:
            hook.remove()
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
