"""Verify actual layer updates, cache identity and completed-epoch resume."""
import contextlib
import copy
import io
import json
from collections import OrderedDict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app import tail_training as tail
from app.network import build_model, normalize_for_model
from tests.test_training import REFERENCE


class CacheImages(Dataset):
    def __init__(self, split='train', count=5):
        if split not in ('train', 'validation'):
            raise AssertionError('Attempted to read holdout')
        self.split = split
        self.ids = [f'{split}-{i}' for i in range(count)]
        self.records = [{'sex': 'M' if i % 2 else 'F', 'boneage': 70 + 10 * i} for i in range(count)]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return torch.full((1, 8, 8), 30 + i * 15, dtype=torch.uint8), torch.tensor([float(i % 2)]), torch.tensor(70. + 10 * i), self.ids[i]


class TinyTail(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict(
            [(f'stem{i}', nn.Identity()) for i in range(7)] +
            [('layer4', nn.Sequential(nn.Identity(), nn.Identity(), nn.Sequential(
                nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2), nn.SiLU()))) ]))
        self.sex_embedding = nn.Sequential(nn.Linear(1, 2), nn.SiLU())
        self.head = nn.Sequential(nn.Dropout(.2), nn.Linear(4, 1))

    def predict_from_features(self, features, sex):
        return self.head(torch.cat((features.mean((2, 3)), self.sex_embedding(sex)), 1)).squeeze(1)


LINEAGE = {'model_id': 'synthetic', 'checkpoint_sha256': 'e' * 64, 'metadata_sha256': 'f' * 64,
           'architecture': 'resnet50-sex-v1', 'image_size': 224,
           'preprocess_version': 'imagenet-gray-letterbox-v1'}


class TailTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, adapt=True):
        path = Path(__file__).resolve().parents[1] / 'configs/train_tail_cpu_adapt.json'
        config = json.loads(path.read_text())
        config.update(threads=1, batch_size=2, adapt_tail=adapt,
                      training_protocol='cpu_last_residual_block_finetuning' if adapt else 'cpu_matched_frozen_control')
        return config

    def test_real_resnet_split_parity_and_only_requested_layers_update(self):
        torch.manual_seed(4)
        model = build_model('resnet50-sex-v1').eval().to(memory_format=torch.channels_last)
        images = normalize_for_model(torch.randint(0, 255, (2, 1, 224, 224), dtype=torch.uint8), LINEAGE['preprocess_version'])
        sex = torch.tensor([[0.], [1.]])
        with torch.inference_mode():
            prefixes = tail.prefix_forward(model, images)
            direct = model(images, sex)
            split = tail.tail_forward(model, prefixes, sex)
        torch.testing.assert_close(direct, split, rtol=1e-5, atol=1e-6)
        prefixes = prefixes.clone()
        dataset = [(prefixes[i], sex[i], torch.tensor(150. + i * 10), str(i)) for i in range(2)]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        tail.train_epoch(model, DataLoader(dataset, batch_size=2), optimizer, self.config(), 0)
        for key, value in model.features.state_dict().items():
            torch.testing.assert_close(value, state['features.' + key], rtol=0, atol=0)
        tail.train_epoch(model, DataLoader(dataset, batch_size=2), optimizer, self.config(), 1)
        changed = []
        for key, value in model.state_dict().items():
            if not torch.equal(value, state[key]):
                changed.append(key)
                self.assertTrue(key.startswith(('features.layer4.2.', 'head.', 'sex_embedding.')), key)
                self.assertFalse(key.endswith(('running_mean', 'running_var', 'num_batches_tracked')))
        self.assertTrue(any(k.startswith('features.layer4.2.') for k in changed))
        self.assertTrue(any(k.startswith('head.') for k in changed))

    def test_cache_resume_identity_and_tamper_rejection(self):
        data = CacheImages()
        model = nn.Identity()
        with tempfile.TemporaryDirectory() as tmp, patch.object(tail, 'FEATURE_SHAPE', (2, 8, 8)), contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            calls = 0
            def interrupt(x, images):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt('interrupted prefix extraction')
                return images[:, :2]
            with patch.object(tail, 'prefix_forward', side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    tail.prepare_split(model, data, root / 'cache', REFERENCE, LINEAGE, 2)
            self.assertEqual(json.loads((root / 'cache/progress.json').read_text())['completed_samples'], 2)
            with patch.object(tail, 'prefix_forward', side_effect=lambda _m, x: x[:, :2]):
                resumed = tail.prepare_split(model, data, root / 'cache', REFERENCE, LINEAGE, 2)
                fresh = tail.prepare_split(model, data, root / 'fresh', REFERENCE, LINEAGE, 2)
            self.assertEqual(resumed['feature_sha256'], fresh['feature_sha256'])
            self.assertEqual(len(tail.PrefixDataset(root / 'cache', data, REFERENCE, LINEAGE)), 5)
            with self.assertRaisesRegex(ValueError, 'identity'):
                tail.prepare_split(model, data, root / 'cache', REFERENCE, {**LINEAGE, 'checkpoint_sha256': '0' * 64}, 2)
            with (root / 'cache/features.npy').open('ab') as f:
                f.write(b'tampered')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                tail.prepare_split(model, data, root / 'cache', REFERENCE, LINEAGE, 2)
            with self.assertRaises(ValueError):
                tail.PrefixDataset(root / 'cache', data, REFERENCE, LINEAGE)

    def test_epoch_resume_matches_continuous_training_and_never_opens_test(self):
        torch.manual_seed(11)
        original = TinyTail()
        config = self.config()
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            root = Path(tmp)
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(tail, 'FEATURE_SHAPE', (2, 8, 8)))
            stack.enter_context(patch.object(tail, 'load_data_reference', return_value=copy.deepcopy(REFERENCE)))
            stack.enter_context(patch.object(tail, 'verify_reference'))
            stack.enter_context(patch.object(tail, 'initial_model', side_effect=lambda *_: (copy.deepcopy(original), LINEAGE)))
            stack.enter_context(patch.object(tail, 'BoneAgeDataset', side_effect=lambda _p, split, *_: CacheImages(split, 5 if split == 'train' else 3)))
            with patch.object(tail, 'prefix_forward', side_effect=lambda _m, x: x[:, :2]):
                for split in ('train', 'validation'):
                    tail.prepare_split(original, CacheImages(split, 5 if split == 'train' else 3), root / 'cache' / split, REFERENCE, LINEAGE, 2)
            complete = tail.run_experiment(root, 'candidate.json', root, root / 'cache', root / 'complete', config)
            real_epoch = tail.train_epoch
            def interrupt(*args):
                if args[-1] == 1:
                    raise KeyboardInterrupt()
                return real_epoch(*args)
            with patch.object(tail, 'train_epoch', side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    tail.run_experiment(root, 'candidate.json', root, root / 'cache', root / 'resumed', config)
            resumed = tail.run_experiment(root, 'candidate.json', root, root / 'cache', root / 'resumed', config, True)
            a = torch.load(root / 'complete/checkpoints/latest.pt', weights_only=True)
            b = torch.load(root / 'resumed/checkpoints/latest.pt', weights_only=True)
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
            self.assertEqual(a['scheduler'], b['scheduler'])
            self.assertEqual(a['best_mae'], b['best_mae'])
            self.assertEqual(a['bad_epochs'], b['bad_epochs'])
            self.assertEqual(complete['best_validation_mae_months'], resumed['best_validation_mae_months'])
            self.assertIsNone(resumed['test'])
            self.assertFalse(resumed['target_passed'])
            with self.assertRaisesRegex(ValueError, 'configuration'):
                tail.run_experiment(root, 'candidate.json', root, root / 'cache', root / 'resumed', {**config, 'seed': 99}, True)

    def test_configs_keep_the_control_comparable_and_reject_unsupported_modes(self):
        root = Path(__file__).resolve().parents[1]
        control = json.loads((root / 'configs/train_tail_cpu_control.json').read_text())
        adapt = json.loads((root / 'configs/train_tail_cpu_adapt.json').read_text())
        tail.validate_config(control); tail.validate_config(adapt)
        ignored = {'adapt_tail', 'training_protocol', 'limitations'}
        self.assertEqual({k:v for k,v in control.items() if k not in ignored}, {k:v for k,v in adapt.items() if k not in ignored})
        for override in ({'augmentation':True}, {'amp':True}, {'bn_strategy':'update_running_stats'}, {'adapt_tail':'false'}, {'loss_beta':0}):
            with self.assertRaises(ValueError):
                tail.validate_config({**adapt, **override})


if __name__ == '__main__':
    unittest.main()
