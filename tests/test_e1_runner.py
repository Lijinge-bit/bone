"""CUDA preflight must gate the complete paired experiment suite."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_e1
from app.training import load_config


class E1RunnerTests(unittest.TestCase):
    def test_gpu_pair_changes_only_the_backbone_freeze_schedule(self):
        root = Path(__file__).resolve().parents[1]
        frozen = load_config(root / 'configs/train_e1_4060_frozen.json')
        tuned = load_config(root / 'configs/train_e1_4060_finetune.json')
        self.assertEqual({k:v for k,v in frozen.items() if k != 'freeze_backbone_epochs'},
                         {k:v for k,v in tuned.items() if k != 'freeze_backbone_epochs'})
        self.assertEqual(frozen['freeze_backbone_epochs'], frozen['epochs'])
        self.assertEqual(tuned['freeze_backbone_epochs'], 3)
        self.assertFalse(tuned['augmentation'])
        self.assertEqual(tuned['device'], 'cuda')

    def fixture(self, root):
        data = root / 'data'; data.mkdir()
        for name in ('audit.json', 'manifest.jsonl', 'rsna-bone-age-v2.zip'):
            (data / name).write_bytes(b'synthetic fixture')
        weights = root / 'initial.pt'; weights.write_bytes(b'checksum fixture')
        return data, weights

    def test_failed_or_non_cuda_preflight_never_starts_training(self):
        for mode in ('exit', 'cpu'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); data, weights = self.fixture(root)
                def native(command, **kwargs):
                    if mode == 'exit':
                        raise subprocess.CalledProcessError(1, command)
                    output = Path(command[command.index('--output') + 1])
                    output.write_text(json.dumps({'status':'ready','resolved_device':'cpu'}))
                with patch.object(run_e1, 'INITIAL_SHA', hashlib.sha256(weights.read_bytes()).hexdigest()), \
                     patch.object(run_e1.subprocess, 'run', side_effect=native) as process:
                    with self.assertRaises((subprocess.CalledProcessError, RuntimeError)):
                        run_e1.run_suite(data, root / 'runs', weights, [20261023])
                self.assertEqual(process.call_count, 1)
                suite = next((root / 'runs').iterdir())
                status = json.loads((suite / 'suite.json').read_text())
                self.assertEqual(status['status'], 'failed')
                self.assertEqual(status['runs'], [])
                self.assertFalse(status['test_evaluated'])
                self.assertFalse(status['model_registered'])

    def test_suite_runs_both_arms_per_seed_without_evaluation_or_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data, weights = self.fixture(root)
            def native(command, **kwargs):
                if command[1].endswith('check_environment.py'):
                    Path(command[command.index('--output') + 1]).write_text(json.dumps({'status':'ready','resolved_device':'cuda:0'}))
            with patch.object(run_e1, 'INITIAL_SHA', hashlib.sha256(weights.read_bytes()).hexdigest()), \
                 patch.object(run_e1.subprocess, 'run', side_effect=native) as process:
                suite = run_e1.run_suite(data, root / 'runs', weights, [1, 2])
            calls = [call.args[0] for call in process.call_args_list]
            self.assertEqual([Path(c[1]).name for c in calls], ['check_environment.py','prepare_training_data.py']+['train.py']*4)
            state = json.loads((suite / 'suite.json').read_text())
            self.assertEqual([(r['arm'],r['seed']) for r in state['runs']], [('frozen',1),('finetune',1),('frozen',2),('finetune',2)])
            self.assertEqual(state['status'], 'completed')
            self.assertFalse(state['test_evaluated'])
            self.assertFalse(state['model_registered'])
