"""Training data checks use original synthetic PNGs, never the real hold-out set."""
import hashlib
import io
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from app.imaging import PREPROCESS_VERSION, letterbox
from app.training_data import BoneAgeDataset, load_data_reference, prepare_training_cache


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TrainingDataTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='bone training ')
        self.directory = Path(self.temporary.name)
        self.archive = self.directory / 'rsna-bone-age-v2.zip'
        self.records = []
        self.originals = {}
        # High frequency content makes direct 224 preprocessing observably different
        # from an interpolation of the historical 96-pixel training cache.
        with ZipFile(self.archive, 'w') as archive:
            rows = (('101', 'train', 'F', 36), ('102', 'train', 'M', 84),
                    ('103', 'validation', 'F', 132), ('104', 'test', 'M', 180))
            for index, (source_id, split, sex, boneage) in enumerate(rows):
                yy, xx = np.indices((320 + index * 3, 192 + index * 2))
                pixels = ((xx * 29 + yy * 17 + (xx % 3) * 70 + index * 13) % 256).astype(np.uint8)
                original = Image.fromarray(pixels)
                self.originals[source_id] = original
                output = io.BytesIO()
                original.save(output, format='PNG')
                data = output.getvalue()
                path = f'boneage-training-dataset/boneage-training-dataset/{source_id}.png'
                archive.writestr(path, data)
                self.records.append({'source_id': source_id, 'kind': 'train',
                    'status': 'valid', 'archive_path': path,
                    'sha256': hashlib.sha256(data).hexdigest(),
                    'width': original.width, 'height': original.height, 'mode': 'L',
                    'boneage': boneage, 'sex': sex, 'split': split,
                    'patient_id': None, 'duplicate_group': str(index)})
            output = io.BytesIO()
            Image.new('L', (128, 192), 244).save(output, format='PNG')
            data = output.getvalue()
            path = 'boneage-test-dataset/boneage-test-dataset/999.png'
            archive.writestr(path, data)
            self.unlabeled = {'source_id': '999', 'kind': 'unlabeled', 'status': 'valid',
                'archive_path': path, 'sha256': hashlib.sha256(data).hexdigest(),
                'boneage': None, 'sex': 'M', 'split': 'unlabeled_prediction',
                'width': 128, 'height': 192, 'mode': 'L', 'patient_id': None}
        self.audit = {'source': 'synthetic-unit-test', 'dataset_version': 2,
            'training_rows': 4, 'matched_training': 4, 'unlabeled_rows': 1,
            'matched_unlabeled': 1, 'splits': {'train': 2, 'validation': 1, 'test': 1},
            'split_seed': 20261022, 'image_size': 96,
            'preprocess_version': PREPROCESS_VERSION}
        self.write_manifest()
        self.write_legacy_cache()

    def tearDown(self):
        self.temporary.cleanup()

    def write_audit(self):
        (self.directory / 'audit.json').write_text(
            json.dumps(self.audit), encoding='utf-8')

    def write_manifest(self):
        path = self.directory / 'manifest.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in
                                [*self.records, self.unlabeled]), encoding='utf-8')
        self.audit['manifest_sha256'] = sha256(path)
        self.write_audit()

    def write_legacy_cache(self, ids=None, splits=None):
        path = self.directory / 'cache-96.npz'
        np.savez(path,
            images=np.stack([letterbox(self.originals[row['source_id']], 96)[0]
                             for row in self.records]),
            targets=np.asarray([row['boneage'] for row in self.records], dtype=np.float32),
            sex=np.asarray([row['sex'] == 'M' for row in self.records], dtype=np.float32),
            ids=np.asarray(ids if ids is not None else [row['source_id'] for row in self.records]),
            split=np.asarray(splits if splits is not None else [0, 0, 1, 2], dtype=np.uint8))
        self.audit['cache_sha256'] = sha256(path)
        self.write_audit()

    def test_reference_preserves_audited_split_and_excludes_unlabeled(self):
        first = load_data_reference(self.directory)
        second = load_data_reference(self.directory)
        self.assertEqual([row['source_id'] for row in first['records']],
                         ['101', '102', '103', '104'])
        self.assertEqual(first['audit']['splits'], {'train': 2, 'validation': 1, 'test': 1})
        self.assertEqual(first['manifest_sha256'], self.audit['manifest_sha256'])
        self.assertEqual(len(first['split_sha256']), 64)
        self.assertEqual(first['split_sha256'], second['split_sha256'])
        self.assertEqual([row['split'] for row in first['records']],
                         ['train', 'train', 'validation', 'test'])

    def test_manifest_checksum_and_declared_counts_are_enforced(self):
        path = self.directory / 'manifest.jsonl'
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaises((ValueError, RuntimeError)):
            load_data_reference(self.directory)
        self.write_manifest()
        self.audit['splits']['train'] = 3
        self.write_audit()
        with self.assertRaises((ValueError, RuntimeError)):
            load_data_reference(self.directory)

    def test_duplicate_training_source_ids_are_rejected(self):
        self.records[1]['source_id'] = self.records[0]['source_id']
        self.write_manifest()
        with self.assertRaises((ValueError, RuntimeError)):
            load_data_reference(self.directory)

    def test_legacy_cache_retains_fixed_ids_labels_and_uint8_tensor_contract(self):
        reference = load_data_reference(self.directory)
        expected = {'train': ['101', '102'], 'validation': ['103'], 'test': ['104']}
        for split, ids in expected.items():
            with self.subTest(split=split):
                dataset = BoneAgeDataset(self.directory, split, 96, reference=reference)
                self.assertEqual(list(dataset.ids), ids)
                self.assertEqual([row['source_id'] for row in dataset.records], ids)
                image, sex, target, source_id = dataset[0]
                row = next(row for row in self.records if row['source_id'] == source_id)
                self.assertEqual(image.shape, (1, 96, 96))
                self.assertEqual(image.dtype, torch.uint8)
                self.assertEqual(sex.shape, (1,))
                self.assertEqual(sex.dtype, torch.float32)
                self.assertEqual(target.shape, ())
                self.assertEqual(target.dtype, torch.float32)
                self.assertEqual(target.item(), row['boneage'])
                self.assertEqual(sex.item(), float(row['sex'] == 'M'))

    def test_legacy_checksum_id_and_partition_mismatches_are_rejected(self):
        path = self.directory / 'cache-96.npz'
        path.write_bytes(path.read_bytes() + b'corrupt')
        with self.assertRaises((ValueError, RuntimeError)):
            BoneAgeDataset(self.directory, 'train', 96)
        # Updating the archive checksum alone must not legitimize swapped IDs.
        self.write_legacy_cache(ids=['102', '101', '103', '104'])
        with self.assertRaises((ValueError, RuntimeError)):
            BoneAgeDataset(self.directory, 'train', 96)
        self.write_legacy_cache(splits=[0, 2, 1, 0])
        with self.assertRaises((ValueError, RuntimeError)):
            BoneAgeDataset(self.directory, 'train', 96)

    def test_missing_high_resolution_cache_requires_preparation(self):
        with self.assertRaisesRegex((FileNotFoundError, ValueError, RuntimeError),
                                    '(?i)prepare|cache'):
            BoneAgeDataset(self.directory, 'train', 224)

    def test_prepared_cache_uses_original_pngs_and_survives_archive_removal(self):
        reference = load_data_reference(self.directory)
        prepare_training_cache(self.directory, 224)
        directory = self.directory / 'training-cache-224'
        self.assertTrue((directory / 'metadata.json').is_file())
        self.assertTrue((directory / 'ids.json').is_file())
        mapped = np.load(directory / 'images.npy', mmap_mode='r', allow_pickle=False)
        self.assertIsInstance(mapped, np.memmap)
        self.assertEqual(mapped.shape, (4, 224, 224))
        self.assertEqual(mapped.dtype, np.uint8)
        del mapped
        # Training must open only cache arrays, and remains usable when original
        # ZIP access is unavailable to spawned data-loader workers.
        self.archive.unlink()
        dataset = BoneAgeDataset(self.directory, 'train', 224, reference=reference)
        self.assertEqual(list(dataset.ids), ['101', '102'])
        image, _, _, source_id = dataset[0]
        direct, _ = letterbox(self.originals[source_id], 224)
        np.testing.assert_array_equal(image[0].numpy(), direct)
        tiny, _ = letterbox(self.originals[source_id], 96)
        interpolated, _ = letterbox(Image.fromarray(tiny), 224)
        self.assertFalse(np.array_equal(image[0].numpy(), interpolated))
        np.testing.assert_array_equal(dataset[0][0].numpy(), dataset[0][0].numpy())
        restored = pickle.loads(pickle.dumps(dataset))
        np.testing.assert_array_equal(restored[0][0].numpy(), image.numpy())

    def test_spawn_worker_reads_cache_without_original_zip(self):
        prepare_training_cache(self.directory, 224)
        self.archive.unlink()
        dataset = BoneAgeDataset(self.directory, 'train', 224)
        loader = DataLoader(dataset, batch_size=2, num_workers=1,
                            multiprocessing_context='spawn', shuffle=False)
        batches = list(loader)
        self.assertEqual(len(batches), 1)
        self.assertEqual(list(batches[0][3]), ['101', '102'])
        self.assertEqual(batches[0][0].shape, (2, 1, 224, 224))

    def test_larger_supported_sizes_keep_holdout_partition(self):
        for size in (384, 512):
            with self.subTest(image_size=size):
                prepare_training_cache(self.directory, size)
                dataset = BoneAgeDataset(self.directory, 'test', size)
                self.assertEqual(list(dataset.ids), ['104'])
                image, _, _, source_id = dataset[0]
                self.assertEqual(image.shape, (1, size, size))
                direct, _ = letterbox(self.originals[source_id], size)
                np.testing.assert_array_equal(image[0].numpy(), direct)

    def test_cache_byte_corruption_is_rejected(self):
        prepare_training_cache(self.directory, 224)
        image_path = self.directory / 'training-cache-224' / 'images.npy'
        with image_path.open('r+b') as stream:
            stream.seek(-1, 2)
            original = stream.read(1)[0]
            stream.seek(-1, 2)
            stream.write(bytes([original ^ 1]))
        with self.assertRaises((ValueError, RuntimeError)):
            BoneAgeDataset(self.directory, 'train', 224)

    def test_prepared_cache_cannot_follow_a_changed_source_manifest(self):
        prepare_training_cache(self.directory, 224)
        self.records[0]['boneage'] += 1
        self.write_manifest()
        with self.assertRaises((ValueError, RuntimeError)):
            BoneAgeDataset(self.directory, 'train', 224)

    def test_prepared_index_must_match_manifest_even_with_updated_file_hash(self):
        prepare_training_cache(self.directory, 224)
        directory = self.directory / 'training-cache-224'
        path = directory / 'ids.json'
        metadata_path = directory / 'metadata.json'
        original_index = json.loads(path.read_text(encoding='utf-8'))
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        changes = (('ids', ['102', '101', '103', '104']),
                   ('splits', ['train', 'test', 'validation', 'train']))
        for field, values in changes:
            with self.subTest(field=field):
                changed = {**original_index, field: values}
                path.write_text(json.dumps(changed), encoding='utf-8')
                metadata['files']['ids.json'] = {
                    'sha256': sha256(path), 'bytes': path.stat().st_size}
                metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
                with self.assertRaises((ValueError, RuntimeError)):
                    BoneAgeDataset(self.directory, 'train', 224)

    def test_original_png_hash_mismatch_cannot_be_cached(self):
        self.records[0]['sha256'] = '0' * 64
        self.write_manifest()
        with self.assertRaises((ValueError, RuntimeError)):
            prepare_training_cache(self.directory, 224)
        self.assertFalse((self.directory / 'training-cache-224' / 'metadata.json').exists())

    def test_unsafe_archive_member_paths_are_rejected(self):
        for unsafe in ('../outside.png', 'C:\\outside.png', '..\\outside.png'):
            with self.subTest(path=unsafe):
                self.records[0]['archive_path'] = unsafe
                self.write_manifest()
                with self.assertRaises((ValueError, RuntimeError)):
                    prepare_training_cache(self.directory, 224)


if __name__ == '__main__':
    unittest.main()
