"""Adapter/contract tests use synthetic weights, never accuracy claims."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.imaging import letterbox, preprocess_version_for
from app.predictor import ModelUnavailable, Predictor

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TORCHVISION = importlib.util.find_spec("torchvision") is not None
if HAS_TORCH:
    import torch
    from app.network import (BoneAgeCNN, build_model, denormalize_target,
                             normalize_for_model, normalize_image)


def metadata_for(checkpoint, architecture="resnet50-sex-v1"):
    resnet = architecture == "resnet50-sex-v1"
    return {
        "schema_version": 2, "model_id": "synthetic-adapter-test", "architecture": architecture,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "image_size": 96, "preprocess_version": preprocess_version_for(architecture),
        "sex_encoding": {"M": 1, "F": 0},
        "target_normalization": {"offset": 120.0, "scale": 60.0},
        "input_channels": 3 if resnet else 1,
        "explanation_method": "Grad-CAM",
        "gradcam_layer": "features.layer4" if resnet else "features",
        "evaluation": {"synthetic_test_only": True, "accuracy_not_evaluated": True},
    }


@unittest.skipUnless(HAS_TORCH, "Optional torch dependency is not installed")
class PreprocessAndLegacyAdapterTests(unittest.TestCase):
    def test_historical_preprocessing_is_exact(self):
        image = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 16, 16)
        expected = image.float().div(255).sub(.5).div(.5)
        self.assertTrue(torch.equal(normalize_image(image), expected))
        self.assertTrue(torch.equal(normalize_for_model(image, "gray-letterbox-v1"), expected))
        self.assertEqual(preprocess_version_for(BoneAgeCNN.architecture), "gray-letterbox-v1")
        self.assertTrue(torch.equal(denormalize_target(torch.tensor([-2., 0., 4.])),
                                    torch.tensor([0., 120., 360.])))

    def test_imagenet_replicates_gray_before_per_channel_normalization(self):
        image = torch.tensor([0, 128, 255], dtype=torch.uint8).reshape(1, 1, 1, 3)
        normalized = normalize_for_model(image, "imagenet-gray-letterbox-v1")
        self.assertEqual(tuple(normalized.shape), (1, 3, 1, 3))
        for channel, (mean, std) in enumerate(zip((.485, .456, .406), (.229, .224, .225))):
            expected = (image.float().div(255)[:, 0] - mean) / std
            torch.testing.assert_close(normalized[:, channel], expected, rtol=0, atol=0)

    def test_unknown_or_malformed_preprocessing_is_rejected(self):
        image = torch.zeros((1, 1, 64, 64), dtype=torch.uint8)
        for tensor in (image.float(), image[:, 0], image.repeat(1, 3, 1, 1), image[:0]):
            with self.subTest(shape=tensor.shape, dtype=tensor.dtype), self.assertRaises(ValueError):
                normalize_for_model(tensor, "gray-letterbox-v1")
        with self.assertRaises(ValueError):
            normalize_for_model(image, "unknown")
        with self.assertRaises(ValueError):
            build_model("unknown")

    def test_old_state_dict_keys_and_roundtrip_are_unchanged(self):
        torch.manual_seed(10)
        direct = BoneAgeCNN().eval()
        expected_keys = {
            "features.0.weight", "features.0.bias", "features.1.weight", "features.1.bias",
            "features.3.weight", "features.3.bias", "features.4.weight", "features.4.bias",
            "features.6.weight", "features.6.bias", "features.7.weight", "features.7.bias",
            "sex_embedding.0.weight", "sex_embedding.0.bias", "head.0.weight", "head.0.bias",
            "head.3.weight", "head.3.bias",
        }
        self.assertEqual(set(direct.state_dict()), expected_keys)
        image, sex = torch.randn(2, 1, 64, 64), torch.tensor([[0.], [1.]])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.pt"
            torch.save(direct.state_dict(), path)
            restored = build_model(BoneAgeCNN.architecture, weights_path=path).eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(image, sex), direct(image, sex), rtol=0, atol=0)


@unittest.skipUnless(HAS_TORCH and HAS_TORCHVISION, "Optional torch/torchvision dependencies are not installed")
class ResNetAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(29)
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        # Explicitly synthetic, controlled regressor weights produce bounded
        # outputs and a positive, nonconstant gradient through genuine ResNet50.
        model = build_model("resnet50-sex-v1").eval()
        with torch.no_grad():
            for parameter in model.head.parameters():
                parameter.zero_()
            model.head[0].weight[0, :2048] = 1 / 2048
            model.head[0].weight[0, 2048] = .2
            model.head[0].bias[0] = .5
            model.head[3].weight[0, 0] = .01
            model.sex_embedding[0].weight.zero_()
            model.sex_embedding[0].bias.zero_()
            model.sex_embedding[0].weight[0, 0] = 1.
        cls.checkpoint = cls.directory / "model.pt"
        torch.save(model.state_dict(), cls.checkpoint)
        cls.meta = metadata_for(cls.checkpoint)
        (cls.directory / "candidate.json").write_text(json.dumps(cls.meta), encoding="utf-8")
        cls.predictor = Predictor(cls.directory, metadata_filename="candidate.json")
        rng = np.random.default_rng(31)
        cls.image = Image.fromarray(rng.integers(0, 256, (143, 83), dtype=np.uint8))
        cls.first = cls.predictor.assess(cls.image, "F", cls.meta["model_id"], cls.meta["checkpoint_sha256"])

    @classmethod
    def tearDownClass(cls):
        del cls.predictor
        cls.temp.cleanup()

    def assess(self, sex="F"):
        return self.predictor.assess(self.image, sex, self.meta["model_id"], self.meta["checkpoint_sha256"])

    def test_resnet_forward_contract_and_supported_sizes(self):
        model = self.predictor.model
        self.assertIsNotNone(model.get_submodule("features.layer4"))
        self.assertEqual({parameter.device.type for parameter in model.parameters()}, {"cpu"})
        with torch.inference_mode():
            for size in (64, 512):
                image = normalize_for_model(torch.zeros(1, 1, size, size, dtype=torch.uint8),
                                            self.meta["preprocess_version"])
                features = model.features(image)
                sex = torch.tensor([[1.]])
                self.assertEqual(tuple(features.shape), (1, 2048, size // 32, size // 32))
                self.assertEqual(tuple(model(image, sex).shape), (1,))
                torch.testing.assert_close(model(image, sex), model.predict_from_features(features, sex))
        self.assertNotEqual(self.assess("F")[0], self.assess("M")[0])

    def test_real_gradcam_is_repeatable_and_crops_letterbox(self):
        value, heatmap, error = self.first
        repeated, repeated_heatmap, repeated_error = self.assess()
        self.assertIsNone(error)
        self.assertIsNone(repeated_error)
        self.assertEqual(value, repeated)
        self.assertTrue(0 <= value <= 360)
        self.assertEqual(heatmap.mode, "RGBA")
        self.assertEqual(heatmap.size, self.image.size)
        self.assertTrue(np.array_equal(np.asarray(heatmap), np.asarray(repeated_heatmap)))
        self.assertGreater(np.ptp(np.asarray(heatmap)[:, :, 3]), 0)
        model = self.predictor.model
        array, transform = letterbox(self.image, self.meta["image_size"])
        image = normalize_for_model(torch.from_numpy(array)[None, None], self.meta["preprocess_version"])
        features = model.features(image)
        features.retain_grad()
        prediction = denormalize_target(model.predict_from_features(features, torch.tensor([[0.]])))
        model.zero_grad(set_to_none=True)
        prediction.backward()
        cam = (features.grad.mean((2, 3), keepdim=True) * features).sum(1).relu().detach().numpy()[0]
        cam = (cam - cam.min()) / np.ptp(cam)
        independently_mapped = Image.fromarray(np.uint8(cam * 255)).resize((96, 96), Image.Resampling.BILINEAR)
        x, y, width, height = (transform[key] for key in ("left", "top", "width", "height"))
        independently_mapped = independently_mapped.crop((x, y, x + width, y + height))
        independently_mapped = independently_mapped.resize(self.image.size, Image.Resampling.BILINEAR)
        alpha = np.uint8(np.asarray(independently_mapped, dtype=np.float32) / 255 * 160)
        self.assertTrue(np.array_equal(np.asarray(heatmap)[:, :, 3], alpha))

    def test_partial_success_when_explanation_fails_or_is_empty(self):
        with patch.object(self.predictor, "_render_heatmap", side_effect=RuntimeError("synthetic rendering failure")):
            value, heatmap, error = self.assess()
        self.assertEqual(value, self.first[0])
        self.assertIsNone(heatmap)
        self.assertIn("解释不可用", error)
        original = self.predictor.model.predict_from_features
        with patch.object(self.predictor.model, "predict_from_features", side_effect=lambda f, s: original(f * 0, s)):
            value, heatmap, error = self.assess()
        self.assertTrue(0 <= value <= 360)
        self.assertIsNone(heatmap)
        self.assertIn("无有效变化", error)

    def test_invalid_output_and_sex_are_rejected(self):
        for sex in ("m", "female", "", None, 1):
            with self.subTest(sex=sex), self.assertRaises(ValueError):
                self.assess(sex)
        for value in (float("nan"), float("inf"), -2.01, 4.01):
            with self.subTest(output=value), patch.object(self.predictor.model, "predict_from_features", return_value=torch.tensor([value])):
                with self.assertRaisesRegex(RuntimeError, "有效性检查"):
                    self.assess()

    def test_official_pretraining_is_explicit_and_cpu_loading_never_downloads(self):
        from torchvision.models import ResNet50_Weights, resnet50
        backbone = resnet50(weights=None)
        with patch("torchvision.models.resnet50", return_value=backbone) as constructor:
            build_model("resnet50-sex-v1", pretrained=True)
            constructor.assert_called_once_with(weights=ResNet50_Weights.DEFAULT)
        del backbone
        with patch("torchvision.models._api.load_state_dict_from_url", side_effect=AssertionError("Unexpected download")):
            model = build_model("resnet50-sex-v1", pretrained=True, weights_path=self.checkpoint)
        self.assertEqual(next(model.parameters()).device.type, "cpu")
        self.assertEqual(model.architecture, "resnet50-sex-v1")

    def test_metadata_contract_rejects_mismatch_but_accepts_legacy(self):
        changes = [
            {"architecture": "unknown"}, {"image_size": 63}, {"image_size": 513}, {"image_size": True},
            {"preprocess_version": "gray-letterbox-v1"}, {"input_channels": 1},
            {"sex_encoding": {"M": 0, "F": 1}}, {"sex_encoding": {"M": True, "F": False}},
            {"target_normalization": {"offset": 0, "scale": 60}},
            {"explanation_method": "attention"}, {"gradcam_layer": "features.layer3"},
            {"schema_version": 1}, {"schema_version": True}, {"checkpoint_sha256": "bad"},
            {"checkpoint": "../model.pt"}, {"checkpoint": "/model.pt"}, {"checkpoint": "..\\model.pt"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            os.link(self.checkpoint, directory / self.checkpoint.name)
            predictor = Predictor(directory, metadata_filename="candidate.json")
            for change in changes:
                meta = dict(self.meta, **change)
                (directory / "candidate.json").write_text(json.dumps(meta))
                with self.subTest(change=change):
                    self.assertIsNone(predictor.metadata())
            for field in ("sex_encoding", "target_normalization", "input_channels", "explanation_method", "gradcam_layer"):
                meta = dict(self.meta)
                del meta[field]
                (directory / "candidate.json").write_text(json.dumps(meta))
                with self.subTest(missing=field):
                    self.assertIsNone(predictor.metadata())
            meta = metadata_for(self.checkpoint, "boneage-small-cnn-v1")
            for field in ("schema_version", "sex_encoding", "target_normalization", "input_channels", "explanation_method", "gradcam_layer"):
                del meta[field]
            (directory / "active.json").write_text(json.dumps(meta))
            self.assertEqual(Predictor(directory).metadata(), meta)
            # Selecting a candidate never implicitly changes active metadata.
            self.assertNotEqual(Predictor(directory).metadata()["architecture"], self.meta["architecture"])

    def test_corrupt_incompatible_and_unsafe_checkpoints_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            checkpoint = directory / "model.pt"
            meta = dict(self.meta)
            checkpoint.write_bytes(b"corrupt synthetic checkpoint")
            (directory / "candidate.json").write_text(json.dumps(meta))
            predictor = Predictor(directory, metadata_filename="candidate.json")
            with self.assertRaisesRegex(ModelUnavailable, "校验失败"):
                predictor.assess(self.image, "F", meta["model_id"], meta["checkpoint_sha256"])
            # A matching digest does not make an incompatible payload loadable.
            for state in ({"unknown": torch.zeros(1)}, {"unsafe": Path("arbitrary-object")}):
                torch.save(state, checkpoint)
                meta["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                (directory / "candidate.json").write_text(json.dumps(meta))
                with self.subTest(state=list(state)), self.assertRaisesRegex(ModelUnavailable, "安全加载"):
                    predictor.assess(self.image, "F", meta["model_id"], meta["checkpoint_sha256"])

    def test_path_escape_and_stale_request_are_rejected(self):
        for filename in ("../candidate.json", "/tmp/candidate.json", "..\\candidate.json"):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                Predictor(self.directory, metadata_filename=filename)
        with self.assertRaises(ModelUnavailable):
            self.predictor.assess(self.image, "F", "stale-id", self.meta["checkpoint_sha256"])
        with self.assertRaisesRegex(ModelUnavailable, "校验失败"):
            self.predictor.assess(self.image, "F", self.meta["model_id"], "0" * 64)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "model.pt").symlink_to(self.checkpoint)
            (directory / "candidate.json").write_text(json.dumps(self.meta))
            self.assertIsNone(Predictor(directory, metadata_filename="candidate.json").metadata())
            (directory / "linked.json").symlink_to(self.directory / "candidate.json")
            self.assertIsNone(Predictor(directory, metadata_filename="linked.json").metadata())

    def test_metadata_changes_invalidate_cached_model_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            os.link(self.checkpoint, directory / self.checkpoint.name)
            metadata = copy.deepcopy(self.meta)
            path = directory / "candidate.json"
            path.write_text(json.dumps(metadata))
            predictor = Predictor(directory, metadata_filename="candidate.json")
            predictor.assess(self.image, "F", metadata["model_id"], metadata["checkpoint_sha256"])
            identity = predictor.loaded_id
            metadata["image_size"] = 128
            path.write_text(json.dumps(metadata))
            predictor.assess(self.image, "F", metadata["model_id"], metadata["checkpoint_sha256"])
            self.assertNotEqual(predictor.loaded_id, identity)


if __name__ == "__main__":
    unittest.main()
