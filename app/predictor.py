"""Metadata-checked CPU inference for active and explicitly selected candidates."""
import hashlib
import io
import json
import re
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from app.imaging import letterbox, preprocess_version_for


class ModelUnavailable(RuntimeError):
    pass


_CONTRACTS = {
    "boneage-small-cnn-v1": {"input_channels": 1, "gradcam_layer": "features"},
    "resnet50-sex-v1": {"input_channels": 3, "gradcam_layer": "features.layer4"},
}
_SEX_ENCODING = {"M": 1, "F": 0}
_TARGET_NORMALIZATION = {"offset": 120.0, "scale": 60.0}


def _plain_filename(value):
    return (isinstance(value, str) and bool(value) and value not in (".", "..")
            and "/" not in value and "\\" not in value and "\x00" not in value
            and Path(value).name == value)


class Predictor:
    def __init__(self, model_dir, metadata_filename="active.json"):
        if not _plain_filename(metadata_filename):
            raise ValueError("Model metadata must be a filename inside the model directory")
        self.directory = Path(model_dir)
        self.metadata_filename = metadata_filename
        self.lock = threading.Lock()
        self.loaded_id = None
        self.model = None

    def metadata(self):
        path = self.directory / self.metadata_filename
        try:
            directory = self.directory.resolve()
            if not path.is_file() or path.resolve().parent != directory:
                return None
            meta = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                return None
            required = {"model_id", "architecture", "checkpoint", "checkpoint_sha256",
                        "image_size", "preprocess_version", "evaluation"}
            if not required <= meta.keys():
                return None
            if not isinstance(meta["model_id"], str) or not meta["model_id"].strip():
                return None
            if not isinstance(meta["architecture"], str) or meta["architecture"] not in _CONTRACTS:
                return None
            if not isinstance(meta["evaluation"], dict) or type(meta["image_size"]) is not int:
                return None
            if not 64 <= meta["image_size"] <= 512:
                return None
            if meta["preprocess_version"] != preprocess_version_for(meta["architecture"]):
                return None
            if not isinstance(meta["checkpoint_sha256"], str) or not re.fullmatch(
                    r"[0-9a-f]{64}", meta["checkpoint_sha256"]):
                return None
            if not _plain_filename(meta["checkpoint"]):
                return None
            checkpoint = self.directory / meta["checkpoint"]
            if checkpoint.resolve().parent != directory or not checkpoint.is_file():
                return None
            schema = meta.get("schema_version", 1)
            if type(schema) is not int or schema not in (1, 2):
                return None
            fields = {"sex_encoding", "target_normalization", "input_channels",
                      "explanation_method", "gradcam_layer"}
            if schema == 2 and not fields <= meta.keys():
                return None
            # New adapters must declare their complete contract. Historical CNN
            # metadata remains valid without the fields introduced in schema 2.
            if meta["architecture"] != "boneage-small-cnn-v1" and schema != 2:
                return None
            if "sex_encoding" in meta:
                encoding = meta["sex_encoding"]
                if (not isinstance(encoding, dict) or encoding != _SEX_ENCODING
                        or any(type(value) is not int for value in encoding.values())):
                    return None
            if "target_normalization" in meta:
                normalization = meta["target_normalization"]
                if (not isinstance(normalization, dict) or normalization != _TARGET_NORMALIZATION
                        or any(type(value) not in (int, float) for value in normalization.values())):
                    return None
            contract = _CONTRACTS[meta["architecture"]]
            if "input_channels" in meta and (type(meta["input_channels"]) is not int
                    or meta["input_channels"] != contract["input_channels"]):
                return None
            if "explanation_method" in meta and meta["explanation_method"] != "Grad-CAM":
                return None
            if "gradcam_layer" in meta and meta["gradcam_layer"] != contract["gradcam_layer"]:
                return None
            return meta
        except (OSError, ValueError, TypeError, RuntimeError):
            return None

    @staticmethod
    def _render_heatmap(cam, image, size, transform):
        # Keep the historical interpolation, padding removal, and RGBA mapping
        # exactly: runtime reference maps depend on these numerical operations.
        heat = Image.fromarray(np.uint8(cam * 255)).resize((size, size), Image.Resampling.BILINEAR)
        left, top = transform["left"], transform["top"]
        heat = heat.crop((left, top, left + transform["width"], top + transform["height"]))
        heat = heat.resize(image.size, Image.Resampling.BILINEAR)
        intensity = np.asarray(heat, dtype=np.float32) / 255
        rgba = np.zeros((image.height, image.width, 4), dtype=np.uint8)
        rgba[:, :, 0] = np.uint8(np.clip(1.5 * intensity, 0, 1) * 255)
        rgba[:, :, 1] = np.uint8(np.clip(1.5 - abs(intensity * 2 - 1), 0, 1) * 160)
        rgba[:, :, 2] = np.uint8((1 - intensity) * 220)
        rgba[:, :, 3] = np.uint8(intensity * 160)
        return Image.fromarray(rgba)

    def assess(self, image, sex, expected_model_id, expected_sha):
        if not isinstance(sex, str) or sex not in _SEX_ENCODING:
            raise ValueError("sex 必须为 F 或 M")
        with self.lock:
            try:
                import torch
                from app.network import build_model, denormalize_target, normalize_for_model
            except ImportError as exc:
                raise ModelUnavailable("尚未安装 CPU 推理依赖") from exc
            meta = self.metadata()
            if not meta or meta["model_id"] != expected_model_id:
                raise ModelUnavailable("模型已变更或不可用，请重新提交")
            checkpoint = self.directory / meta["checkpoint"]
            try:
                checkpoint_data = checkpoint.read_bytes()
            except OSError as exc:
                raise ModelUnavailable("模型权重不可读，请重新提交") from exc
            actual_sha = hashlib.sha256(checkpoint_data).hexdigest()
            if actual_sha != meta["checkpoint_sha256"] or actual_sha != expected_sha:
                raise ModelUnavailable("模型校验失败，已停止使用该权重")
            metadata_sha = hashlib.sha256(json.dumps(
                meta, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")).hexdigest()
            identity = (meta["model_id"], actual_sha, metadata_sha)
            if self.loaded_id != identity:
                try:
                    torch.set_num_threads(2)
                    model = build_model(meta["architecture"], pretrained=False)
                    # Deserialize the exact bytes whose digest was checked,
                    # rather than reopening a potentially replaced checkpoint.
                    state = torch.load(io.BytesIO(checkpoint_data), map_location="cpu", weights_only=True)
                    model.load_state_dict(state, strict=True)
                    model.eval()
                except Exception as exc:
                    raise ModelUnavailable("模型权重无法安全加载或与声明结构不兼容") from exc
                self.model, self.loaded_id = model, identity
            del checkpoint_data
            array, transform = letterbox(image, int(meta["image_size"]))
            tensor = normalize_for_model(torch.from_numpy(array).unsqueeze(0).unsqueeze(0),
                                         meta["preprocess_version"])
            sex_tensor = torch.tensor([[float(_SEX_ENCODING[sex])]])
            with torch.enable_grad():
                features = self.model.features(tensor)
                prediction = denormalize_target(self.model.predict_from_features(features, sex_tensor))
                value = float(prediction.detach().item())
                if not np.isfinite(value) or value < 0 or value > 360:
                    raise RuntimeError("模型输出超出已配置的研究有效性检查范围，需人工处理")
                try:
                    features.retain_grad()
                    self.model.zero_grad(set_to_none=True)
                    prediction.backward()
                    if features.grad is None:
                        raise RuntimeError("Missing feature gradients")
                    weights = features.grad.mean(dim=(2, 3), keepdim=True)
                    cam = (weights * features).sum(dim=1).relu().detach().numpy()[0]
                    if not np.isfinite(cam).all():
                        raise RuntimeError("Non-finite attention map")
                    if float(cam.max() - cam.min()) < 1e-8:
                        return value, None, "本次梯度关注图无有效变化，解释不可用"
                    cam = (cam - cam.min()) / (cam.max() - cam.min())
                    return value, self._render_heatmap(cam, image, int(meta["image_size"]), transform), None
                except Exception:
                    # A valid scalar prediction remains useful when attention
                    # rendering/backprop fails; callers expose partial success.
                    return value, None, "本次梯度关注图生成失败，解释不可用"
