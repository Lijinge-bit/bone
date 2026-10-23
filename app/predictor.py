import hashlib
import json
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from app.imaging import PREPROCESS_VERSION, letterbox


class ModelUnavailable(RuntimeError):
    pass


class Predictor:
    def __init__(self, model_dir):
        self.directory = Path(model_dir)
        self.lock = threading.Lock()
        self.loaded_id = None
        self.model = None

    def metadata(self):
        path = self.directory / "active.json"
        if not path.exists():
            return None
        try:
            meta = json.loads(path.read_text())
            if not isinstance(meta, dict):
                return None
            required = {"model_id", "architecture", "checkpoint", "checkpoint_sha256",
                        "image_size", "preprocess_version", "evaluation"}
            if not required <= meta.keys():
                return None
            if not isinstance(meta["evaluation"], dict) or not isinstance(meta["image_size"], int):
                return None
            if not 64 <= meta["image_size"] <= 512:
                return None
            if meta["preprocess_version"] != PREPROCESS_VERSION:
                return None
            checkpoint = self.directory / meta["checkpoint"]
            if checkpoint.parent.resolve() != self.directory.resolve() or not checkpoint.is_file():
                return None
            return meta
        except (OSError, ValueError, TypeError):
            return None

    def assess(self, image, sex, expected_model_id, expected_sha):
        with self.lock:
            try:
                import torch
                from app.network import BoneAgeCNN, denormalize_target, normalize_image
            except ImportError as exc:
                raise ModelUnavailable("尚未安装 CPU 推理依赖") from exc
            meta = self.metadata()
            if not meta or meta["model_id"] != expected_model_id:
                raise ModelUnavailable("模型已变更或不可用，请重新提交")
            if meta["architecture"] != BoneAgeCNN.architecture:
                raise ModelUnavailable("模型结构与当前推理器不兼容")
            checkpoint = self.directory / meta["checkpoint"]
            actual_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if actual_sha != meta["checkpoint_sha256"] or actual_sha != expected_sha:
                raise ModelUnavailable("模型校验失败，已停止使用该权重")
            identity = (meta["model_id"], actual_sha)
            if self.loaded_id != identity:
                torch.set_num_threads(2)
                model = BoneAgeCNN()
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                model.load_state_dict(state, strict=True)
                model.eval()
                self.model, self.loaded_id = model, identity
            array, transform = letterbox(image, int(meta["image_size"]))
            tensor = normalize_image(torch.from_numpy(array).unsqueeze(0).unsqueeze(0))
            sex_tensor = torch.tensor([[1.0 if sex == "M" else 0.0]])
            with torch.enable_grad():
                features = self.model.features(tensor)
                features.retain_grad()
                prediction = denormalize_target(self.model.predict_from_features(features, sex_tensor))
                value = float(prediction.detach().item())
                if not np.isfinite(value) or value < 0 or value > 360:
                    raise RuntimeError("模型输出超出已配置的研究有效性检查范围，需人工处理")
                self.model.zero_grad(set_to_none=True)
                prediction.backward()
                weights = features.grad.mean(dim=(2, 3), keepdim=True)
                cam = (weights * features).sum(dim=1).relu().detach().numpy()[0]
            if float(cam.max() - cam.min()) < 1e-8:
                return value, None, "本次梯度关注图无有效变化，解释不可用"
            cam = (cam - cam.min()) / (cam.max() - cam.min())
            # Remove letterbox padding before mapping attention to the original PNG.
            size = int(meta["image_size"])
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
            return value, Image.fromarray(rgba), None
