import hashlib
import io
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError

PREPROCESS_VERSION = "gray-letterbox-v1"
IMAGENET_PREPROCESS_VERSION = "imagenet-gray-letterbox-v1"
QC_VERSION = "research-triage-v1"


def preprocess_version_for(architecture):
    """Return the preprocessing contract of a supported model architecture."""
    versions = {
        "boneage-small-cnn-v1": PREPROCESS_VERSION,
        "resnet50-sex-v1": IMAGENET_PREPROCESS_VERSION,
    }
    try:
        return versions[architecture]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unsupported model architecture: {architecture!r}") from exc


class ImageValidationError(ValueError):
    pass


def validate_png(data, max_pixels=25_000_000):
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ImageValidationError("首版仅支持 PNG；请提供原始 PNG 影像")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as im:
                if im.format != "PNG" or getattr(im, "n_frames", 1) != 1:
                    raise ImageValidationError("不支持动画或多帧 PNG")
                if im.width * im.height > max_pixels:
                    raise ImageValidationError("图片像素数量超过 2500 万限制")
                if min(im.size) < 64:
                    raise ImageValidationError("图片过小：宽高均需至少 64 像素")
                if im.mode != "L" or data[24] != 8:
                    raise ImageValidationError("当前模型支持 8 位单通道灰度 PNG；彩色和高位深图像暂未验证")
                im.verify()
            with Image.open(io.BytesIO(data)) as im:
                image = im.copy()
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageValidationError("PNG 损坏、编码不支持或无法安全解码") from exc
    return image


def quality_check(image):
    # Signals are research triage only; never delete an image or assert clinical quality.
    gray = np.asarray(image.resize((256, 256)), dtype=np.float32)
    p05, p50, p95 = np.percentile(gray, [5, 50, 95])
    metrics = {
        "median": round(float(p50), 2), "p05": round(float(p05), 2),
        "p95": round(float(p95), 2), "dynamic_range": round(float(p95 - p05), 2),
        "dark_fraction": round(float((gray < 20).mean()), 3),
        "bright_fraction": round(float((gray > 235).mean()), 3),
        "qc_version": QC_VERSION,
    }
    flags = []
    if p50 < 35:
        flags.append({"code": "dark", "label": "整体偏暗，请核对骨骼细节"})
    if p50 > 190:
        flags.append({"code": "bright", "label": "整体偏亮，请检查对比度"})
    if p95 - p05 < 65:
        flags.append({"code": "low_contrast", "label": "对比度较低，建议人工复核"})
    if metrics["dark_fraction"] > .60:
        flags.append({"code": "background", "label": "暗背景占比较大，请检查目标手部"})
    if min(image.size) < 512:
        flags.append({"code": "resolution", "label": "分辨率较低，注意解剖细节"})
    if max(image.size) / min(image.size) > 2.2:
        flags.append({"code": "aspect", "label": "构图较窄，请核对完整手部"})
    # Every imported image needs a human check for extra fingers/text/cropping:
    # these cannot be established by brightness statistics.
    return flags, metrics


def letterbox(image, size):
    image = image.convert("L")
    scale = min(size / image.width, size / image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    left, top = (size - width) // 2, (size - height) // 2
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(resized, (left, top))
    transform = {"left": left, "top": top, "width": width, "height": height,
                 "original_width": image.width, "original_height": image.height}
    return np.asarray(canvas).copy(), transform


def fingerprint(data):
    return hashlib.sha256(data).hexdigest()
