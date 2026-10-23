"""Bone age model adapters; target labels and IDs are never model inputs."""
from collections import OrderedDict

import torch
from torch import nn

from app.imaging import (IMAGENET_PREPROCESS_VERSION, PREPROCESS_VERSION,
                         preprocess_version_for)


class BoneAgeCNN(nn.Module):
    architecture = "boneage-small-cnn-v1"

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.sex_embedding = nn.Sequential(nn.Linear(1, 8), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(64 * 4 * 4 + 8, 128), nn.SiLU(),
                                  nn.Dropout(.15), nn.Linear(128, 1))

    def predict_from_features(self, features, sex):
        return self.head(torch.cat((self.pool(features).flatten(1),
                                    self.sex_embedding(sex)), dim=1)).squeeze(1)

    def forward(self, image, sex):
        return self.predict_from_features(self.features(image), sex)


class ResNet50Sex(nn.Module):
    """Official torchvision ResNet50 features and a sex-conditioned regressor.

    ``pretrained=False`` never downloads weights.  Only an explicit training
    request for ``pretrained=True`` enables torchvision's official weights.
    """
    architecture = "resnet50-sex-v1"

    def __init__(self, pretrained=False):
        super().__init__()
        # Keep torchvision optional for the original small CNN and old weights.
        from torchvision.models import ResNet50_Weights, resnet50

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.features = nn.Sequential(OrderedDict(
            (name, getattr(backbone, name)) for name in
            ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")
        ))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sex_embedding = nn.Sequential(nn.Linear(1, 32), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(2048 + 32, 256), nn.SiLU(),
                                  nn.Dropout(.2), nn.Linear(256, 1))

    def predict_from_features(self, features, sex):
        return self.head(torch.cat((self.pool(features).flatten(1),
                                    self.sex_embedding(sex)), dim=1)).squeeze(1)

    def forward(self, image, sex):
        return self.predict_from_features(self.features(image), sex)


def build_model(architecture, pretrained=False, weights_path=None):
    """Build an adapter and optionally load its complete, strict CPU state dict.

    Local adapter weights take precedence over pretrained initialization, so
    loading a checkpoint cannot trigger an unnecessary network download.
    """
    preprocess_version_for(architecture)  # Reject unknown architectures early.
    if architecture == BoneAgeCNN.architecture:
        if pretrained and weights_path is None:
            raise ValueError("The small CNN has no official pretrained weights")
        model = BoneAgeCNN()
    else:
        model = ResNet50Sex(pretrained=bool(pretrained and weights_path is None))
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    return model


def normalize_image(image_tensor):
    return image_tensor.float().div(255).sub(.5).div(.5)


def normalize_for_model(image_tensor, preprocess_version):
    """Normalize letterboxed, uint8 N1HW images using an explicit contract."""
    if not isinstance(image_tensor, torch.Tensor) or image_tensor.dtype != torch.uint8:
        raise ValueError("Model images must be a uint8 N1HW tensor")
    if image_tensor.ndim != 4 or image_tensor.shape[1] != 1 or min(image_tensor.shape) < 1:
        raise ValueError("Model images must be a nonempty uint8 N1HW tensor")
    if preprocess_version == PREPROCESS_VERSION:
        # Preserve operation order and rounding of the historical model.
        return normalize_image(image_tensor)
    if preprocess_version == IMAGENET_PREPROCESS_VERSION:
        image = image_tensor.float().div(255).repeat(1, 3, 1, 1)
        mean = image.new_tensor((.485, .456, .406)).view(1, 3, 1, 1)
        std = image.new_tensor((.229, .224, .225)).view(1, 3, 1, 1)
        return image.sub(mean).div(std)
    raise ValueError(f"Unsupported preprocessing version: {preprocess_version!r}")


def denormalize_target(output):
    return output * 60.0 + 120.0
