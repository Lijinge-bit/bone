"""A small CPU baseline; target labels and IDs are never model inputs."""
import torch
from torch import nn


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


def normalize_image(image_tensor):
    return image_tensor.float().div(255).sub(.5).div(.5)


def denormalize_target(output):
    return output * 60.0 + 120.0
