import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b3,
    EfficientNet_B3_Weights
)


class MelanomaClassifier(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        if pretrained:
            weights = EfficientNet_B3_Weights.DEFAULT
        else:
            weights = None

        self.model = efficientnet_b3(weights=weights)

        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.model(x)