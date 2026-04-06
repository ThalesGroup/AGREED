"""
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016, 
All rights reserved.
"""

import torch
import torch.nn as nn
from typing import List, Union


ARCHITECTURES = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

class VGG(nn.Module):
    def __init__(
        self, num_classes: int = 1000, model_name: str = "VGG16", 
        init_weights: bool = True, dropout: float = 0.5, batch_norm: bool = True,
        adaptive_pool_size: int = 7, layerwidths: List[int] = [4096, 4096]
    ) -> None:
        super().__init__()
        self.features = make_layers(ARCHITECTURES[model_name], batch_norm=batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((adaptive_pool_size, adaptive_pool_size))
        self.classifier = nn.Sequential(
            nn.Linear(512 * adaptive_pool_size * adaptive_pool_size, layerwidths[0]),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(layerwidths[0], layerwidths[1]),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(layerwidths[1], num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(architecture: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in architecture:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = int(v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
