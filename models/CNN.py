import torch
import torch.nn as nn
import torch.nn.functional as F
from models.binary_basic import *
from models.quant_basic import *

class CNN_MNIST_V1(nn.Module):
    """
        V1 CNN model for MNIST, offline version
    """
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32, affine=True),
                nn.ReLU()
            ),
            # (32,28,28)
            nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32, affine=True),
                nn.ReLU(),
                nn.AvgPool2d(2,2)
            ),
            # (32,14,14)
            nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64, affine=True),
                nn.ReLU()
            ),
            # (64,14,14)
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64, affine=True),
                nn.ReLU(),
                nn.AvgPool2d(2,2)
            ),
            # (64,7,7)
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128, affine=True),
                nn.ReLU()
            ),
            # (128,7,7)
            nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128, affine=True),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            # (128,1,1)
        )
        self.classifiers = nn.Sequential(
            nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(128, 10, bias=True)
            )
        )

    def forward(self, x:torch.Tensor):
        return self.classifiers(self.features(x))