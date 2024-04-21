import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hedge import NN_Online

class CNN_online_MNIST_V1(NN_Online):
    """
        V1 CNN model for MNIST
    """
    
    def _module_compose(self):
        features = [
            nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1, bias=False),
                nn.InstanceNorm2d(32, affine=True),
                nn.ReLU()
            ),
            # (32,28,28)
            nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1, bias=False),
                nn.InstanceNorm2d(32, affine=True),
                nn.ReLU(),
                nn.AvgPool2d(2,2)
            ),
            # (32,14,14)
            nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1, bias=False),
                nn.InstanceNorm2d(64, affine=True),
                nn.ReLU()
            ),
            # (64,14,14)
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1, bias=False),
                nn.InstanceNorm2d(64, affine=True),
                nn.ReLU(),
                nn.AvgPool2d(2,2)
            ),
            # (64,7,7)
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1, bias=False),
                nn.InstanceNorm2d(128, affine=True),
                nn.ReLU()
            ),
            # (128,7,7)
            nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1, bias=False),
                nn.InstanceNorm2d(128, affine=True),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            # (128,1,1)
        ]
        classifiers = [
            nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(32*28*28, 10, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(32*14*14, 10, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(64*14*14, 10, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(64*7*7, 10, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(128*7*7, 10, bias=False)
            ),
            nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(128, 10, bias=True)
            ),
        ]
        return nn.ModuleList(features), nn.ModuleList(classifiers)