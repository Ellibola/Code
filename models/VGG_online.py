import torch
import torch.nn as nn
import torch.nn.functional as F
OL_TYPE = 'exp3_jump_avg'
if OL_TYPE=='hedge':
    from models.hedge import NN_Online
elif OL_TYPE=='eg':
    from models.eg import NN_Online
elif OL_TYPE=='exp3':
    from models.exp3 import NN_Online
elif OL_TYPE=='exp3_avg':
    from models.exp3_avg import NN_Online
elif OL_TYPE=='exp3_jump':
    from models.exp3_jump import NN_Online
elif OL_TYPE=='exp3_jump_avg':
    from models.exp3_jump_avg import NN_Online
elif OL_TYPE=='plain_ol':
    from models.plain_ol import NN_Online
else:
    raise NotImplementedError
from models.binary_basic import *
from models.quant_basic import *
from models.ExpNorm import *

class VGG_c100_online(NN_Online):
    """
        Full precision VGG-11 for cifar-100 online learning
    """
    def _module_compose(self):
        features = [
            # (32, 32, 3)
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(64, affine=True),
                nn.ReLU(),
            ),
            # (32, 32, 64)
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(64, affine=True),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ),
            # (16, 16, 64)
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(128, affine=True),
                nn.ReLU(),
            ),
            # (16, 16, 128)
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(128, affine=True),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ),
            # (8, 8, 128)
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(256, affine=True),
                nn.ReLU(),
            ),
            # (8, 8, 256)
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(256, affine=True),
                nn.ReLU(),
            ),
            # (8, 8, 256)
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(256, affine=True),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ),
            # (4, 4, 256)
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(512, affine=True),
                nn.ReLU(),
            ),
            # (4, 4, 512)
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(512, affine=True),
                nn.ReLU(),
            ),
            # (4, 4, 512)
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(512, affine=True),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            ),
        ]
        classifiers = [
            # (32, 32, 64)
            nn.Sequential(
                nn.AvgPool2d(8,8),
                nn.Flatten(),
                nn.Linear(1024, 100, bias=False),
                nn.LayerNorm(100)
            ),
            # (16, 16, 64)
            nn.Sequential(
                nn.AvgPool2d(8,8),
                nn.Flatten(),
                nn.Linear(256, 100, bias=False),
                nn.LayerNorm(100)
            ),
            # (16, 16, 128)
            nn.Sequential(
                nn.AvgPool2d(8,8),
                nn.Flatten(),
                nn.Linear(512, 100, bias=False),
                nn.LayerNorm(100)
            ),
            # (8, 8, 128)
            nn.Sequential(
                nn.AvgPool2d(4,4),
                nn.Flatten(),
                nn.Linear(512, 100, bias=False),
                nn.LayerNorm(100)
            ),
            # (8, 8, 256)
            nn.Sequential(
                nn.AvgPool2d(4,4),
                nn.Flatten(),
                nn.Linear(1024, 100, bias=False),
                nn.LayerNorm(100)
            ),
            # (8, 8, 256)
            nn.Sequential(
                nn.AvgPool2d(4,4),
                nn.Flatten(),
                nn.Linear(1024, 100, bias=False),
                nn.LayerNorm(100)
            ),
            # (4, 4, 256)
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, 100, bias=False),
                nn.LayerNorm(100)
            ),
            # (4, 4, 512)
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 100, bias=False),
                nn.LayerNorm(100)
            ),
            # (4, 4, 512)
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 100, bias=False),
                nn.LayerNorm(100)
            ),
            # (4, 4, 512)
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 100, bias=False),
                nn.LayerNorm(100)
            )
        ]
        return nn.ModuleList(features), nn.ModuleList(classifiers)
    
class VGG_c100_Quant_online(nn.Module):
    """
        Quantized VGG-11 for cifar-100
    """
    def _module_compose(self):
        features = [
            # (32, 32, 3)
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(64, affine=True),
                nn.ReLU(),
            ),
            # (32, 32, 64)
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(64, affine=True),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ),
            # (16, 16, 64)
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(128, affine=True),
                nn.ReLU(),
            ),
            # (16, 16, 128)
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(128, affine=True),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ),
            # (8, 8, 128)
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(256, affine=True),
                nn.ReLU(),
            ),
            # (8, 8, 256)
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(256, affine=True),
                nn.ReLU(),
            ),
            # (8, 8, 256)
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(256, affine=True),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ),
            # (4, 4, 256)
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(512, affine=True),
                nn.ReLU(),
            ),
            # (4, 4, 512)
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(512, affine=True),
                nn.ReLU(),
            ),
            # (4, 4, 512)
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(512, affine=True),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            ),
        ]
        classifiers = [
            # (32, 32, 64)
            nn.Sequential(
                nn.AvgPool2d(8,8),
                nn.Flatten(),
                nn.Linear(1024, 100, bias=False),
                nn.LayerNorm(100)
            ),
            # (16, 16, 64)
            nn.Sequential(
                nn.AvgPool2d(8,8),
                nn.Flatten(),
                nn.Linear(256, 100, bias=False),
                nn.LayerNorm(100)
            ),
            # (16, 16, 128)
            nn.Sequential(
                nn.AvgPool2d(8,8),
                nn.Flatten(),
                nn.Linear(512, 100, bias=False),
                nn.LayerNorm(100)
            ),
            # (8, 8, 128)
            nn.Sequential(
                nn.AvgPool2d(4,4),
                nn.Flatten(),
                nn.Linear(512, 100, bias=False),
                nn.LayerNorm(100)
            ),
            # (8, 8, 256)
            nn.Sequential(
                nn.AvgPool2d(4,4),
                nn.Flatten(),
                nn.Linear(1024, 100, bias=False),
                nn.LayerNorm(100)
            ),
            # (8, 8, 256)
            nn.Sequential(
                nn.AvgPool2d(4,4),
                nn.Flatten(),
                nn.Linear(1024, 100, bias=False),
                nn.LayerNorm(100)
            ),
            # (4, 4, 256)
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, 100, bias=False),
                nn.LayerNorm(100)
            ),
            # (4, 4, 512)
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 100, bias=False),
                nn.LayerNorm(100)
            ),
            # (4, 4, 512)
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 100, bias=False),
                nn.LayerNorm(100)
            ),
            # (4, 4, 512)
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 100, bias=False),
                nn.LayerNorm(100)
            )
        ]
        return nn.ModuleList(features), nn.ModuleList(classifiers)
    

class VGG_c10_online_plain(NN_Online):
    """
        Full precision VGG-11 for cifar-10 online learning
    """
    def _module_compose(self):
        features = [
            # (32, 32, 3)
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(64, affine=True),
                nn.ReLU(),
            ),
            # (32, 32, 64)
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(64, affine=True),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ),
            # (16, 16, 64)
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(128, affine=True),
                nn.ReLU(),
            ),
            # (16, 16, 128)
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(128, affine=True),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ),
            # (8, 8, 128)
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(256, affine=True),
                nn.ReLU(),
            ),
            # (8, 8, 256)
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(256, affine=True),
                nn.ReLU(),
            ),
            # (8, 8, 256)
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(256, affine=True),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ),
            # (4, 4, 256)
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(512, affine=True),
                nn.ReLU(),
            ),
            # (4, 4, 512)
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(512, affine=True),
                nn.ReLU(),
            ),
            # (4, 4, 512)
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(512, affine=True),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            ),
        ]
        classifiers = [
            # (4, 4, 512)
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 10, bias=False),
                nn.LayerNorm(10)
            )
        ]
        return nn.ModuleList(features), nn.ModuleList(classifiers)
    
class VGG_c10_online(NN_Online):
    """
        Full precision VGG-11 for cifar-10 online learning
    """
    def _module_compose(self):
        features = [
            # (32, 32, 3)
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(64, affine=True),
                nn.ReLU(),
            ),
            # (32, 32, 64)
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(64, affine=True),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ),
            # (16, 16, 64)
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(128, affine=True),
                nn.ReLU(),
            ),
            # (16, 16, 128)
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(128, affine=True),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ),
            # (8, 8, 128)
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(256, affine=True),
                nn.ReLU(),
            ),
            # (8, 8, 256)
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(256, affine=True),
                nn.ReLU(),
            ),
            # (8, 8, 256)
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(256, affine=True),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ),
            # (4, 4, 256)
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(512, affine=True),
                nn.ReLU(),
            ),
            # (4, 4, 512)
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(512, affine=True),
                nn.ReLU(),
            ),
            # (4, 4, 512)
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(512, affine=True),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            ),
        ]
        classifiers = [
            # (32, 32, 64)
            nn.Sequential(
                nn.AvgPool2d(8,8),
                nn.Flatten(),
                nn.Linear(1024, 10, bias=False),
                nn.LayerNorm(10)
            ),
            # (16, 16, 64)
            nn.Sequential(
                nn.AvgPool2d(8,8),
                nn.Flatten(),
                nn.Linear(256, 10, bias=False),
                nn.LayerNorm(10)
            ),
            # (16, 16, 128)
            nn.Sequential(
                nn.AvgPool2d(8,8),
                nn.Flatten(),
                nn.Linear(512, 10, bias=False),
                nn.LayerNorm(10)
            ),
            # (8, 8, 128)
            nn.Sequential(
                nn.AvgPool2d(4,4),
                nn.Flatten(),
                nn.Linear(512, 10, bias=False),
                nn.LayerNorm(10)
            ),
            # (8, 8, 256)
            nn.Sequential(
                nn.AvgPool2d(4,4),
                nn.Flatten(),
                nn.Linear(1024, 10, bias=False),
                nn.LayerNorm(10)
            ),
            # (8, 8, 256)
            nn.Sequential(
                nn.AvgPool2d(4,4),
                nn.Flatten(),
                nn.Linear(1024, 10, bias=False),
                nn.LayerNorm(10)
            ),
            # (4, 4, 256)
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, 10, bias=False),
                nn.LayerNorm(10)
            ),
            # (4, 4, 512)
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 10, bias=False),
                nn.LayerNorm(10)
            ),
            # (4, 4, 512)
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 10, bias=False),
                nn.LayerNorm(10)
            ),
            # (4, 4, 512)
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 10, bias=False),
                nn.LayerNorm(10)
            )
        ]
        return nn.ModuleList(features), nn.ModuleList(classifiers)
    
class VGG_c10_online_bn(NN_Online):
    """
        Full precision VGG-11 for cifar-10 online learning
    """
    def _module_compose(self):
        features = [
            # (32, 32, 3)
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
                ExpNorm2d(64, affine=True),
                nn.ReLU(),
            ),
            # (32, 32, 64)
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                ExpNorm2d(64, affine=True),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ),
            # (16, 16, 64)
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
                ExpNorm2d(128, affine=True),
                nn.ReLU(),
            ),
            # (16, 16, 128)
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
                ExpNorm2d(128, affine=True),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ),
            # (8, 8, 128)
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
                ExpNorm2d(256, affine=True),
                nn.ReLU(),
            ),
            # (8, 8, 256)
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
                ExpNorm2d(256, affine=True),
                nn.ReLU(),
            ),
            # (8, 8, 256)
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
                ExpNorm2d(256, affine=True),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ),
            # (4, 4, 256)
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False),
                ExpNorm2d(512, affine=True),
                nn.ReLU(),
            ),
            # (4, 4, 512)
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
                ExpNorm2d(512, affine=True),
                nn.ReLU(),
            ),
            # (4, 4, 512)
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
                ExpNorm2d(512, affine=True),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            ),
        ]
        classifiers = [
            # (32, 32, 64)
            nn.Sequential(
                nn.AvgPool2d(8,8),
                nn.Flatten(),
                nn.Linear(1024, 10, bias=False),
                ExpNorm1d(10)
            ),
            # (16, 16, 64)
            nn.Sequential(
                nn.AvgPool2d(8,8),
                nn.Flatten(),
                nn.Linear(256, 10, bias=False),
                ExpNorm1d(10)
            ),
            # (16, 16, 128)
            nn.Sequential(
                nn.AvgPool2d(8,8),
                nn.Flatten(),
                nn.Linear(512, 10, bias=False),
                ExpNorm1d(10)
            ),
            # (8, 8, 128)
            nn.Sequential(
                nn.AvgPool2d(4,4),
                nn.Flatten(),
                nn.Linear(512, 10, bias=False),
                ExpNorm1d(10)
            ),
            # (8, 8, 256)
            nn.Sequential(
                nn.AvgPool2d(4,4),
                nn.Flatten(),
                nn.Linear(1024, 10, bias=False),
                ExpNorm1d(10)
            ),
            # (8, 8, 256)
            nn.Sequential(
                nn.AvgPool2d(4,4),
                nn.Flatten(),
                nn.Linear(1024, 10, bias=False),
                ExpNorm1d(10)
            ),
            # (4, 4, 256)
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, 10, bias=False),
                ExpNorm1d(10)
            ),
            # (4, 4, 512)
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 10, bias=False),
                ExpNorm1d(10)
            ),
            # (4, 4, 512)
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 10, bias=False),
                ExpNorm1d(10)
            ),
            # (4, 4, 512)
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 10, bias=False),
                ExpNorm1d(10)
            )
        ]
        return nn.ModuleList(features), nn.ModuleList(classifiers)