# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# Binary basics
from models.binary_basic import *
from models.quant_basic import *
# Residule blocks
from models.res_blk import *
# Online CNN basic
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
else:
    raise NotImplementedError
import math

RESNET_CONFIG = {
    10: [1, 1, 1, 1],
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]
}
IF_8BIT_FIRST_LAST = True

class ResNet_ol(NN_Online):
    """
        Base class for all sorts of full-precision ResNets with online learning
    """
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 device: torch.device, 
                 res_blk: BasicBlock | BottleNeckBlock, 
                 blk_nums: list[int]=[3, 4, 6, 3], 
                 second_layer_channel: int=256
        ) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.blk_nums = blk_nums
        self.res_blk = res_blk
        self.second_layer_channel = second_layer_channel
        self.lin_dim = self.second_layer_channel*8
        # self.module_list = self.compose_layers().to(device)
        super(self, NN_Online).__init__()

    def _make_layers(self, in_chan, out_chan, blk_num):
        layers = [
            nn.Sequential(
                self.res_blk(in_chan, out_chan),
                nn.InstanceNorm2d(out_chan),
                nn.ReLU()
            )
        ]
        classifiers = [
            nn.Sequential(
                nn.AdaptiveAvgPool2d(int(math.sqrt(self.lin_dim/out_chan))),
                nn.Flatten(),
                nn.Linear(self.lin_dim, self.out_dim, bias=False),
                nn.LayerNorm(self.out_dim)
            )
        ]
        for _ in range(1, blk_num):
            layers += [
                nn.Sequential(
                    self.res_blk(out_chan, out_chan),
                    nn.InstanceNorm2d(out_chan),
                    nn.ReLU()
                )
            ]
            classifiers += [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(int(math.sqrt(self.lin_dim/out_chan))),
                    nn.Flatten(),
                    nn.Linear(self.lin_dim, self.out_dim, bias=False),
                    nn.LayerNorm(self.out_dim)
                )
            ]
        return layers, classifiers
    
    def compose_layers(self) -> list:
        m_list = []
        classifiers = []
        # First layer
        m_list += [
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.InstanceNorm2d(64, affine=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        ]
        classifiers += [
            nn.Sequential(
                nn.AdaptiveAvgPool2d(int(math.sqrt(self.lin_dim/64))),
                nn.Flatten(),
                nn.Linear(self.lin_dim, self.out_dim, bias=False),
                nn.LayerNorm(self.out_dim)
            )
        ]
        # Second layer
        f, c = self._make_layers(64, self.second_layer_channel, blk_num=self.blk_nums[0])
        m_list += f
        classifiers += c
        # Third to fifth layer
        for i in range(1, 4):
            f, c = self._make_layers(self.second_layer_channel*2**(i-1), self.second_layer_channel*2**(i), self.blk_nums[i])
            m_list += f
            classifiers += c
        return m_list, classifiers
    
    def _module_compose(self):
        features, classifiers = self.compose_layers()
        return nn.ModuleList(features), nn.ModuleList(classifiers)