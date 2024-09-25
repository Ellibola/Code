# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# Binary basics
from models.binary_basic import *
from models.quant_basic import *
# Residule blocks
from models.res_blk import *

RESNET_CONFIG = {
    10: [1, 1, 1, 1],
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]
}
IF_8BIT_FIRST_LAST = True

class ResNet(nn.Module):
    """
        Base class for all sorts of full-precision ResNets
    """
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 device: torch.device, 
                 res_blk: BasicBlock | BottleNeckBlock, 
                 blk_nums: list[int]=[3, 4, 6, 3], 
                 second_layer_channel: int=256,
                 norm_2d = nn.BatchNorm2d|nn.InstanceNorm2d,
                 norm_1d = nn.BatchNorm1d|nn.LayerNorm
        ) -> None:
        super(ResNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.blk_nums = blk_nums
        self.res_blk = res_blk
        self.norm_2d = norm_2d
        self.norm_1d = norm_1d
        self.second_layer_channel = second_layer_channel
        self.module_list = self.compose_layers().to(device)
    
    def _make_layers(self, in_chan, out_chan, blk_num):
        layers = [
            self.res_blk(in_chan, out_chan, norm_2d=self.norm_2d),
            # self.norm_2d(out_chan, affine=True, track_running_stats=True),
            nn.ReLU()
        ]
        for _ in range(1, blk_num):
            layers += [
                self.res_blk(out_chan, out_chan, norm_2d=self.norm_2d),
                # self.norm_2d(out_chan, affine=True, track_running_stats=True),
                nn.ReLU()
            ]
        return layers
    
    def compose_layers(self) -> list:
        m_list = []
        # First layer
        if self.in_dim>32:
            m_list += [
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
                self.norm_2d(64, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ]
        else:
            m_list += [
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                self.norm_2d(64, affine=True, track_running_stats=True),
                nn.ReLU()
            ]
        # Second layer
        m_list += self._make_layers(64, self.second_layer_channel, blk_num=self.blk_nums[0])
        # Third to fifth layer
        for i in range(1, 4):
            m_list += self._make_layers(self.second_layer_channel*2**(i-1), self.second_layer_channel*2**(i), self.blk_nums[i])
        # Classifier
        m_list += [
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(self.second_layer_channel*8, self.out_dim, bias=True),
            # self.norm_1d(self.out_dim)
        ]
        return nn.Sequential(*m_list)
    
    def forward(self, x:torch.Tensor):
        return self.module_list(x)
    
class QResNet(nn.Module):
    """
        Base class for all sorts of multi-bit quantized ResNets
    """
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 device: torch.device, 
                 res_blk: QBasicBlock | QBottleNeckBlock, 
                 blk_nums: list[int]=[3, 4, 6, 3], 
                 second_layer_channel: int=256, 
                 bit_w=8, 
                 bit_a=8
        ) -> None:
        super(self, QResNet).__init__()
        self.bit_w = bit_w
        self.bit_a = bit_a
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.blk_nums = blk_nums
        self.res_blk = res_blk
        self.second_layer_channel = second_layer_channel
        self.module_list = self.compose_layers().to(device)

    def _make_layers(self, in_chan, out_chan, blk_num, if_first=False):
        layers = [
            get_quant('a', 8 if if_first else self.bit_a, self.norm),
            self.res_blk(in_chan, out_chan, bit_w=self.bit_w, bit_a=self.bit_a, norm=self.norm),
            nn.BatchNorm2d(out_chan)
        ]
        for _ in range(1, blk_num):
            layers += [
                get_quant('a', self.bit_a, self.norm),
                self.res_blk(out_chan, out_chan, bit_w=self.bit_w, bit_a=self.bit_a, norm=self.norm),
                nn.BatchNorm2d(out_chan)
            ]
        return layers

    def compose_layers(self) -> list:
        m_list = []
        # First layer
        m_list += [
            QConv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False, bit_w=8 if IF_8BIT_FIRST_LAST else self.bit_w),
            nn.BatchNorm2d(64, affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        # Second layer
        m_list += self._make_layers(64, self.second_layer_channel, blk_num=self.blk_nums[0], if_first=True)
        # Third to fifth layer
        for i in range(1, 4):
            m_list += self._make_layers(self.second_layer_channel*2**(i-1), self.second_layer_channel*2**(i), self.blk_nums[i])
        # Classifier
        m_list += [
            nn.AdaptiveMaxPool2d(1),
            get_quant('a', 8 if IF_8BIT_FIRST_LAST else self.bit_a, self.norm),
            QLinear(self.second_layer_channel*8, self.out_dim, bias=False, bit_w=8 if IF_8BIT_FIRST_LAST else self.bit_w, bit_a=8 if IF_8BIT_FIRST_LAST else self.bit_a, norm=self.norm),
            nn.BatchNorm1d(self.out_dim)
        ]
        return nn.Sequential(*m_list)

    def forward(self, x:torch.Tensor):
        return self.module_list(x)
    

def resnet(dataset='cifar100', device=torch.device('cpu'), num_layer=50 , bit_w=8, bit_a=8, norm='bn'):
    assert bit_w in [1, 2, 3, 4, 8, 32], "Unsupported weight quantization bit:{}".format(bit_w)
    assert bit_a in [1, 2, 3, 4, 8, 32], "Unsupported activation quantization bit:{}".format(bit_a)
    assert num_layer in RESNET_CONFIG.keys(), "Unsupported number of layers({}) for ResNet".format(num_layer)
    in_dim = 32 if 'mnist' in dataset else \
             32 if 'cifar' in dataset else \
             224 if 'imagenet' in dataset else \
             224 if 'caltech' in dataset else \
             256
    out_dim = 10 if dataset=='mnist' else \
              10 if dataset=='cifar10' else \
              100 if dataset=='cifar100' else \
              256 if 'caltech' in dataset else \
              1000 if dataset=='imagenet' else \
              31
    blk_nums = RESNET_CONFIG[num_layer]
    if num_layer<50:
        res_blk = BasicBlock if (bit_w==32)&(bit_a==32) else \
                QBasicBlock
    else:
        res_blk = BottleNeckBlock if (bit_w==32)&(bit_a==32) else \
                QBottleNeckBlock
    second_layer_channel = 64 if num_layer<50 else 256
    if (bit_w==32)&(bit_a==32):
        return ResNet(
            in_dim=in_dim, 
            out_dim=out_dim, 
            device=device, 
            res_blk=res_blk,
            blk_nums=blk_nums,
            second_layer_channel=second_layer_channel,
            norm_2d=nn.BatchNorm2d if norm=='bn' else nn.InstanceNorm2d,
            norm_1d=nn.BatchNorm1d if norm=='bn' else nn.LayerNorm,
            ).to(device)
    else:
        return QResNet(
                in_dim=in_dim, 
                out_dim=out_dim, 
                device=device, 
                res_blk=res_blk,
                blk_nums=blk_nums,
                second_layer_channel=second_layer_channel,
                bit_w=bit_w,
                bit_a=bit_a,
            ).to(device)
    

