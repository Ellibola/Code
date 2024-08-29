import torch
import torch.nn as nn
import torch.nn.functional as F
# Hedge
from models.hedge import NN_Online
# EG
# from models.eg import NN_Online
from models.binary_basic import *
from models.quant_basic import *

class VGG_c100(nn.Module):
    """
        Full precision VGG-11 for cifar-100
    """
    def __init__(self) -> None:
        super(VGG_c100, self).__init__()
        self.module_list = nn.Sequential(
            # (32, 32, 3)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (32, 32, 64)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # (16, 16, 64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # (16, 16, 128)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # (8, 8, 128)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # (8, 8, 256)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # (8, 8, 256)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # (4, 4, 256)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # (4, 4, 512)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # (4, 4, 512)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            # (1, 1, 512)
            nn.Flatten(),
            nn.Linear(512, 100, bias=False),
            nn.BatchNorm1d(100)
        )
    
    def forward(self, x:torch.Tensor):
        return self.module_list(x)
    
class VGG_c100_olnorm(nn.Module):
    """
        Full precision VGG-11 for cifar-100
    """
    def __init__(self) -> None:
        super(VGG_c100_olnorm, self).__init__()
        self.module_list = nn.Sequential(
            # (32, 32, 3)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            # (32, 32, 64)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # (16, 16, 64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            # (16, 16, 128)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # (8, 8, 128)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(),
            # (8, 8, 256)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(),
            # (8, 8, 256)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # (4, 4, 256)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(),
            # (4, 4, 512)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(),
            # (4, 4, 512)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            # (1, 1, 512)
            nn.Flatten(),
            nn.Linear(512, 100, bias=False),
            nn.LayerNorm(100)
        )
    
    def forward(self, x:torch.Tensor):
        return self.module_list(x)

class VGG_c10_olnorm(nn.Module):
    """
        Full precision VGG-11 for cifar-100
    """
    def __init__(self) -> None:
        super(VGG_c10_olnorm, self).__init__()
        self.module_list = nn.Sequential(
            # (32, 32, 3)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            # (32, 32, 64)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # (16, 16, 64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            # (16, 16, 128)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # (8, 8, 128)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(),
            # (8, 8, 256)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(),
            # (8, 8, 256)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # (4, 4, 256)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(),
            # (4, 4, 512)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(),
            # (4, 4, 512)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            # (1, 1, 512)
            nn.Flatten(),
            nn.Linear(512, 10, bias=False),
            nn.LayerNorm(10)
        )
    
    def forward(self, x:torch.Tensor):
        return self.module_list(x)

class VGG_c10(nn.Module):
    """
        Full precision VGG-11 for cifar-100
    """
    def __init__(self) -> None:
        super(VGG_c10, self).__init__()
        self.module_list = nn.Sequential(
            # (32, 32, 3)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            # (32, 32, 64)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # (16, 16, 64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            # (16, 16, 128)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # (8, 8, 128)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(),
            # (8, 8, 256)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(),
            # (8, 8, 256)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # (4, 4, 256)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(),
            # (4, 4, 512)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(),
            # (4, 4, 512)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            # (1, 1, 512)
            nn.Flatten(),
            nn.Linear(512, 10, bias=False),
            nn.BatchNorm1d(10)
        )
    
    def forward(self, x:torch.Tensor):
        return self.module_list(x)

class VGG_c100_Quant(nn.Module):
    """
        Quantized VGG-11 for cifar-100
    """
    def __init__(self, bit_w, bit_a) -> None:
        super(VGG_c100_Quant, self).__init__()
        self.bit_w = bit_w
        self.bit_a = bit_a
        self.module_list = nn.Sequential(
            # (32, 32, 3)
            QConv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bit_w = bit_w, bias=False),
            nn.BatchNorm2d(64),
            get_quant('a', bit=self.bit_a),
            # (32, 32, 64)
            QConv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bit_w = bit_w, bias=False),
            nn.BatchNorm2d(64),
            get_quant('a', bit=self.bit_a),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # (16, 16, 64)
            QConv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bit_w = bit_w, bias=False),
            nn.BatchNorm2d(128),
            get_quant('a', bit=self.bit_a),
            # (16, 16, 128)
            QConv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bit_w = bit_w, bias=False),
            nn.BatchNorm2d(128),
            get_quant('a', bit=self.bit_a),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # (8, 8, 128)
            QConv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bit_w = bit_w, bias=False),
            nn.BatchNorm2d(256),
            get_quant('a', bit=self.bit_a),
            # (8, 8, 256)
            QConv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bit_w = bit_w, bias=False),
            nn.BatchNorm2d(256),
            get_quant('a', bit=self.bit_a),
            # (8, 8, 256)
            QConv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bit_w = bit_w, bias=False),
            nn.BatchNorm2d(256),
            get_quant('a', bit=self.bit_a),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # (4, 4, 256)
            QConv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bit_w = bit_w, bias=False),
            nn.BatchNorm2d(512),
            get_quant('a', bit=self.bit_a),
            # (4, 4, 512)
            QConv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bit_w = bit_w, bias=False),
            nn.BatchNorm2d(512),
            get_quant('a', bit=self.bit_a),
            # (4, 4, 512)
            QConv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bit_w = bit_w, bias=False),
            nn.BatchNorm2d(512),
            get_quant('a', bit=self.bit_a),
            nn.AdaptiveAvgPool2d(1),
            # (1, 1, 512)
            nn.Flatten(),
            QLinear(512, 100, bit_w = bit_w, bias=False),
            nn.BatchNorm1d(100)
        )
    
    def forward(self, x:torch.Tensor):
        return self.module_list(x)