# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# Online CNN basic
from models.hedge import NN_Online
# Binary basics
from models.binary_basic import *
# Quantization basics
from models.quant_basic import *
# Parameter avg
from models.para_avg_layer import *

class Conv2d_DS(nn.Module):
    """ Depth seperable Conv layers """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(Conv2d_DS, self).__init__()
        # Depth wise convolution
        self.conv_dw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=in_channels
        )
        self.bn_dw = nn.InstanceNorm2d(in_channels, affine=True)
        self.conv_pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            groups=1
        )
    def forward(self, input: Tensor) -> Tensor:
        input = self.conv_dw(input)
        input = self.bn_dw(input)
        input = self.conv_pw(input)
        return input

class AConv2d_DS(nn.Module):
    """ Depth seperable Conv layers """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, gamma=0.99):
        super(AConv2d_DS, self).__init__()
        # Depth wise convolution
        self.conv_dw = AConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=in_channels,
            gamma=gamma
        )
        self.bn_dw = AInstanceNorm2d(in_channels, affine=True, gamma=gamma)
        self.conv_pw = AConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            groups=1,
            gamma=gamma
        )
    def forward(self, input: Tensor) -> Tensor:
        input = self.conv_dw(input)
        input = self.bn_dw(input)
        input = self.conv_pw(input)
        return input

class QConv2d_DS(nn.Module):
    """ Quantized Depth seperable Conv layers """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, bit_w=8):
        # super(BQConv2d_DS, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)
        super(QConv2d_DS, self).__init__()
        # Depth wise convolution
        self.conv_dw = QConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=in_channels,
            bit_w=bit_w
        )
        self.bn_dw = nn.InstanceNorm2d(in_channels, affine=True)
        self.conv_pw = QConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            groups=1,
            bit_w=bit_w
        )
    def forward(self, input: Tensor) -> Tensor:
        input = self.conv_dw(input)
        input = self.bn_dw(input)
        input = self.conv_pw(input)
        return input

class MobileNetV1_online_c100(NN_Online):
    """
        MobileNet V1 for CIFAR100
    """
    def _module_compose(self):
        features = [
            nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(32, affine=True),
                nn.ReLU(),
            ),
            # (32, 32, 32)
            nn.Sequential(
                Conv2d_DS(32, 64, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(64, affine=True),
                nn.ReLU(),
            ),
            # (64, 32, 32)
            nn.Sequential(
                Conv2d_DS(64, 128, kernel_size=3, padding=1, stride=2, bias=False),
                nn.InstanceNorm2d(128, affine=True),
                nn.ReLU(),
            ),
            # (128, 16, 16)
            nn.Sequential(
                Conv2d_DS(128, 128, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(128, affine=True),
                nn.ReLU(),
            ),
            # (128, 16, 16)
            nn.Sequential(
                Conv2d_DS(128, 256, kernel_size=3, padding=1, stride=2, bias=False),
                nn.InstanceNorm2d(256, affine=True),
                nn.ReLU(),
            ),
            # (256, 8, 8)
            nn.Sequential(
                Conv2d_DS(256, 256, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(256, affine=True),
                nn.ReLU(),
            ),
            # (256, 8, 8)
            nn.Sequential(
                Conv2d_DS(256, 512, kernel_size=3, padding=1, stride=2, bias=False),
                nn.InstanceNorm2d(512, affine=True),
                nn.ReLU(),
            ),
            # (512, 4, 4)
            nn.Sequential(
                Conv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(512, affine=True),
                nn.ReLU(),
            ),
            # (512, 4, 4)
            nn.Sequential(
                Conv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(512, affine=True),
                nn.ReLU(),
            ),
            # (512, 4, 4)
            nn.Sequential(
                Conv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(512, affine=True),
                nn.ReLU(),
            ),
            # (512, 4, 4)
            nn.Sequential(
                Conv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(512, affine=True),
                nn.ReLU(),
            ),
            # (512, 4, 4)
            nn.Sequential(
                Conv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(512, affine=True),
                nn.ReLU(),
            ),
            # (512, 4, 4)
            nn.Sequential(
                Conv2d_DS(512, 1024, kernel_size=3, padding=1, stride=2, bias=False),
                nn.InstanceNorm2d(1024, affine=True),
                nn.ReLU(),
            ),
            # (1024, 2, 2)
            nn.Sequential(
                Conv2d_DS(1024, 1024, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(1024, affine=True),
                nn.AdaptiveAvgPool2d(1),
                nn.ReLU(),
            )
            # (1024, 1, 1)
        ]
        classifiers = [
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(32*32*32, 100, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(64*32*32, 100, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(128*16*16, 100, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(128*16*16, 100, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(256*8*8, 100, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(256*8*8, 100, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(512*4*4, 100, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(512*4*4, 100, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(512*4*4, 100, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(512*4*4, 100, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(512*4*4, 100, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(512*4*4, 100, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(1024*2*2, 100, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(1024, 100, bias=True)
            ),
        ]
        return nn.ModuleList(features), nn.ModuleList(classifiers)

class MobileNetV1_online_c100_avg(NN_Online):
    """
        MobileNet V1 for CIFAR100 with parameter averaging
    """
    def __init__(self, gamma) -> None:
        self.gamma=gamma
        super().__init__()

    def _module_compose(self):
        features = nn.ModuleList([
            nn.Sequential(
                AConv2d(3, 32, kernel_size=3, padding=1, stride=1, bias=False, gamma=self.gamma),
                AInstanceNorm2d(32, affine=True, gamma=self.gamma),
                nn.ReLU(),
            ),  # Output: (32, 32, 32)
            nn.Sequential(
                AConv2d_DS(32, 64, kernel_size=3, padding=1, stride=1, bias=False, gamma=self.gamma),
                AInstanceNorm2d(64, affine=True, gamma=self.gamma),
                nn.ReLU(),
            ),  # Output: (64, 32, 32)
            nn.Sequential(
                AConv2d_DS(64, 128, kernel_size=3, padding=1, stride=2, bias=False, gamma=self.gamma),
                AInstanceNorm2d(128, affine=True, gamma=self.gamma),
                nn.ReLU(),
            ),  # Output: (128, 16, 16)
            nn.Sequential(
                AConv2d_DS(128, 128, kernel_size=3, padding=1, stride=1, bias=False, gamma=self.gamma),
                AInstanceNorm2d(128, affine=True, gamma=self.gamma),
                nn.ReLU(),
            ),  # Output: (128, 16, 16)
            nn.Sequential(
                AConv2d_DS(128, 256, kernel_size=3, padding=1, stride=2, bias=False, gamma=self.gamma),
                AInstanceNorm2d(256, affine=True, gamma=self.gamma),
                nn.ReLU(),
            ),  # Output: (256, 8, 8)
            nn.Sequential(
                AConv2d_DS(256, 256, kernel_size=3, padding=1, stride=1, bias=False, gamma=self.gamma),
                AInstanceNorm2d(256, affine=True, gamma=self.gamma),
                nn.ReLU(),
            ),  # Output: (256, 8, 8)
            nn.Sequential(
                AConv2d_DS(256, 512, kernel_size=3, padding=1, stride=2, bias=False, gamma=self.gamma),
                AInstanceNorm2d(512, affine=True, gamma=self.gamma),
                nn.ReLU(),
            ),  # Output: (512, 4, 4)
            nn.Sequential(
                AConv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False, gamma=self.gamma),
                AInstanceNorm2d(512, affine=True, gamma=self.gamma),
                nn.ReLU(),
            ),  # Repeated blocks for (512, 4, 4)
            nn.Sequential(
                AConv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False, gamma=self.gamma),
                AInstanceNorm2d(512, affine=True, gamma=self.gamma),
                nn.ReLU(),
            ),
            nn.Sequential(
                AConv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False, gamma=self.gamma),
                AInstanceNorm2d(512, affine=True, gamma=self.gamma),
                nn.ReLU(),
            ),
            nn.Sequential(
                AConv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False, gamma=self.gamma),
                AInstanceNorm2d(512, affine=True, gamma=self.gamma),
                nn.ReLU(),
            ),
            nn.Sequential(
                AConv2d_DS(512, 1024, kernel_size=3, padding=1, stride=2, bias=False, gamma=self.gamma),
                AInstanceNorm2d(1024, affine=True, gamma=self.gamma),
                nn.ReLU(),
            ),  # Output: (1024, 2, 2)
            nn.Sequential(
                AConv2d_DS(1024, 1024, kernel_size=3, padding=1, stride=1, bias=False, gamma=self.gamma),
                AInstanceNorm2d(1024, affine=True, gamma=self.gamma),
                nn.AdaptiveAvgPool2d(1),
                nn.ReLU(),
            )  # Output: (1024, 1, 1)
        ])
        classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                ALinear(32 * 32 * 32, 100, bias=True, gamma=self.gamma)
            ),  # Classifier for the first stage output
            nn.Sequential(
                nn.Flatten(),
                ALinear(64 * 32 * 32, 100, bias=True, gamma=self.gamma)
            ),  # Classifier for the second stage output
            nn.Sequential(
                nn.Flatten(),
                ALinear(128 * 16 * 16, 100, bias=True, gamma=self.gamma)
            ),  # Classifier for the third stage output
            nn.Sequential(
                nn.Flatten(),
                ALinear(128 * 16 * 16, 100, bias=True, gamma=self.gamma)
            ),  # Classifier for the fourth stage output
            nn.Sequential(
                nn.Flatten(),
                ALinear(256 * 8 * 8, 100, bias=True, gamma=self.gamma)
            ),  # Classifier for the fifth stage output
            nn.Sequential(
                nn.Flatten(),
                ALinear(256 * 8 * 8, 100, bias=True, gamma=self.gamma)
            ),  # Classifier for the sixth stage output
            nn.Sequential(
                nn.Flatten(),
                ALinear(512 * 4 * 4, 100, bias=True, gamma=self.gamma)
            ),  # Classifier for the seventh stage output
            nn.Sequential(
                nn.Flatten(),
                ALinear(512 * 4 * 4, 100, bias=True, gamma=self.gamma)
            ),  # Classifier for the eighth stage output
            nn.Sequential(
                nn.Flatten(),
                ALinear(512 * 4 * 4, 100, bias=True, gamma=self.gamma)
            ),  # Classifier for the ninth stage output
            nn.Sequential(
                nn.Flatten(),
                ALinear(512 * 4 * 4, 100, bias=True, gamma=self.gamma)
            ),  # Classifier for the tenth stage output
            nn.Sequential(
                nn.Flatten(),
                ALinear(512 * 4 * 4, 100, bias=True, gamma=self.gamma)
            ),  # Classifier for the eleventh stage output
            nn.Sequential(
                nn.Flatten(),
                ALinear(512 * 4 * 4, 100, bias=True, gamma=self.gamma)
            ),  # Classifier for the twelfth stage output
            nn.Sequential(
                nn.Flatten(),
                ALinear(1024 * 2 * 2, 100, bias=True, gamma=self.gamma)
            ),  # Classifier for the thirteenth stage output
            nn.Sequential(
                nn.Flatten(),
                ALinear(1024, 100, bias=True, gamma=self.gamma)
            ),  # Classifier for the last stage output after global pooling
        ])
        return features, classifiers



class MobileNetV1_online_c100_Quant(NN_Online):
    def __init__(self, bit_w, bit_a) -> None:
        self.bit_w = bit_w
        self.bit_a = bit_a
        super().__init__()

    def _module_compose(self):
        features = [
            nn.Sequential(
                QConv2d(3, 32, kernel_size=3, padding=1, stride=1, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(32, affine=True),
                get_quant('a', bit=self.bit_a),
            ),
            # (32, 32, 32)
            nn.Sequential(
                QConv2d_DS(32, 64, kernel_size=3, padding=1, stride=1, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(64, affine=True),
                get_quant('a', bit=self.bit_a),
            ),
            # (64, 32, 32)
            nn.Sequential(
                QConv2d_DS(64, 128, kernel_size=3, padding=1, stride=2, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(128, affine=True),
                get_quant('a', bit=self.bit_a),
            ),
            # (128, 16, 16)
            nn.Sequential(
                QConv2d_DS(128, 128, kernel_size=3, padding=1, stride=1, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(128, affine=True),
                get_quant('a', bit=self.bit_a),
            ),
            # (128, 16, 16)
            nn.Sequential(
                QConv2d_DS(128, 256, kernel_size=3, padding=1, stride=2, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(256, affine=True),
                get_quant('a', bit=self.bit_a),
            ),
            # (256, 8, 8)
            nn.Sequential(
                QConv2d_DS(256, 256, kernel_size=3, padding=1, stride=1, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(256, affine=True),
                get_quant('a', bit=self.bit_a),
            ),
            # (256, 8, 8)
            nn.Sequential(
                QConv2d_DS(256, 512, kernel_size=3, padding=1, stride=2, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(512, affine=True),
                get_quant('a', bit=self.bit_a),
            ),
            # (512, 4, 4)
            nn.Sequential(
                QConv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(512, affine=True),
                get_quant('a', bit=self.bit_a),
            ),
            # (512, 4, 4)
            nn.Sequential(
                QConv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(512, affine=True),
                get_quant('a', bit=self.bit_a),
            ),
            # (512, 4, 4)
            nn.Sequential(
                QConv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(512, affine=True),
                get_quant('a', bit=self.bit_a),
            ),
            # (512, 4, 4)
            nn.Sequential(
                QConv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(512, affine=True),
                get_quant('a', bit=self.bit_a),
            ),
            # (512, 4, 4)
            nn.Sequential(
                QConv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(512, affine=True),
                get_quant('a', bit=self.bit_a),
            ),
            # (512, 4, 4)
            nn.Sequential(
                QConv2d_DS(512, 1024, kernel_size=3, padding=1, stride=2, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(1024, affine=True),
                get_quant('a', bit=self.bit_a),
            ),
            # (1024, 4, 4)
            nn.Sequential(
                QConv2d_DS(1024, 1024, kernel_size=3, padding=1, stride=1, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(1024, affine=True),
                nn.AdaptiveAvgPool2d(1),
                get_quant('a', bit=self.bit_a),
            )
            # (1024, 1, 1)
        ]
        classifiers = [
            nn.Sequential(
                nn.Flatten(),
                QLinear(32*32*32, 100, bias=True, bit_w=self.bit_w),     
            ),
            nn.Sequential(
                nn.Flatten(),
                QLinear(64*32*32, 100, bias=True, bit_w=self.bit_w),    
            ),
            nn.Sequential(
                nn.Flatten(),
                QLinear(128*16*16, 100, bias=True, bit_w=self.bit_w),    
            ),
            nn.Sequential(
                nn.Flatten(),
                QLinear(128*16*16, 100, bias=True, bit_w=self.bit_w),     
            ),
            nn.Sequential(
                nn.Flatten(),
                QLinear(256*8*8, 100, bias=True, bit_w=self.bit_w),    
            ),
            nn.Sequential(
                nn.Flatten(),
                QLinear(256*8*8, 100, bias=True, bit_w=self.bit_w),  
            ),
            nn.Sequential(
                nn.Flatten(),
                QLinear(512*4*4, 100, bias=True, bit_w=self.bit_w),   
            ),
            nn.Sequential(
                nn.Flatten(),
                QLinear(512*4*4, 100, bias=True, bit_w=self.bit_w),
            ),
            nn.Sequential(
                nn.Flatten(),
                QLinear(512*4*4, 100, bias=True, bit_w=self.bit_w), 
            ),
            nn.Sequential(
                nn.Flatten(),
                QLinear(512*4*4, 100, bias=True, bit_w=self.bit_w), 
            ),
            nn.Sequential(
                nn.Flatten(),
                QLinear(512*4*4, 100, bias=True, bit_w=self.bit_w), 
            ),
            nn.Sequential(
                nn.Flatten(),
                QLinear(512*4*4, 100, bias=True, bit_w=self.bit_w), 
            ),
            nn.Sequential(
                nn.Flatten(),
                QLinear(1024*2*2, 100, bias=True, bit_w=self.bit_w), 
            ),
            nn.Sequential(
                nn.Flatten(),
                QLinear(1024, 100, bias=True, bit_w=self.bit_w)
            ),
        ]
        return nn.ModuleList(features), nn.ModuleList(classifiers)