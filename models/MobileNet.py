# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# Binary basics
from models.binary_basic import *
# Quantization basics
from models.quant_basic import *

class Conv2d_DS(nn.Module):
    """ Depth seperable Conv layers """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        # super(BQConv2d_DS, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)
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
        self.bn_dw = nn.BatchNorm2d(in_channels, affine=True)
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
        self.bn_dw = nn.BatchNorm2d(in_channels, affine=True)
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

class MobileNetV1_c100(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module_list = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            Conv2d_DS(32, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            Conv2d_DS(64, 128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            Conv2d_DS(128, 128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            Conv2d_DS(128, 256, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(),
            Conv2d_DS(256, 256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(),
            Conv2d_DS(256, 512, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(),
            Conv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(),
            Conv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(),
            Conv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(),
            Conv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(),
            Conv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(),
            Conv2d_DS(512, 1024, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(1024, affine=True),
            nn.ReLU(),
            Conv2d_DS(1024, 1024, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(1024, affine=True),
            nn.AdaptiveAvgPool2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 100, bias=False),
            nn.BatchNorm1d(100, affine=True)
        )
    def forward(self, x:torch.Tensor):
        return self.module_list(x)

class MobileNetV1_c100_Quant(nn.Module):
    def __init__(self, bit_w, bit_a) -> None:
        super().__init__()
        self.module_list = nn.Sequential(
            QConv2d(3, 32, kernel_size=3, padding=1, stride=1, bias=False, bit_w=bit_w),
            nn.BatchNorm2d(32, affine=True),
            get_quant('a', bit_a),
            QConv2d_DS(32, 64, kernel_size=3, padding=1, stride=1, bias=False, bit_w=bit_w),
            nn.BatchNorm2d(64, affine=True),
            get_quant('a', bit_a),
            QConv2d_DS(64, 128, kernel_size=3, padding=1, stride=2, bias=False, bit_w=bit_w),
            nn.BatchNorm2d(128, affine=True),
            get_quant('a', bit_a),
            QConv2d_DS(128, 128, kernel_size=3, padding=1, stride=1, bias=False, bit_w=bit_w),
            nn.BatchNorm2d(128, affine=True),
            get_quant('a', bit_a),
            QConv2d_DS(128, 256, kernel_size=3, padding=1, stride=2, bias=False, bit_w=bit_w),
            nn.BatchNorm2d(256, affine=True),
            get_quant('a', bit_a),
            QConv2d_DS(256, 256, kernel_size=3, padding=1, stride=1, bias=False, bit_w=bit_w),
            nn.BatchNorm2d(256, affine=True),
            get_quant('a', bit_a),
            QConv2d_DS(256, 512, kernel_size=3, padding=1, stride=2, bias=False, bit_w=bit_w),
            nn.BatchNorm2d(512, affine=True),
            get_quant('a', bit_a),
            QConv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False, bit_w=bit_w),
            nn.BatchNorm2d(512, affine=True),
            get_quant('a', bit_a),
            QConv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False, bit_w=bit_w),
            nn.BatchNorm2d(512, affine=True),
            get_quant('a', bit_a),
            QConv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False, bit_w=bit_w),
            nn.BatchNorm2d(512, affine=True),
            get_quant('a', bit_a),
            QConv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False, bit_w=bit_w),
            nn.BatchNorm2d(512, affine=True),
            get_quant('a', bit_a),
            QConv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False, bit_w=bit_w),
            nn.BatchNorm2d(512, affine=True),
            get_quant('a', bit_a),
            QConv2d_DS(512, 1024, kernel_size=3, padding=1, stride=2, bias=False, bit_w=bit_w),
            nn.BatchNorm2d(1024, affine=True),
            get_quant('a', bit_a),
            QConv2d_DS(1024, 1024, kernel_size=3, padding=1, stride=1, bias=False, bit_w=bit_w),
            nn.BatchNorm2d(1024, affine=True),
            nn.AdaptiveAvgPool2d(1),
            get_quant('a', bit_a),
            nn.Flatten(),
            QLinear(1024, 100, bias=False, bit_w=bit_w),
            nn.BatchNorm1d(100, affine=True)
        )

    def forward(self, x:torch.Tensor):
        return self.module_list(x)
    

class MobileNetV1_calt101(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module_list = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            Conv2d_DS(32, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            Conv2d_DS(64, 128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            Conv2d_DS(128, 128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            Conv2d_DS(128, 256, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(),
            Conv2d_DS(256, 256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(),
            Conv2d_DS(256, 512, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(),
            Conv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(),
            Conv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(),
            Conv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(),
            Conv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(),
            Conv2d_DS(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(),
            Conv2d_DS(512, 1024, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(1024, affine=True),
            nn.ReLU(),
            Conv2d_DS(1024, 1024, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(1024, affine=True),
            nn.AdaptiveAvgPool2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 101, bias=False),
            nn.BatchNorm1d(101, affine=True)
        )
    def forward(self, x:torch.Tensor):
        return self.module_list(x)