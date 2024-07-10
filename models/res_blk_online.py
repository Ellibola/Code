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
# Exp avg norm
from models.ExpNorm import ExpNorm1d, ExpNorm2d

######### Online blocks with instance/layer norm #########

"""
    Full precision blocks
"""
class BottleNeckBlock_ins(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, neck_reduction=4) -> None:
        super(BottleNeckBlock_ins, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.neck_reduction = neck_reduction
        self.neck_channel = int(self.out_channel/neck_reduction)
        if self.in_channel!=self.out_channel:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=1, stride=2, bias=False),
                nn.InstanceNorm2d(self.out_channel, affine=True)
            )
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=self.neck_channel, kernel_size=1, bias=False),
                nn.InstanceNorm2d(self.neck_channel, affine=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.neck_channel, out_channels=self.neck_channel, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(self.neck_channel, affine=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.neck_channel, out_channels=self.out_channel, kernel_size=1, bias=False),
                nn.InstanceNorm2d(self.out_channel, affine=True)
            )
        else:
            self.proj = lambda x: x
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=self.neck_channel, kernel_size=1, bias=False),
                nn.InstanceNorm2d(self.neck_channel, affine=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.neck_channel, out_channels=self.neck_channel, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(self.neck_channel, affine=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.neck_channel, out_channels=self.out_channel, kernel_size=1, bias=False),
                nn.InstanceNorm2d(self.out_channel, affine=True)
            )
    
    def forward(self, x: torch.Tensor):
        x_ = x.clone()
        x = self.layers(x)
        x_ = self.proj(x_)
        return x+x_
    
class BasicBlock_ins(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super(BasicBlock_ins, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        if self.in_channel!=self.out_channel:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=1, stride=2, bias=False),
                nn.InstanceNorm2d(self.out_channel, affine=True)
            )
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(self.out_channel, affine=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(self.out_channel, affine=True)
            )
        else:
            self.proj = lambda x: x
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(self.out_channel, affine=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(self.out_channel, affine=True)
            )
    def forward(self, x: torch.Tensor):
        x_ = x.clone()
        x = self.layers(x)
        x_ = self.proj(x_)
        return x+x_
    


"""
    Multi-bit quantized blocks
"""
class QBottleNeckBlock_ins(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, neck_reduction=4, bit_w=8, bit_a=8, norm=False) -> None:
        super(QBottleNeckBlock_ins, self).__init__()
        self.bit_w = bit_w
        self.bit_a = bit_a
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.neck_reduction = neck_reduction
        self.neck_channel = int(self.out_channel/neck_reduction)
        if self.in_channel!=self.out_channel:
            self.proj = nn.Sequential(
                QConv2d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=1, stride=2, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(self.out_channel, affine=True)
            )
            self.layers = nn.Sequential(
                QConv2d(in_channels=in_channel, out_channels=self.neck_channel, kernel_size=1, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(self.neck_channel, affine=True),
                get_quant('a', bit_a, norm),
                QConv2d(in_channels=self.neck_channel, out_channels=self.neck_channel, kernel_size=3, stride=2, padding=1, bias=False, bit_w=self.bit_w, bit_a=self.bit_a, norm=norm),
                nn.InstanceNorm2d(self.neck_channel, affine=True),
                get_quant('a', bit_a, norm),
                QConv2d(in_channels=self.neck_channel, out_channels=self.out_channel, kernel_size=1, bias=False, bit_w=self.bit_w, bit_a=self.bit_a, norm=norm),
                nn.InstanceNorm2d(out_channel, affine=True)
            )
        else:
            self.proj = lambda x: x
            self.layers = nn.Sequential(
                QConv2d(in_channels=in_channel, out_channels=self.neck_channel, kernel_size=1, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(self.neck_channel, affine=True),
                get_quant('a', bit_a, norm),
                QConv2d(in_channels=self.neck_channel, out_channels=self.neck_channel, kernel_size=3, padding=1, bias=False, bit_w=self.bit_w, bit_a=self.bit_a, norm=norm),
                nn.InstanceNorm2d(self.neck_channel, affine=True),
                get_quant('a', bit_a, norm),
                QConv2d(in_channels=self.neck_channel, out_channels=self.out_channel, kernel_size=1, bias=False, bit_w=self.bit_w, bit_a=self.bit_a, norm=norm),
                nn.InstanceNorm2d(out_channel, affine=True)
            )
    
    def forward(self, x: torch.Tensor):
        x_ = x.clone()
        x = self.layers(x)
        x_ = self.proj(x_)
        return x+x_
    
class QBasicBlock_ins(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, bit_w=8, bit_a=8, norm=False) -> None:
        super(QBasicBlock_ins, self).__init__()
        self.bit_w = bit_w
        self.bit_a = bit_a
        self.in_channel = in_channel
        self.out_channel = out_channel
        if self.in_channel!=self.out_channel:
            self.proj = nn.Sequential(
                QConv2d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=1, stride=2, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(self.out_channel, affine=True)
            )
            self.layers = nn.Sequential(
                QConv2d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=3, stride=2, padding=1, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(self.out_channel, affine=True),
                get_quant('a', bit_a, norm),
                QConv2d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, padding=1, bias=False, bit_w=self.bit_w, bit_a=self.bit_a, norm=norm),
                nn.InstanceNorm2d(self.out_channel, affine=True)
            )
        else:
            self.proj = lambda x: x
            self.layers = nn.Sequential(
                QConv2d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=3, stride=1, padding=1, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(self.out_channel, affine=True),
                get_quant('a', bit_a, norm),
                QConv2d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, padding=1, bias=False, bit_w=self.bit_w, bit_a=self.bit_a, norm=norm),
                nn.InstanceNorm2d(self.out_channel, affine=True)
            )
    def forward(self, x: torch.Tensor):
        x_ = x.clone()
        x = self.layers(x)
        x_ = self.proj(x_)
        return x+x_
    

######### Online blocks with exp norm #########

"""
    Full precision blocks
"""
class BottleNeckBlock_ins(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, neck_reduction=4) -> None:
        super(BottleNeckBlock_ins, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.neck_reduction = neck_reduction
        self.neck_channel = int(self.out_channel/neck_reduction)
        if self.in_channel!=self.out_channel:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=1, stride=2, bias=False),
                ExpNorm2d(self.out_channel, affine=True)
            )
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=self.neck_channel, kernel_size=1, bias=False),
                ExpNorm2d(self.neck_channel, affine=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.neck_channel, out_channels=self.neck_channel, kernel_size=3, stride=2, padding=1, bias=False),
                ExpNorm2d(self.neck_channel, affine=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.neck_channel, out_channels=self.out_channel, kernel_size=1, bias=False),
                ExpNorm2d(self.out_channel, affine=True)
            )
        else:
            self.proj = lambda x: x
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=self.neck_channel, kernel_size=1, bias=False),
                ExpNorm2d(self.neck_channel, affine=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.neck_channel, out_channels=self.neck_channel, kernel_size=3, padding=1, bias=False),
                ExpNorm2d(self.neck_channel, affine=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.neck_channel, out_channels=self.out_channel, kernel_size=1, bias=False),
                ExpNorm2d(self.out_channel, affine=True)
            )
    
    def forward(self, x: torch.Tensor):
        x_ = x.clone()
        x = self.layers(x)
        x_ = self.proj(x_)
        return x+x_
    
class BasicBlock_ins(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super(BasicBlock_ins, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        if self.in_channel!=self.out_channel:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=1, stride=2, bias=False),
                ExpNorm2d(self.out_channel, affine=True)
            )
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=3, stride=2, padding=1, bias=False),
                ExpNorm2d(self.out_channel, affine=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, padding=1, bias=False),
                ExpNorm2d(self.out_channel, affine=True)
            )
        else:
            self.proj = lambda x: x
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=3, stride=1, padding=1, bias=False),
                ExpNorm2d(self.out_channel, affine=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, padding=1, bias=False),
                ExpNorm2d(self.out_channel, affine=True)
            )
    def forward(self, x: torch.Tensor):
        x_ = x.clone()
        x = self.layers(x)
        x_ = self.proj(x_)
        return x+x_
    


"""
    Multi-bit quantized blocks
"""
class QBottleNeckBlock_ins(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, neck_reduction=4, bit_w=8, bit_a=8, norm=False) -> None:
        super(QBottleNeckBlock_ins, self).__init__()
        self.bit_w = bit_w
        self.bit_a = bit_a
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.neck_reduction = neck_reduction
        self.neck_channel = int(self.out_channel/neck_reduction)
        if self.in_channel!=self.out_channel:
            self.proj = nn.Sequential(
                QConv2d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=1, stride=2, bias=False, bit_w=self.bit_w),
                ExpNorm2d(self.out_channel, affine=True)
            )
            self.layers = nn.Sequential(
                QConv2d(in_channels=in_channel, out_channels=self.neck_channel, kernel_size=1, bias=False, bit_w=self.bit_w),
                ExpNorm2d(self.neck_channel, affine=True),
                get_quant('a', bit_a, norm),
                QConv2d(in_channels=self.neck_channel, out_channels=self.neck_channel, kernel_size=3, stride=2, padding=1, bias=False, bit_w=self.bit_w, bit_a=self.bit_a, norm=norm),
                ExpNorm2d(self.neck_channel, affine=True),
                get_quant('a', bit_a, norm),
                QConv2d(in_channels=self.neck_channel, out_channels=self.out_channel, kernel_size=1, bias=False, bit_w=self.bit_w, bit_a=self.bit_a, norm=norm),
                ExpNorm2d(out_channel, affine=True)
            )
        else:
            self.proj = lambda x: x
            self.layers = nn.Sequential(
                QConv2d(in_channels=in_channel, out_channels=self.neck_channel, kernel_size=1, bias=False, bit_w=self.bit_w),
                ExpNorm2d(self.neck_channel, affine=True),
                get_quant('a', bit_a, norm),
                QConv2d(in_channels=self.neck_channel, out_channels=self.neck_channel, kernel_size=3, padding=1, bias=False, bit_w=self.bit_w, bit_a=self.bit_a, norm=norm),
                ExpNorm2d(self.neck_channel, affine=True),
                get_quant('a', bit_a, norm),
                QConv2d(in_channels=self.neck_channel, out_channels=self.out_channel, kernel_size=1, bias=False, bit_w=self.bit_w, bit_a=self.bit_a, norm=norm),
                ExpNorm2d(out_channel, affine=True)
            )
    
    def forward(self, x: torch.Tensor):
        x_ = x.clone()
        x = self.layers(x)
        x_ = self.proj(x_)
        return x+x_
    
class QBasicBlock_ins(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, bit_w=8, bit_a=8, norm=False) -> None:
        super(QBasicBlock_ins, self).__init__()
        self.bit_w = bit_w
        self.bit_a = bit_a
        self.in_channel = in_channel
        self.out_channel = out_channel
        if self.in_channel!=self.out_channel:
            self.proj = nn.Sequential(
                QConv2d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=1, stride=2, bias=False, bit_w=self.bit_w),
                ExpNorm2d(self.out_channel, affine=True)
            )
            self.layers = nn.Sequential(
                QConv2d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=3, stride=2, padding=1, bias=False, bit_w=self.bit_w),
                ExpNorm2d(self.out_channel, affine=True),
                get_quant('a', bit_a, norm),
                QConv2d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, padding=1, bias=False, bit_w=self.bit_w, bit_a=self.bit_a, norm=norm),
                ExpNorm2d(self.out_channel, affine=True)
            )
        else:
            self.proj = lambda x: x
            self.layers = nn.Sequential(
                QConv2d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=3, stride=1, padding=1, bias=False, bit_w=self.bit_w),
                ExpNorm2d(self.out_channel, affine=True),
                get_quant('a', bit_a, norm),
                QConv2d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, padding=1, bias=False, bit_w=self.bit_w, bit_a=self.bit_a, norm=norm),
                ExpNorm2d(self.out_channel, affine=True)
            )
    def forward(self, x: torch.Tensor):
        x_ = x.clone()
        x = self.layers(x)
        x_ = self.proj(x_)
        return x+x_