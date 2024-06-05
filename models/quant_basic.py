# Pytorch
from typing import Any, Union
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
import torch.nn.functional as F
from torch import Tensor
# Math
import math

A_QUANT_METHOD = 'LSQ'  # WARNING: you also need to change accordingly in file utils.py. It should be either 'PACT' or 'LSQ'

class UniQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:Tensor, bit:int):
        s = 2**bit-1
        return (x*s).round()/s
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:Tensor):
        return x
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs
    
class LSQ_gscale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:Tensor, scale:float):
        ctx.other = scale
        return x
    @staticmethod
    def backward(ctx, grad_outputs):
        scale = ctx.other
        return grad_outputs.mul(scale), None

class UniQ_w(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:Tensor, bit:int):
        s = 2**bit-1
        return (x*s).round()/s
    @staticmethod
    def backward(ctx, grad_outputs:Tensor):
        return grad_outputs.sign(), None

class WQ_DoReFa(nn.Module):
    def __init__(self, bit=8) -> None:
        super(WQ_DoReFa, self).__init__()
        self.bit = bit
        self.quant = UniQ().apply
    
    def forward(self, x:Tensor):
        x = x.tanh()
        x = x.div(x.abs().max()+1e-8).mul(0.5) + 0.5
        x = 2 * self.quant(x, self.bit) - 1
        return x


class AQ_LSQ(nn.Module):
    def __init__(self, bit:int) -> None:
        super(AQ_LSQ, self).__init__()
        self.bit = bit
        self.Qp = 2 ** self.bit - 1
        self.step_size = nn.Parameter(torch.tensor(0.0))
        self.init_flag = True
        self.ste = STE().apply
        self.gscale = LSQ_gscale().apply
    
    def forward(self, x:torch.Tensor):
        if self.init_flag:
            self.step_size.data = 2 * x.detach().abs().mean() / math.sqrt(self.Qp)
            self.init_flag = False
        step_size = self.gscale(self.step_size, 1 / math.sqrt(self.Qp * x.shape[1:].numel())).abs()
        x = self.ste(x.div(step_size + 1e-12))
        x = torch.clamp(x, 0, self.Qp)
        return x.mul(step_size + 1e-12)


class AQ_PACT(nn.Module):
    def __init__(self, bit=8, norm=False):
        super(AQ_PACT, self).__init__()
        self.bit = bit
        self.quant = UniQ().apply
        self.th = nn.Parameter(torch.tensor(10.0))

    def forward(self, x: Tensor):
        x = x.relu()
        x = torch.where(x>self.th.abs(), self.th.abs(), x)
        x = self.quant(x.div(self.th.abs()+1e-8), self.bit).mul(self.th.abs()+1e-8)
        return x

# Get the corresponding quantization function
def get_quant(target='w', bit=8, norm=False):
    assert bit>1, "For binary quantization please use binarization functions!"
    if target=='w':
        return WQ_DoReFa(bit)
    if target=='a':
        return AQ_LSQ(bit) if A_QUANT_METHOD=='LSQ' else AQ_PACT(bit)



"""
    Quantized layers
""" 
class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, bit_w=8, bit_a=8, norm=False) -> None:
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bit_w = bit_w
        self.bit_a = bit_a
        self.quant_w = get_quant('w', self.bit_w)
        # self.quant_a = get_quant('a', self.bit_a, norm) if bit_a != 1 else get_bin_fun('a')
        # This fanin dose not consider dilation
        self.fanin = (self.kernel_size[0]*self.kernel_size[1]) * self.in_channels / self.groups
        assert self.fanin!=0, "Invalid fanin of this layer:{}".format(self.__name__)

    def forward(self, x: Tensor) -> Tensor:
        q_w = self.quant_w(self.weight)
        # q_a = self.quant_a(x)
        return F.conv2d(x, q_w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class QConv2d_wp(nn.Conv2d):
    "QConv2d with pooling layer merged"
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, bit_w=8, bit_a=8, norm=False) -> None:
        super(QConv2d_wp, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bit_w = bit_w
        self.bit_a = bit_a
        self.quant_w = get_quant('w', self.bit_w)
        # This fanin dose not consider dilation
        self.fanin = (self.kernel_size[0]*self.kernel_size[1]) * self.in_channels / self.groups
        assert self.fanin!=0, "Invalid fanin of this layer:{}".format(self.__name__)

    def forward(self, x: Tensor) -> Tensor:
        q_w = self.quant_w(self.weight)
        x = F.max_pool2d(x, 2, 2)
        return F.conv2d(x, q_w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bit_w=8, bit_a=8, norm=False):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.bit_w = bit_w
        self.bit_a = bit_a
        self.quant_w = get_quant('w', self.bit_w)
        # self.quant_a = get_quant('a', self.bit_a, norm) if bit_a != 1 else get_bin_fun('a')
        self.fanin = self.in_features
        assert self.fanin!=0, "Invalid fanin of this layer:{}".format(self.__name__)

    def forward(self, x: Tensor) -> Tensor:
        q_w = self.quant_w(self.weight) if self.bias==None else self.quant_w(self.weight) / self.fanin
        # q_a = self.quant_a(x)
        return F.linear(x, q_w, self.bias)