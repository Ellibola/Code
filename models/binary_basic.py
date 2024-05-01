# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

"""
    Auto grad functions
"""
class Bin_w(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input_b = torch.sign(input)
        return input_b

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_inputs = torch.sign(grad_outputs)
        # grad_inputs = grad_outputs
        return grad_inputs
# -1, 1 version
class Bin_a(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input_b = torch.sign(input)
        ctx.save_for_backward(input.detach())
        return input_b

    @staticmethod
    def backward(ctx, grad_outputs):
        input, = ctx.saved_tensors
        center = (input.abs()<1).float()
        decay = (1.0-(input).abs()).mul(5.0).exp()
        return grad_outputs * (center + (1-center) * decay)
    
class Bin_a_kd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input_b = torch.sign(input)
        return input_b

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs

"""
    Pytorch module version
"""
class Bin_w_m(nn.Module):
    def __init__(self) -> None:
        super(Bin_w_m, self).__init__()
        self.bin_fun = Bin_w.apply
    def forward(self, x:torch.Tensor):
        return self.bin_fun(x)

class Bin_a_m(nn.Module):
    def __init__(self) -> None:
        super(Bin_a_m, self).__init__()
        self.bin_fun = Bin_a.apply
    def forward(self, x:torch.Tensor):
        return self.bin_fun(x)

class Bin_a_kd_m(nn.Module):
    def __init__(self) -> None:
        super(Bin_a_kd_m, self).__init__()
        self.bin_fun = Bin_a_kd.apply
    def forward(self, x:torch.Tensor):
        return self.bin_fun(x)

# Binary functions
def get_bin_fun(target='w', return_type='apply'):
    
        
    if return_type=='apply':
        return Bin_w.apply if target=='w' else (Bin_a_kd.apply if target=='ag' else Bin_a.apply)
    else:
        return Bin_w_m() if target=='w' else (Bin_a_kd_m() if target=='ag' else Bin_a_m())

class BConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bin_w = get_bin_fun('w')
        # self.bin_a = get_bin_fun('a')
        # This fanin dose not consider dilation
        self.fanin = (self.kernel_size[0]*self.kernel_size[1]) * self.in_channels / self.groups
        assert self.fanin!=0, "Invalid fanin of this layer:{}".format(self.__name__)
    def forward(self, input: Tensor) -> Tensor:
        b_weight = self.bin_w(self.weight) if self.bias==None else self.bin_w(self.weight) / self.fanin
        # b_input = self.bin_a(input)
        return F.conv2d(input, b_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class BLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BLinear, self).__init__(in_features, out_features, bias)
        self.bin_w = get_bin_fun('w')
        # self.bin_a = get_bin_fun('a')
        self.fanin = self.in_features
        assert self.fanin!=0, "Invalid fanin of this layer:{}".format(self.__name__)
    def forward(self, input: Tensor) -> Tensor:
        b_weight = self.bin_w(self.weight) if self.bias==None else self.bin_w(self.weight) / self.fanin
        # b_input = self.bin_a(input)
        return F.linear(input, b_weight, self.bias)

class BConv2d_DS(nn.Module):
    """ Depth seperable Conv layers """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        # super(BConv2d_DS, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)
        super(BConv2d_DS, self).__init__()
        # Depth wise convolution
        self.conv_dw = BConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=in_channels
        )
        self.bn_dw = nn.BatchNorm2d(in_channels, affine=True)
        self.conv_pw = BConv2d(
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