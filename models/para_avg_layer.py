import torch
import torch.nn as nn 
import torch.nn.functional as F

class AConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | torch.Tuple[int], stride: int | torch.Tuple[int] = 1, padding: str | int | torch.Tuple[int] = 0, dilation: int | torch.Tuple[int] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None, gamma=0.99):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.weight_avg = self.weight.data.detach().clone()
        self.gamma = gamma

    def forward(self, x:torch.Tensor):
        # During training
        if self.training:
            # Update the weight
            self.weight_avg = self.weight_avg * self.gamma + (1 - self.gamma) * self.weight.data.detach()
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(x, self.weight_avg, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
class ALinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, gamma=0.99) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.weight_avg = self.weight.data.detach().clone()
        if self.bias:
            self.bias_avg = self.bias.data.detach().clone()
        self.gamma = gamma

    def forward(self, x:torch.Tensor):
        # During training
        if self.training:
            # Update the weight and bias
            self.weight_avg = self.weight_avg * self.gamma + (1 - self.gamma) * self.weight.data.detach()
            if self.bias:
                self.bias_avg = self.bias_avg * self.gamma + (1 - self.gamma) * self.bias.data.detach()
            return F.linear(x, self.weight, self.bias)
        else:
            return F.linear(x, self.weight_avg, self.bias)
        
class AInstanceNorm2d(nn.InstanceNorm2d):
    def __init__(self, num_features: int, eps: float = 0.00001, momentum: float = 0.1, affine: bool = False, track_running_stats: bool = False, device=None, dtype=None, gamma=0.99) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)
        if self.affine:
            self.weight_avg = self.weight.data.detach().clone()
            self.bias_avg = self.bias.data.detach().clone()
        self.gamma = gamma

    def forward(self, x:torch.Tensor):
        # During training
        if self.training:
            # Update the weight and bias
            if self.affine:
                self.weight_avg = self.weight_avg * self.gamma + (1 - self.gamma) * self.weight.data.detach()
                self.bias_avg = self.bias_avg * self.gamma + (1 - self.gamma) * self.bias.data.detach()
            return F.instance_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats, self.momentum, self.eps
            )
        else:
            return F.instance_norm(
                input, self.running_mean, self.running_var, self.weight_avg, self.bias_avg,
                self.training or not self.track_running_stats, self.momentum, self.eps
            )

class AInstanceNorm1d(nn.InstanceNorm1d):
    def __init__(self, num_features: int, eps: float = 0.00001, momentum: float = 0.1, affine: bool = False, track_running_stats: bool = False, device=None, dtype=None, gamma=0.99) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)
        if self.affine:
            self.weight_avg = self.weight.data.detach().clone()
            self.bias_avg = self.bias.data.detach().clone()
        self.gamma = gamma

    def forward(self, x:torch.Tensor):
        # During training
        if self.training:
            # Update the weight and bias
            if self.affine:
                self.weight_avg = self.weight_avg * self.gamma + (1 - self.gamma) * self.weight.data.detach()
                self.bias_avg = self.bias_avg * self.gamma + (1 - self.gamma) * self.bias.data.detach()
            return F.instance_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats, self.momentum, self.eps
            )
        else:
            return F.instance_norm(
                input, self.running_mean, self.running_var, self.weight_avg, self.bias_avg,
                self.training or not self.track_running_stats, self.momentum, self.eps
            )