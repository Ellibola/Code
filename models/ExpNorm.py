import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=1e-3, affine=True):
        super(ExpNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_sqr_mean', torch.zeros(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        nn.init.zeros_(self.running_mean)
        nn.init.zeros_(self.running_sqr_mean)
        self.num_batches_tracked.zero_()
        
    def forward(self, x: torch.Tensor):
        assert len(x.shape)==4, "The input should be 2D features, got dimension:{}".format(x.shape)
        if self.training:
            # Update the statistics
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x.mean(dim=[0,2,3])
                self.running_sqr_mean = (1 - self.momentum) * self.running_sqr_mean + self.momentum * x.square().mean(dim=[0,2,3])
                self.num_batches_tracked += 1
                mean = self.running_mean
                var = self.running_sqr_mean - self.running_mean.square()
        else:
            mean = self.running_mean
            var = self.running_sqr_mean - self.running_mean.square()
        
        x = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        
        if self.affine:
            x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        
        return x
    
class ExpNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=1e-3, affine=True):
        super(ExpNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_sqr_mean', torch.zeros(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        nn.init.zeros_(self.running_mean)
        nn.init.zeros_(self.running_sqr_mean)
        self.num_batches_tracked.zero_()
        
    def forward(self, x: torch.Tensor):
        assert len(x.shape)==2, "The input should be 1D feature, got dimension:{}".format(x.shape)
        if self.training:
            # Update the statistics
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x
                self.running_sqr_mean = (1 - self.momentum) * self.running_sqr_mean + self.momentum * x.square()
                self.num_batches_tracked += 1
                mean = self.running_mean
                var = self.running_sqr_mean - self.running_mean.square()
        else:
            mean = self.running_mean
            var = self.running_sqr_mean - self.running_mean.square()
        
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        if self.affine:
            x = x * self.weight + self.bias
        
        return x