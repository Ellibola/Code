import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, batch_size=32):
        super(ExpNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.batch_size = batch_size
        self.gotcha = False
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('batch_sum', torch.zeros(num_features))
        self.register_buffer('batch_sqr_sum', torch.zeros(num_features))
        self.register_buffer('num_batches_tracked', torch.zeros(1))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        nn.init.zeros_(self.running_mean)
        nn.init.ones_(self.running_var)
        nn.init.zeros_(self.batch_sum)
        nn.init.zeros_(self.batch_sqr_sum)
        nn.init.zeros_(self.num_batches_tracked)
        self.gotcha = False

    def forward(self, x: torch.Tensor):
        assert len(x.shape)==4, "The input should be 2D features, got dimension:{}".format(x.shape)
        with torch.no_grad():
            if self.training:
                # Update the batch statistics
                if self.num_batches_tracked<self.batch_size:
                    self.batch_sum += x.mean(dim=[0,2,3])
                    self.batch_sqr_sum += x.square().mean(dim=[0,2,3])
                    self.num_batches_tracked += 1
                else:
                    self.batch_sum = x.mean(dim=[0,2,3])
                    self.batch_sqr_sum = x.square().mean(dim=[0,2,3])
                    self.num_batches_tracked = torch.ones_like(self.num_batches_tracked)

                mean_b = self.batch_sum/self.num_batches_tracked if self.num_batches_tracked>1 else \
                         self.batch_sum
                sqr_b = self.batch_sqr_sum/self.num_batches_tracked if self.num_batches_tracked>1 else \
                        self.batch_sqr_sum
                var_b = sqr_b - mean_b.square()
                # Update running statistic when a batch finished
                if self.num_batches_tracked==self.batch_size:
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_b if self.gotcha else\
                                        mean_b
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_b if self.gotcha else\
                                        var_b
                    self.gotcha = True
                mean = mean_b
                var = var_b

            else:
                mean_b = self.batch_sum/self.num_batches_tracked if self.num_batches_tracked!=0 else torch.zeros_like(self.batch_sum).detach()
                sqr_b = self.batch_sqr_sum/self.num_batches_tracked if self.num_batches_tracked!=0 else torch.ones_like(self.batch_sqr_sum).detach()
                var_b = sqr_b - mean_b.square()
                mean = self.running_mean if self.gotcha else mean_b
                var = self.running_var if self.gotcha else var_b

        
        x = (x - mean[None, :, None, None].detach()) / torch.sqrt(var[None, :, None, None].detach() + self.eps)
        
        if self.affine:
            x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        
        return x
    
class ExpNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, batch_size=200):
        super(ExpNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.batch_size = batch_size
        self.gotcha = False

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.register_buffer('batch_sum', torch.zeros(num_features))
        self.register_buffer('batch_sqr_sum', torch.zeros(num_features))
        self.register_buffer('num_batches_tracked', torch.zeros(1))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        nn.init.zeros_(self.running_mean)
        nn.init.ones_(self.running_var)
        nn.init.zeros_(self.batch_sum)
        nn.init.zeros_(self.batch_sqr_sum)
        nn.init.zeros_(self.num_batches_tracked)
        self.gotcha = False
        
    def forward(self, x: torch.Tensor):
        assert len(x.shape)==2, "The input should be 1D feature, got dimension:{}".format(x.shape)
        with torch.no_grad():
            if self.training:
                # Update the batch statistics
                if self.num_batches_tracked<self.batch_size:
                    self.batch_sum += x.squeeze().clone().detach()
                    self.batch_sqr_sum += x.square().squeeze().clone().detach()
                    self.num_batches_tracked += 1
                else:
                    self.batch_sum = x.squeeze().clone().detach()
                    self.batch_sqr_sum = x.square().squeeze().clone().detach()
                    self.num_batches_tracked = torch.ones_like(self.num_batches_tracked)

                mean_b = self.batch_sum/self.num_batches_tracked if self.num_batches_tracked>1 else\
                         self.batch_sum
                sqr_b = self.batch_sqr_sum/self.num_batches_tracked if self.num_batches_tracked>1 else\
                        self.batch_sqr_sum
                var_b = sqr_b - mean_b.square() if self.num_batches_tracked>1 else torch.ones_like(mean_b).detach()

                if self.num_batches_tracked==self.batch_size:
                    # Update running statistic
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_b if self.gotcha else\
                                        mean_b
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_b if self.gotcha else\
                                        var_b
                    self.gotcha = True
                mean = mean_b
                var = torch.where(var_b==0, 1, var_b)

            else:
                mean_b = self.batch_sum/self.num_batches_tracked if self.num_batches_tracked!=0 else torch.zeros_like(self.batch_sum).detach()
                sqr_b = self.batch_sqr_sum/self.num_batches_tracked if self.num_batches_tracked!=0 else torch.ones_like(self.batch_sqr_sum).detach()
                var_b = sqr_b - mean_b.square() if self.num_batches_tracked>1 else torch.ones_like(sqr_b).detach()
                mean = self.running_mean if self.gotcha else mean_b
                var = self.running_var if self.gotcha else var_b
        assert (not torch.isnan(var).any())&(not torch.isinf(var).any())&(torch.positive(var).any()),"Var:{} is not valid, with :running var:{}, var_b:{}, n_batch:{}, mean_b_squre:{}, sqr_b:{}".format(var, self.running_var, var_b, self.num_batches_tracked,mean_b.square(),sqr_b)
        assert (not torch.isnan(mean).any())&(not torch.isinf(mean).any()),"Mean is not valid, with :running var:{}, var_b:{}".format(self.running_mean, mean_b)
        x = (x - mean.unsqueeze(0).detach()) / torch.sqrt(var.unsqueeze(0).detach() + self.eps)
        
        if self.affine:
            x = x * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        
        return x