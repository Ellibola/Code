import torch
import torch.nn as nn
import torch.nn.functional as F

class NN_Online(nn.Module):
    """
        Online deep learning with standard backpropagation
    """
    def __init__(self) -> None:
        super(NN_Online, self).__init__()
        self.features, self.classifiers = self._module_compose()

    def _module_compose(self)->list[list[nn.Module], list[nn.Module]]:
        raise NotImplementedError
    
    def set_hyper_params(self, beta:float, s:float):
        pass
    
    def forward(self, x:torch.Tensor):
        for module in self.features:
            x = module(x)
        return self.classifiers[-1](x)
    
    def step(self, x: torch.Tensor, target: torch.Tensor, optimizer: torch.optim.Optimizer):
        # Update trainable parameters
        self.train()
        optimizer.zero_grad()
        output = self.forward(x)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()