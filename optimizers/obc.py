# This file implements FTRL with linearized loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as OPT

class OBC(OPT.Optimizer):
    def __init__(self, params, base_optimizer_class:OPT.SGD|OPT.AdamW, eta:float=0.99, **base_optimizer_args) -> None:
        defaults = dict(eta=eta, **base_optimizer_args)
        super().__init__(params, defaults)
        self.eta = eta
        self.shadow_params = []
        # Create shadow parameters and preserve the same parameter group structure
        for group in params:
            shadow_group = []
            for p in group['params']:
                shadow_param = p.clone().detach()
                shadow_param.requires_grad = False
                shadow_group.append(shadow_param)
            self.shadow_params.append({'params': shadow_group, **{k: v for k, v in group.items() if k != 'params'}})
        
        # Initialize the base optimizer with shadow parameters
        self.base_optimizer = base_optimizer_class(self.shadow_params, **base_optimizer_args)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):
        self._cuda_graph_capture_health_check()
        loss = None
        if closure is not None:
            loss = closure()

        # Copy gradients from the original parameters to the shadow parameters
        for group, shadow_group in zip(self.param_groups, self.shadow_params):
            for p, shadow_p in zip(group['params'], shadow_group['params']):
                if p.grad is not None:
                    shadow_p.grad = p.grad.clone().detach()
        
        # Perform the base optimizer step on shadow parameters
        self.base_optimizer.step()
        
        # Update original parameters
        for group, shadow_group in zip(self.param_groups, self.shadow_params):
            for p, shadow_p in zip(group['params'], shadow_group['params']):
                if p.grad is None:
                    continue
                
                # Update original parameters
                p.data = self.eta * p.data + (1 - self.eta) * shadow_p.data

        return loss