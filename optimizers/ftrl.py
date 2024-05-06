# This file implements FTRL with linearized loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as OPT


class FTRL(OPT.Optimizer):
    def __init__(self, params, alpha:float=0.01) -> None:
        defaults = dict(alpha=alpha)
        super().__init__(params, defaults)

    def step(self, closure=None):
        self._cuda_graph_capture_health_check()
        loss = None
        if closure is not None:
            loss = closure()

        # Initialize the step count, if not initialized
        if hasattr(self,'num_step'):
            self.num_step = 1.0
        # Update the num of step
        self.num_step += 1

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    # Initialize grad sum if not already done
                    if hasattr(p, 'grad_sum') is False:
                        p.grad_sum = torch.zeros_like(p.grad.data)
                    # Update the grad sum
                    p.grad_sum += p.grad.data
                    # Update the parameters using FTRL
                    p.data.copy_(-p.grad_sum.div(group['alpha']*(self.num_step**0.5)))
        return loss