# This file implements FTRL with linearized loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as OPT

class FTRL(OPT.Optimizer):
    def __init__(self, params, alpha:float=0.01) -> None:
        defaults = dict(alpha=alpha)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):
        self._cuda_graph_capture_health_check()
        loss = None
        if closure is not None:
            loss = closure()

        # Initialize the step count, if not initialized
        if not hasattr(self,'num_step'):
            self.num_step = 1.0
        # Update the num of step
        self.num_step += 1

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    param_state = self.state[p]
                    # Initialize grad sum if not already done
                    if 'grad_sum' not in param_state:
                        param_state['grad_sum'] = torch.zeros(p.data.shape, device=p.device)
                    # Update the grad sum
                    grad_sum = param_state['grad_sum']
                    grad_sum.add_(p.grad.data)
                    # Update the parameters using FTRL
                    p.data.copy_(-grad_sum.div(group['alpha']*(self.num_step**0.5)))
        return loss