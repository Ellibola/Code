import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import matplotlib.pyplot as plt

A_QUANT_METHOD = 'LSQ'  # WARNING: you also need to change accordingly in file quant_basics.py. It should be either 'PACT' or 'LSQ'

def get_para_group(model:nn.Module, weight_decay:float):
    """
        Batch normalization layers should not apply weight decay
    """
    params_decay = []
    params_other = []
    for name, param in model.named_parameters():
        # For LSQ, the quant param should not be decayed, while for PACT it should !
        if A_QUANT_METHOD == 'LSQ':
            if len(param.shape)<=1:
                print(name)
                params_other.append(param)
            else:
                params_decay.append(param)
        else:
            if len(param.shape)==1:
                print(name)
                params_other.append(param)
            else:
                params_decay.append(param)
    return [
        {'params': params_decay, 'weight_decay': weight_decay},
        {'params': params_other, 'weight_decay': 0.0}
    ]


def correct_count(output, labels):
    c_t1 = output.topk(1, dim=1).indices.squeeze().eq(labels).type(torch.float).sum().item()
    c_t5 = output.topk(5, dim=1).indices.eq(labels.view(-1,1)).type(torch.float).sum(dim=1).sum().item()
    num = len(labels)
    return c_t1, c_t5, num

def validation_t1_t5(model, testloader, device):
    model.eval()
    correct_t1 = 0
    correct_t5 = 0
    total_num = 0
    with torch.no_grad():
        for images, labels in testloader:
            output = F.softmax(model.forward(images.to(device)), dim=1)
            c_t1, c_t5, num = correct_count(output, labels.to(device))
            correct_t1 += c_t1
            correct_t5 += c_t5
            total_num += num

    acc_t1 = correct_t1 / total_num * 100
    acc_t5 = correct_t5 / total_num * 100

    return acc_t1, acc_t5

# Cosine annealing LR scheduler with warmup 
class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
        Taking from MQbench
    """
    def __init__(self, optimizer, max_iter, min_lr, base_lr, warmup_lr, warmup_steps, last_iter=-1):
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.warmup_steps = warmup_steps
        self.max_iter = max_iter
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch=last_iter)

    def get_lr(self):
        if self.last_epoch<self.warmup_steps:
            target_lr = (self.warmup_lr - self.base_lr) / (self.warmup_steps+1) * (self.last_epoch+1) + self.base_lr
            return [target_lr for _ in self.base_lrs]
        step_ratio = (self.last_epoch-self.warmup_steps) / (self.max_iter-self.warmup_steps)
        target_lr = self.min_lr + (self.warmup_lr - self.min_lr)*(1 + math.cos(math.pi * step_ratio)) / 2
        return [target_lr for _ in self.base_lrs]
    

# Model parameter counting
def para_count(model: nn.Module):
    count=0
    for para in model.parameters():
        count += para.data.nelement()
    return count