import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CNN_Online import *
from models.MobileNet import *
from models.MobileNet_Online import *

def model_wizard(
        dataset:str="mnist", 
        bit_w:int=32, 
        bit_a:int=32, 
        version:str='V1', 
        device=torch.device('cpu'),
        online=True
    ):
    if dataset=='mnist':
        if (bit_w==32)&(bit_a==32)&(version=='V1')&online:
            return CNN_online_MNIST_V1().to(device)
        elif (bit_w==1)&(bit_a==1)&(version=='V1')&online:
            return CNN_online_MNIST_W1A1_V1().to(device)
        elif (bit_w in [2, 4, 8])|(bit_a in [2, 4, 8])&online:
            return CNN_online_MNIST_Quant_V1(bit_w=bit_w, bit_a=bit_a).to(device)
        else:
            raise NotImplementedError
    elif dataset=='cifar100':
        if (bit_w==32)&(bit_a==32)&online:
            return MobileNetV1_online_c100().to(device)
        elif (bit_w in [2, 4, 8])|(bit_a in [2, 4, 8])&online:
            return MobileNetV1_online_c100_Quant(bit_w=bit_w, bit_a=bit_a).to(device)
        elif (bit_w==32)&(bit_a==32):
            return MobileNetV1_c100().to(device)
        elif (bit_w in [2, 4, 8])|(bit_a in [2, 4, 8]):
            return MobileNetV1_c100_Quant(bit_w=bit_w, bit_a=bit_a).to(device)
    elif dataset=='caltech101':
        if (bit_w==32)&(bit_a==32)&online:
            raise NotImplementedError
        elif (bit_w==32)&(bit_a==32):
            return MobileNetV1_calt101().to(device)
    else:
        raise NotImplementedError