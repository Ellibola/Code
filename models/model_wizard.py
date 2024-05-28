import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CNN_Online import *
from models.CNN import *
from models.MobileNet import *
from models.MobileNet_Online import *

def model_wizard(
        dataset:str="mnist", 
        bit_w:int=32, 
        bit_a:int=32, 
        version:str='V1', 
        device=torch.device('cpu'),
        online=True,
        **kwargs
    ):
    if dataset=='mnist':
        if (bit_w==32)&(bit_a==32)&(version=='V1')&online:
            return CNN_online_MNIST_V1().to(device)
        elif (bit_w==1)&(bit_a==1)&(version=='V1')&online:
            return CNN_online_MNIST_W1A1_V1().to(device)
        elif ((bit_w in [2, 4, 8, 16])|(bit_a in [2, 4, 8, 16]))&online&(version=='V1'):
            return CNN_online_MNIST_Quant_V1(bit_w=bit_w, bit_a=bit_a).to(device)
        elif (bit_w==32)&(bit_a==32)&(version=='V1'):
            return CNN_MNIST_V1().to(device)
        elif (bit_w==32)&(bit_a==32)&(version=='V2')&online:
            return CNN_online_MNIST_V2().to(device)
        else:
            raise NotImplementedError
    elif dataset=='cifar100':
        if (bit_w==32)&(bit_a==32)&online:
            if (kwargs['if_avg'] if "if_avg" in kwargs.keys() else False):
                return MobileNetV1_online_c100_avg(kwargs['gamma']).to(device)
            return MobileNetV1_online_c100().to(device)
        elif (bit_w in [2, 4, 8, 16])|(bit_a in [2, 4, 8, 16])&online:
            return MobileNetV1_online_c100_Quant(bit_w=bit_w, bit_a=bit_a).to(device)
        elif (bit_w==32)&(bit_a==32):
            if (kwargs['if_insnorm'] if "if_insnorm" in kwargs.keys() else False):
                return MobileNetV1_c100_insnorm().to(device)
            return MobileNetV1_c100().to(device)
        elif (bit_w in [2, 4, 8, 16])|(bit_a in [2, 4, 8, 16]):
            return MobileNetV1_c100_Quant(bit_w=bit_w, bit_a=bit_a).to(device)
    elif dataset=='caltech101':
        if (bit_w==32)&(bit_a==32)&online:
            raise NotImplementedError
        elif (bit_w==32)&(bit_a==32):
            return MobileNetV1_calt101().to(device)
    elif dataset=='imagenet':
        if (bit_w==32)&(bit_a==32)&online&(version=='V1'):
            return MobileNetV1_online_imagenet().to(device)
        elif (bit_w==32)&(bit_a==32)&(version=='V1'):
            return MobileNetV1_imagenet().to(device)
        elif (bit_w==32)&(bit_a==32)&online&(version=='V2'):
            return MobileNetV1_online_imagenet_V2().to(device)
        elif (bit_w==32)&(bit_a==32)&online&(version=='V3'):
            return MobileNetV1_online_imagenet_V3().to(device)
    else:
        raise NotImplementedError