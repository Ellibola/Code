import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CNN_Online import *

def model_wizard(
        dataset:str="mnist", 
        bit_w:int=32, 
        bit_a:int=32, 
        version:str='V1', 
        device=torch.device('cpu')
    ):
    if dataset=='mnist':
        if (bit_w==32)&(bit_a==32)&(version=='V1'):
            return CNN_online_MNIST_V1().to(device)
        elif (bit_w==1)&(bit_a==1)&(version=='V1'):
            return CNN_online_MNIST_W1A1_V1().to(device)
        elif (bit_w in [2, 4, 8])|(bit_a in [2, 4, 8]):
            return CNN_online_MNIST_Quant_V1(bit_w=bit_w, bit_a=bit_a).to(device)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError