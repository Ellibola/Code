import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CNN_Online import *
from models.CNN import *
from models.MobileNet import *
from models.MobileNet_Online import *
from models.VGG import *
from models.VGG_online import *
from models.resnet import resnet
from models.resnet_online import resnet_ol

def model_wizard(
        dataset:str="mnist", 
        bit_w:int=32, 
        bit_a:int=32, 
        version:str='V1', 
        device=torch.device('cpu'),
        online=True,
        **kwargs
    ):
    if 'ol_type' not in kwargs.keys():
        kwargs['ol_type'] = 'plain_ol'
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
        if (bit_w==32)&(bit_a==32)&online&(version=='V1'):
            if (kwargs['if_avg'] if "if_avg" in kwargs.keys() else False):
                return MobileNetV1_online_c100_avg(kwargs['gamma']).to(device)
            return MobileNetV1_online_c100().to(device)
        elif (bit_w==32)&(bit_a==32)&(version=='V1'):
            if (kwargs['if_insnorm'] if "if_insnorm" in kwargs.keys() else False):
                return MobileNetV1_c100_insnorm().to(device)
            return MobileNetV1_c100().to(device)
        elif ((bit_w in [2, 4, 8, 16, 24])|(bit_a in [2, 4, 8, 16, 24]))&online&(version=='V1'):
            return MobileNetV1_online_c100_Quant(bit_w=bit_w, bit_a=bit_a).to(device)
        elif ((bit_w in [2, 4, 8, 16, 24])|(bit_a in [2, 4, 8, 16, 24]))&online&(version=='V2'):
            return MobileNetV1_online_c100_Quant_V2_EXPAVGNorm(bit_w=bit_w, bit_a=bit_a).to(device)
        elif ((bit_w in [2, 4, 8, 16, 24])|(bit_a in [2, 4, 8, 16, 24]))&online&(version=='V3'):
            return MobileNetV1_online_c100_Quant_V3_FPclassifier(bit_w=bit_w, bit_a=bit_a).to(device)
        elif (bit_w==32)&(bit_a==32)&(version=='V4')&online:
            return VGG_c100_online().to(device)
        elif (bit_w==32)&(bit_a==32)&(version=='V4'):
            return VGG_c100().to(device)
        elif (bit_w==32)&(bit_a==32)&(version=='V5'):
            return VGG_c100_olnorm().to(device)
        elif ((bit_w in [2, 4, 8, 16])|(bit_a in [2, 4, 8, 16]))&(version=='V4'):
            return VGG_c100_Quant(bit_w=bit_w, bit_a=bit_a).to(device)
        elif ((bit_w in [2, 4, 8, 16])|(bit_a in [2, 4, 8, 16]))&(version=='V1'):
            return MobileNetV1_c100_Quant(bit_w=bit_w, bit_a=bit_a).to(device)
        elif (bit_w==32)&(bit_a==32)&('resnet' in version)&online:
            n_layer = int(version.replace('resnet',''))
            return resnet_ol('cifar100',device,n_layer,kwargs['ol_type'])
        elif (bit_w==32)&(bit_a==32)&('resnet' in version):
            if ('in' in version):
                n_layer = int(version.replace('inresnet',''))
                return resnet('cifar100', device, n_layer, bit_w=32, bit_a=32, norm='in')
            else:
                n_layer = int(version.replace('resnet',''))
                return resnet('cifar100', device, n_layer, bit_w=32, bit_a=32)
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
        elif (bit_w==32)&(bit_a==32)&online&('resnet' in version):
            n_layer = int(version.replace('resnet',''))
            return resnet_ol('imagenet',device,n_layer,kwargs['ol_type'])
    elif dataset=='cifar10':
        if (bit_w==32)&(bit_a==32)&online&(version=='vgg11'):
            return VGG_c10_online_plain().to(device)
        if (bit_w==32)&(bit_a==32)&online&(version=='vgg11_ol'):
            return VGG_c10_online().to(device)
        if (bit_w==32)&(bit_a==32)&online&(version=='vgg11_exp'):
            return VGG_c10_online_bn().to(device)
        if (bit_w==32)&(bit_a==32)&(version=='vgg11'):
            return VGG_c10_olnorm().to(device)
        if (bit_w==32)&(bit_a==32)&(version=='vgg11_bn'):
            return VGG_c10().to(device)
        if (bit_w==32)&(bit_a==32)&online&('resnet' in version):
            n_layer = int(version.replace('resnet',''))
            return resnet_ol('cifar10', device, n_layer, kwargs['ol_type'])
        if (bit_w==32)&(bit_a==32)&('resnet' in version):
            if 'inresnet' in version:
                n_layer = int(version.replace('inresnet',''))
                norm = 'in'
            else:
                n_layer = int(version.replace('resnet',''))
                norm = 'bn'
            return resnet('cifar10', device, n_layer, 32, 32, norm)
    else:
        raise NotImplementedError