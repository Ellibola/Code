# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# Residule blocks
from models.res_blk_online import *
import math

RESNET_CONFIG = {
    10: [1, 1, 1, 1],
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]
}

CLASSIFIER_CONFIG = {
    32  :   {64:4, 128:4, 256:4, 512:1, 1024:1, 2048:1},
    256 :   {64:8, 128:4, 256:4, 512:1, 1024:1, 2048:1},
    224 :   {64:7, 128:3, 256:3, 512:1, 1024:1, 2048:1},
}

def get_online_resnet(ol_type):
    if ol_type=='exp3_jump_avg':
        from models.exp3_jump_avg import NN_Online
    elif ol_type=='plain_ol':
        from models.plain_ol import NN_Online
    else:
        raise NotImplementedError
    class ResNet_ol(NN_Online):
        """
            Base class for all sorts of full-precision ResNets with online learning
        """
        def __init__(self, 
                    in_dim: int, 
                    out_dim: int, 
                    device: torch.device, 
                    res_blk: BasicBlock_ins | BottleNeckBlock_ins, 
                    blk_nums: list[int]=[3, 4, 6, 3], 
                    second_layer_channel: int=256
            ) -> None:
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.device = device
            self.blk_nums = blk_nums
            self.res_blk = res_blk
            self.second_layer_channel = second_layer_channel
            self.lin_dim = self.second_layer_channel*8
            self.clf_config = CLASSIFIER_CONFIG[self.in_dim]
            # self.module_list = self.compose_layers().to(device)
            super(ResNet_ol, self).__init__()

        def _make_layers(self, in_chan, out_chan, blk_num, stride=2):
            layers = [
                nn.Sequential(
                    self.res_blk(in_chan, out_chan),
                    nn.InstanceNorm2d(out_chan, affine=True),
                    nn.ReLU()
                )
            ]
            classifiers = [
                nn.Sequential(
                    # nn.AdaptiveAvgPool2d(int(math.sqrt(self.lin_dim/out_chan))),
                    nn.AdaptiveAvgPool2d(self.clf_config[out_chan]),
                    nn.Flatten(),
                    nn.Linear(out_chan*self.clf_config[out_chan]**2, self.out_dim, bias=False),
                    nn.LayerNorm(self.out_dim)
                )
            ]
            for _ in range(1, blk_num):
                layers += [
                    nn.Sequential(
                        self.res_blk(out_chan, out_chan),
                        nn.InstanceNorm2d(out_chan, affine=True),
                        nn.ReLU()
                    )
                ]
                classifiers += [
                    nn.Sequential(
                        nn.AdaptiveAvgPool2d(self.clf_config[out_chan]),
                        nn.Flatten(),
                        # nn.Linear(self.lin_dim, self.out_dim, bias=False),
                        nn.Linear(out_chan*self.clf_config[out_chan]**2, self.out_dim, bias=False),
                        nn.LayerNorm(self.out_dim)
                    )
                ]
            return layers, classifiers
        
        def compose_layers(self) -> list:
            m_list = []
            classifiers = []
            # First layer
            if self.in_dim>32:
                m_list += [
                    nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
                        nn.InstanceNorm2d(64, affine=True),
                        nn.ReLU(),
                        nn.AvgPool2d(kernel_size=2, stride=2)
                    )
                ]
            else:
                m_list += [
                    nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.InstanceNorm2d(64, affine=True),
                        nn.ReLU()
                    )
                ]
            classifiers += [
                nn.Sequential(
                    # nn.AdaptiveAvgPool2d(int(math.sqrt(self.lin_dim/64))),
                    nn.AdaptiveAvgPool2d(self.clf_config[64]),
                    nn.Flatten(),
                    nn.Linear(64*self.clf_config[64]**2, self.out_dim, bias=False),
                    nn.LayerNorm(self.out_dim)
                )
            ]
            # Second layer
            f, c = self._make_layers(64, self.second_layer_channel, blk_num=self.blk_nums[0],stride=1)
            m_list += f
            classifiers += c
            # Third to fifth layer
            for i in range(1, 4):
                f, c = self._make_layers(self.second_layer_channel*2**(i-1), self.second_layer_channel*2**(i), self.blk_nums[i])
                m_list += f
                classifiers += c
            return m_list, classifiers
        
        def _module_compose(self):
            features, classifiers = self.compose_layers()
            return nn.ModuleList(features), nn.ModuleList(classifiers)
    
    return ResNet_ol
    
def resnet_ol(dataset='cifar100', device=torch.device('cpu'), num_layer=50 , ol_type='plain_ol'):
    assert num_layer in RESNET_CONFIG.keys(), "Unsupported number of layers({}) for ResNet".format(num_layer)
    in_dim = 32 if 'mnist' in dataset else \
             32 if 'cifar' in dataset else \
             224 if 'imagenet' in dataset else \
             224 if 'caltech' in dataset else \
             256
    out_dim = 10 if 'mnist' in dataset else \
              10 if dataset=='cifar10' else \
              100 if dataset=='cifar100' else \
              256 if 'caltech' in dataset else \
              1000 if 'imagenet' in dataset else \
              31
    blk_nums = RESNET_CONFIG[num_layer]
    if num_layer<50:
        res_blk = BasicBlock_ins
    else:
        res_blk = BottleNeckBlock_ins
    second_layer_channel = 64 if num_layer<50 else 256
    ResNet_ol = get_online_resnet(ol_type)
    return ResNet_ol(
        in_dim=in_dim, 
        out_dim=out_dim, 
        device=device, 
        res_blk=res_blk,
        blk_nums=blk_nums,
        second_layer_channel=second_layer_channel
        ).to(device)