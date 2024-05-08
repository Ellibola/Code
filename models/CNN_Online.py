import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hedge import NN_Online
from models.binary_basic import *
from models.quant_basic import *


class CNN_online_MNIST_V1(NN_Online):
    """
        V1 CNN model for MNIST
    """

    def _module_compose(self):
        features = [
            nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1, bias=False),
                nn.InstanceNorm2d(32, affine=True),
                nn.ReLU()
            ),
            # (32,28,28)
            nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1, bias=False),
                nn.InstanceNorm2d(32, affine=True),
                nn.ReLU(),
                nn.AvgPool2d(2,2)
            ),
            # (32,14,14)
            nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1, bias=False),
                nn.InstanceNorm2d(64, affine=True),
                nn.ReLU()
            ),
            # (64,14,14)
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1, bias=False),
                nn.InstanceNorm2d(64, affine=True),
                nn.ReLU(),
                nn.AvgPool2d(2,2)
            ),
            # (64,7,7)
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1, bias=False),
                nn.InstanceNorm2d(128, affine=True),
                nn.ReLU()
            ),
            # (128,7,7)
            nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1, bias=False),
                nn.InstanceNorm2d(128, affine=True),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            # (128,1,1)
        ]
        classifiers = [
            nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(32*28*28, 10, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(32*14*14, 10, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(64*14*14, 10, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(64*7*7, 10, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(128*7*7, 10, bias=False)
            ),
            nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(128, 10, bias=True)
            ),
        ]
        return nn.ModuleList(features), nn.ModuleList(classifiers)
    

class CNN_online_MNIST_W1A1_V1(NN_Online):
    """
        Binarized V1 CNN model for MNIST
    """
    
    def _module_compose(self):
        features = [
            nn.Sequential(
                BConv2d(1, 32, 3, padding=1, bias=False),
                nn.InstanceNorm2d(32, affine=True),
                get_bin_fun('a', return_type='class')
            ),
            # (32,28,28)
            nn.Sequential(
                BConv2d(32, 32, 3, padding=1, bias=False),
                nn.InstanceNorm2d(32, affine=True),
                nn.MaxPool2d(2,2),
                get_bin_fun('a', return_type='class')
            ),
            # (32,14,14)
            nn.Sequential(
                BConv2d(32, 64, 3, padding=1, bias=False),
                nn.InstanceNorm2d(64, affine=True),
                get_bin_fun('a', return_type='class')
            ),
            # (64,14,14)
            nn.Sequential(
                BConv2d(64, 64, 3, padding=1, bias=False),
                nn.InstanceNorm2d(64, affine=True),
                nn.MaxPool2d(2,2),
                get_bin_fun('a', return_type='class')
            ),
            # (64,7,7)
            nn.Sequential(
                BConv2d(64, 128, 3, padding=1, bias=False),
                nn.InstanceNorm2d(128, affine=True),
                get_bin_fun('a', return_type='class')
            ),
            # (128,7,7)
            nn.Sequential(
                BConv2d(128, 128, 3, padding=1, bias=False),
                nn.InstanceNorm2d(128, affine=True),
                nn.AdaptiveAvgPool2d(1),
                get_bin_fun('a', return_type='class')
            )
            # (128,1,1)
        ]
        classifiers = [
            nn.Sequential(
                nn.Flatten(start_dim=1),
                BLinear(32*28*28, 10, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(start_dim=1),
                BLinear(32*14*14, 10, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(start_dim=1),
                BLinear(64*14*14, 10, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(start_dim=1),
                BLinear(64*7*7, 10, bias=True)
            ),
            nn.Sequential(
                nn.Flatten(start_dim=1),
                BLinear(128*7*7, 10, bias=False)
            ),
            nn.Sequential(
                nn.Flatten(start_dim=1),
                BLinear(128, 10, bias=True)
            ),
        ]
        return nn.ModuleList(features), nn.ModuleList(classifiers)
    
class CNN_online_MNIST_Quant_V1(NN_Online):
    """
        Quantized V1 CNN model for MNIST
    """
    def __init__(self, bit_w, bit_a) -> None:
        self.bit_w = bit_w
        self.bit_a = bit_a
        super().__init__()

    def _module_compose(self):
        features = [
            nn.Sequential(
                QConv2d(1, 32, 3, padding=1, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(32, affine=True),
                get_quant('a', bit=self.bit_a)
            ),
            # (32,28,28)
            nn.Sequential(
                QConv2d(32, 32, 3, padding=1, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(32, affine=True),
                nn.MaxPool2d(2,2),
                get_quant('a', bit=self.bit_a)
            ),
            # (32,14,14)
            nn.Sequential(
                QConv2d(32, 64, 3, padding=1, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(64, affine=True),
                get_quant('a', bit=self.bit_a)
            ),
            # (64,14,14)
            nn.Sequential(
                QConv2d(64, 64, 3, padding=1, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(64, affine=True),
                nn.MaxPool2d(2,2),
                get_quant('a', bit=self.bit_a)
            ),
            # (64,7,7)
            nn.Sequential(
                QConv2d(64, 128, 3, padding=1, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(128, affine=True),
                get_quant('a', bit=self.bit_a)
            ),
            # (128,7,7)
            nn.Sequential(
                QConv2d(128, 128, 3, padding=1, bias=False, bit_w=self.bit_w),
                nn.InstanceNorm2d(128, affine=True),
                nn.AdaptiveAvgPool2d(1),
                get_quant('a', bit=self.bit_a)
            )
            # (128,1,1)
        ]
        classifiers = [
            nn.Sequential(
                nn.Flatten(start_dim=1),
                QLinear(32*28*28, 10, bias=True, bit_w=self.bit_w)
            ),
            nn.Sequential(
                nn.Flatten(start_dim=1),
                QLinear(32*14*14, 10, bias=True, bit_w=self.bit_w)
            ),
            nn.Sequential(
                nn.Flatten(start_dim=1),
                QLinear(64*14*14, 10, bias=True, bit_w=self.bit_w)
            ),
            nn.Sequential(
                nn.Flatten(start_dim=1),
                QLinear(64*7*7, 10, bias=True, bit_w=self.bit_w)
            ),
            nn.Sequential(
                nn.Flatten(start_dim=1),
                QLinear(128*7*7, 10, bias=False, bit_w=self.bit_w)
            ),
            nn.Sequential(
                nn.Flatten(start_dim=1),
                QLinear(128, 10, bias=True, bit_w=self.bit_w)
            ),
        ]
        return nn.ModuleList(features), nn.ModuleList(classifiers)