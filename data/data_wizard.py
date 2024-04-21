import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data
import torchvision as TV


def data_wizard(name:str, batch_size:int, val_par:float | None):
    # Compose the dataset
    if name=='mnist':
        train_trans = TV.transforms.Compose([
            TV.transforms.Resize(32),
            TV.transforms.RandomCrop(28),
            TV.transforms.RandomRotation(5),
            TV.transforms.ToTensor()
        ])
        test_trans = TV.transforms.Compose([
            TV.transforms.Resize(28),
            TV.transforms.ToTensor()
        ])
        train_set = TV.datasets.MNIST(root='./data/_mnist', train=True, transform=train_trans, download=True)
        test_set = TV.datasets.MNIST(root='./data/_mnist', train=True, transform=test_trans, download=True)
            
    else:
        raise NotImplementedError
    
    # Split the val set from the train set
    if (val_par>0)&(val_par<1):
        n_train = int((1-val_par)*len(train_set))
        n_val = len(train_set) - n_train
        train_set, val_set = torch.utils.data.random_split(train_set, [n_train, n_val])
    
    # Prepare the dataloader
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, 200, shuffle=False)
    test_loader = DataLoader(test_set, 200, shuffle=False)
    return train_loader, val_loader, test_loader