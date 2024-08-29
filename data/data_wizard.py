import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data
import torchvision as TV

# For divid dataset into train/val
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms):
        super(CustomDataset, self).__init__()
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = self.transforms(image)
        return image, label


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
        test_set = TV.datasets.MNIST(root='./data/_mnist', train=False, transform=test_trans, download=True)
    elif name=='cifar100':
        train_trans = TV.transforms.Compose([
            TV.transforms.Resize(40),
            TV.transforms.RandomCrop(32),
            TV.transforms.RandomRotation(5),
            TV.transforms.ToTensor()
        ])
        test_trans = TV.transforms.Compose([
            TV.transforms.Resize(32),
            TV.transforms.ToTensor()
        ])
        train_set = TV.datasets.CIFAR100(root='./data/_cifar100', train=True, transform=train_trans, download=True)
        test_set = TV.datasets.CIFAR100(root='./data/_cifar100', train=False, transform=test_trans, download=True)
    elif name=='cifar10':
        train_trans = TV.transforms.Compose([
            TV.transforms.RandomCrop(32, padding=4),
            TV.transforms.RandomHorizontalFlip(),
            TV.transforms.AutoAugment(TV.transforms.AutoAugmentPolicy.CIFAR10),
            TV.transforms.ToTensor(),
            TV.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        test_trans = TV.transforms.Compose([
            TV.transforms.Resize(32),
            TV.transforms.ToTensor(),
            TV.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train_set = TV.datasets.CIFAR10(root='./data/_cifar10', train=True, transform=train_trans, download=True)
        test_set = TV.datasets.CIFAR10(root='./data/_cifar10', train=False, transform=test_trans, download=True)
    elif name=='caltech101':
        train_trans = TV.transforms.Compose([
            TV.transforms.Resize(300),
            TV.transforms.RandomCrop(256),
            TV.transforms.RandomRotation(5),
            TV.transforms.ToTensor()
        ])
        test_trans = TV.transforms.Compose([
            TV.transforms.Resize(256),
            TV.transforms.ToTensor()
        ])
        c101 = TV.datasets.Caltech101(root='./data/_caltech101', download=True)
        n_test = int(0.1 * len(c101))
        n_train = len(c101) - n_test
        train_set, test_set = torch.utils.data.random_split(c101, [n_train, n_test])
        train_set, test_set = CustomDataset(train_set, train_trans), CustomDataset(test_set, test_trans)
    elif name=='imagenet':
        train_trans = TV.transforms.Compose([
            TV.transforms.AutoAugment(TV.transforms.AutoAugmentPolicy.IMAGENET),
            TV.transforms.Resize([224,224]),
            TV.transforms.ToTensor()
        ])
        test_trans = TV.transforms.Compose([
            TV.transforms.Resize([224,224]),
            TV.transforms.ToTensor()
        ])
        train_set = TV.datasets.ImageFolder(root="/dataset/imagenet/train", transform=train_trans)
        test_set = TV.datasets.ImageFolder(root="/dataset/imagenet/val", transform=test_trans)
    else:
        raise NotImplementedError
    
    # Split the val set from the train set
    if (val_par>0)&(val_par<1):
        n_train = int((1-val_par)*len(train_set))
        n_val = len(train_set) - n_train
        train_set, val_set = torch.utils.data.random_split(train_set, [n_train, n_val])
    
    # Prepare the dataloader
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=3)
    val_loader = DataLoader(val_set, 200, shuffle=False, num_workers=3)
    test_loader = DataLoader(test_set, 200, shuffle=False, num_workers=3)
    return train_loader, val_loader, test_loader