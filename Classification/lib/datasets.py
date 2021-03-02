import torchvision
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Dataset

def cifar10(root='./data', download=True, train_transform=None, val_transform=None):
    dataset = {
        'train': CIFAR10(
            root=root, train=True, download=download, transform=train_transform
        ),
        'val': CIFAR10(
            root=root, train=False, download=download, transform=val_transform
        )
    }        
        
    return dataset

def cifar100(root='./data', download=True, train_transform=None, val_transform=None):
    dataset = {
        'train': CIFAR100(
            root=root, train=True, download=download, transform=train_transform
        ),
        'val': CIFAR100(
            root=root, train=False, download=download, transform=val_transform
        )
    }        
        
    return dataset