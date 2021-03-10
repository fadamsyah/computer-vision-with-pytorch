import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100

class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, root='./data', download=True, train_transform=None, val_transform=None):
        super().__init__()
        
        self.root = root
        self.download = download
        self.train_transform = train_transform
        self.val_transform = val_transform
    
    def prepare_data(self):
        self.dataset = {
            'train': CIFAR100(
                root=self.root, train=True, download=self.download, transform=self.train_transform
            ),
            'val': CIFAR100(
                root=self.root, train=False, download=self.download, transform=self.val_transform
            )
        }
        
    def train_dataloader(self):
        ...
    
    def val_dataloader(self):
        ...
        
        