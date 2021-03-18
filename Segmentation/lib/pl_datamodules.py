import os
import pytorch_lightning as pl
import random
from torch.utils.data import DataLoader
from .datasets import CarvanaDataset

class CarvanaDataModule(pl.LightningDataModule):
    def __init__(self, dir_images, dir_masks, val_size=0.2,
                 transform={'train': None, 'val': None}):
        super().__init__()
        
        # Get corresponding image-mask pairs
        path_images = [os.path.join(dir_images, path)
                            for path in os.listdir(dir_images)]
        path_masks = [os.path.join(dir_masks, path)
                           for path in os.listdir(dir_masks)]
        self.dataset = self.images_to_labels(path_images,
                                             path_masks)
        
        # Shuffle the dataset
        random.seed(30)
        random.shuffle(self.dataset)

        # Save the pct of validation data
        self.val_size = val_size
        
        # Save the transformation
        self.transform = transform
        
        # Parameters of dataloaders
        self.params = {
            'train': {
                'batch_size': 16, 'shuffle': True,
                'num_workers': 4, 'drop_last': True,
                'pin_memory': True
            },
            'val': {
                'batch_size': 16, 'shuffle': False,
                'num_workers': 4, 'drop_last': False,
                'pin_memory': True
            }
        }
        
    def setup(self, stage=None):
        split = int(len(self.dataset) * (1. - self.val_size))
        self.carvana_dataset = {
            'train': CarvanaDataset(self.dataset[:split], transform=self.transform['train']),
            'val': CarvanaDataset(self.dataset[split:], transform=self.transform['val'])
        }
    
    def train_dataloader(self):
        return self.generate_dataloader('train')

    def val_dataloader(self):
        return self.generate_dataloader('val')
        
    def images_to_labels(self, images, masks):
        return [[img, mask] for img, mask in zip(sorted(images), sorted(masks))]    
    
    def generate_dataloader(self, stage):
        return DataLoader(self.carvana_dataset[stage],
                          batch_size=self.params[stage]['batch_size'],
                          shuffle=self.params[stage]['shuffle'],
                          num_workers=self.params[stage]['num_workers'],
                          drop_last=self.params[stage]['drop_last'],
                          pin_memory=self.params[stage]['pin_memory'])
        
    def configure_train_dataloader(self, **kwargs):
        self.configure_dataloader('train', kwargs)
    
    def configure_val_dataloader(self, **kwargs):
        self.configure_dataloader('val', kwargs)
        
    def configure_dataloader(self, stage, kwargs):
        for key, val in kwargs.items():
            if key in self.params[stage]:
                self.params[stage][key] = val