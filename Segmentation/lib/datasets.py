import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
    
class CarvanaDataset(Dataset):
    # https://www.kaggle.com/c/carvana-image-masking-challenge
    # https://www.kaggle.com/fadillahadamsyah/carvana-segmentation
    def __init__(self, path_dataset, transform=None):
        self.dataset = path_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img = self.load_image(idx)
        mask = self.load_mask(idx)
        
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        
        # This is only for 1 class
        return img, mask.unsqueeze_(0)
    
    def load_image(self, idx):
        img = Image.open(self.dataset[idx][0])
        img = np.array(img, dtype=np.float32) / 255.
        
        return img
    
    def load_mask(self, idx):
        # This is only for 1 class
        mask = Image.open(self.dataset[idx][1])
        mask = np.array(mask, dtype=np.float32)
        
        return mask