import numpy as np
import pandas as pd
import torch
import cv2

# https://www.kaggle.com/balraj98/cvcclinicdb
class CVCClinicDB(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get the image & mask path
        row = self.df.iloc[idx]
        
        # Open the files        
        image = cv2.cvtColor(cv2.imread(row['image']), cv2.COLOR_BGR2RGB) / 255.
        mask = self.encode_mask(cv2.imread(row['mask'], cv2.IMREAD_GRAYSCALE) / 255.)
        
        # Transformation
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask[None,:,:]
    
    def encode_mask(self, mask):
        mask[mask >= 0.5] = 1.
        mask[mask < 0.5] = 0.
        
        return mask.astype('float32')