import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from .dataset import CVCClinicDB

# https://www.kaggle.com/balraj98/cvcclinicdb
class CVCDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, dir_path, val_size=0.2, transform_train=None, transform_val=None):
        self.df = pd.read_csv(csv_path)
        self.df = pd.DataFrame(data={'image': [os.path.join(dir_path, path) for path in self.df['png_image_path']],
                                     'mask': [os.path.join(dir_path, path) for path in self.df['png_mask_path']]})        
        self.transform = {
            'train': transform_train,
            'val': transform_val
        }
        self.val_size = val_size
        
    def setup(self, stage=None):
        self.df_train, self.df_val = train_test_split(self.df, test_size=self.val_size,
                                                      random_state=42)
        