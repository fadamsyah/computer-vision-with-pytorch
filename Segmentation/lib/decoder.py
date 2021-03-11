# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

################################################################################
### Decoder for EfficientNet
################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

class ConvBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self._conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self._conv_block(x)

class Decoder(pl.LightningModule):
    def __init__(self, num_class=1, encoder_name='efficientnet-b2',
                 final_activation='sigmoid', aspp_out_channels=1024):
        super().__init__()
        
        self._final_activation = final_activation
        
        # Parameters
        params_dict = {
            # Coefficients: (res, filters, stem filters)
            'efficientnet-b2': ([130, 65, 33, 17, 17, 9, 9],
                                [16, 24, 48, 88, 120, 208, 352], 32)
        }
        self._params = params_dict[encoder_name]
        
        # Upsample block
        for i in range(len(self._params[0])):
            n = len(self._params[0]) - 1
            exec(f'self._up{i} = nn.UpsamplingBilinear2d((self._params[0][{n-i}], self._params[0][{n-i}]))')
        self._up7 = nn.UpsamplingBilinear2d((self._params[0][0] * 2, self._params[0][0] * 2))
        
        # Conv Block
        in_channels = aspp_out_channels + self._params[1][-1]
        out_channels = self._params[1][-1]
        self._conv_block0 = ConvBlock(in_channels, out_channels)
        for i in range(len(self._params[1]) - 1):
            in_channels = self._params[1][len(self._params[1])-1-i] + self._params[1][len(self._params[1])-1-i-1]
            out_channels = self._params[1][len(self._params[1])-1-i-1]
            exec(f'self._conv_block{i+1} = ConvBlock({in_channels}, {out_channels})')
        self._conv_block7 = ConvBlock(self._params[-1] + self._params[1][0], num_class)

    def forward(self, x0, x1, x2, x3, x4, x5, x6, x7, x8):
        y = self._conv_block0(torch.cat([self._up0(x0), x1], dim=1))
        y = self._conv_block1(torch.cat([self._up1(y), x2], dim=1))
        y = self._conv_block2(torch.cat([self._up2(y), x3], dim=1))
        y = self._conv_block3(torch.cat([self._up3(y), x4], dim=1))
        y = self._conv_block4(torch.cat([self._up4(y), x5], dim=1))
        y = self._conv_block5(torch.cat([self._up5(y), x6], dim=1))
        y = self._conv_block6(torch.cat([self._up6(y), x7], dim=1))
        y = self._conv_block7(self._up7(torch.cat([y, x8], dim=1)))
        
        if self._final_activation == 'sigmoid':
            y = torch.sigmoid(y)
        
        return y