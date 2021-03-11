# https://github.com/DebeshJha/2020-CBMS-DoubleU-Net/blob/master/model.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

class ASPP(pl.LightningModule):
    def __init__(self, in_channels, out_channels, input_shape):
        super().__init__()
        
        self._avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._conv0 = nn.Conv2d(in_channels, out_channels, 1, padding=0,
                                stride=1, bias=False)
        self._bn0 = nn.BatchNorm2d(out_channels)
        self._up0 = nn.UpsamplingBilinear2d(input_shape)
        
        self._conv1 = nn.Conv2d(in_channels, out_channels, 1, padding=0,
                                stride=1, dilation=1, bias=False)
        self._bn1 = nn.BatchNorm2d(out_channels)
        
        self._conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6,
                                stride=1, dilation=6, bias=False)
        self._bn2 = nn.BatchNorm2d(out_channels)

        self._conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12,
                                stride=1, dilation=12, bias=False)
        self._bn3 = nn.BatchNorm2d(out_channels)
        
        self._conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18,
                                stride=1, dilation=18, bias=False)
        self._bn4 = nn.BatchNorm2d(out_channels)
        
        self._conv5 =  nn.Conv2d(5*out_channels, out_channels, 1, padding=0,
                                stride=1, dilation=1, bias=False)
        self._bn5 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        y0 = self._avgpool(x)
        y0 = self._conv0(y0)
        y0 = self._bn0(y0)
        y0 = F.relu(y0)
        y0 = self._up0(y0)
        
        y1 = self._conv1(x)
        y1 = self._bn1(y1)
        y1 = F.relu(y1)
        
        y2 = self._conv2(x)
        y2 = self._bn2(y2)
        y2 = F.relu(y2)
        
        y3 = self._conv3(x)
        y3 = self._bn3(y3)
        y3 = F.relu(y3)
        
        y4 = self._conv4(x)
        y4 = self._bn4(y4)
        y4 = F.relu(y4)
        
        y5 = torch.cat([y0, y1, y2, y3, y4], dim=1)
        y5 = self._conv5(y5)
        y5 = self._bn5(y5)
        y5 = F.relu(y5)
        
        return y5
