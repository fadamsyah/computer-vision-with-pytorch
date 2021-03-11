# https://github.com/DebeshJha/2020-CBMS-DoubleU-Net/blob/master/model.py
# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

from .aspp import ASPP
from .decoder import Decoder
from .efficientnet import EfficientNet

class SegmentationModel(pl.LightningModule):
    def __init__(self, encoder_name='efficientnet-b2', num_classes=2, pretrained=False,
                 final_activation='linear', aspp_out_channels=1024,
                 loss=F.cross_entropy, metric=None,
                 optim_names='adamw'):
        super().__init__()
        
        # Encoder
        if pretrained: self.encoder = EfficientNet.from_pretrained(encoder_name, num_classes=1)
        else: self.encoder = EfficientNet.from_name(encoder_name, num_classes=1)
        
        self._encoder_block_index = []
        for i, block in enumerate(self.encoder._blocks):
            if isinstance(block._block_args.stride, list):
                self._encoder_block_index.append(i)
                
        # Aspp
        params_dict = {
            # Coefficients: (top_conv_filters, top_conv_res)
            'efficientnet-b2': (1408, (9,9))
        }
        aspp_params = params_dict[encoder_name]         
        self.aspp = ASPP(aspp_params[0], aspp_out_channels, aspp_params[1])

        # Decoder
        self.decoder = Decoder(num_classes, encoder_name,
                               final_activation, aspp_out_channels)

        # Loss & Metric
        self.loss = loss
        self.metric = metric

        # Optimizer parameters
        # Default parameters of optimizers
        self.optim_names = optim_names
        self.optim_params = {
            'adamw': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08,
                      'weight_decay': 0.01, 'amsgrad': False},
            'sgd': {'lr': 0.01, 'momentum': 0, 'dampening': 0,
                    'weight_decay': 0, 'nesterov': False}
        }
        
    def forward(self, inputs):
        # Encoder
        x0 = self.encoder._swish(self.encoder._bn0(self.encoder._conv_stem(inputs)))
        x1 = self.encoder_block(x0, 0)
        x2 = self.encoder_block(x1, 1)
        x3 = self.encoder_block(x2, 2)
        x4 = self.encoder_block(x3, 3)
        x5 = self.encoder_block(x4, 4)
        x6 = self.encoder_block(x5, 5)
        x7 = self.encoder_block(x6, 6)
        x8 = self.encoder._swish(self.encoder._bn1(self.encoder._conv_head(x7)))
        
        # Aspp
        x8 = self.aspp(x8)
        
        # Decoder
        out = self.decoder(x8, x7, x6, x5, x4, x3, x2, x1, x0)
        
        return out
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        if self.metric == 'acc':
            acc = accuracy(torch.argmax(y_hat, dim=1), y)
            self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
        return loss
    
    def validation_step(self,batch,batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = self.loss(y_hat,y)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        
        if self.metric == 'acc':
            acc = accuracy(torch.argmax(y_hat, dim=1), y)
            self.log("val_acc",acc,prog_bar=True,logger=True),
        
    def encoder_block(self, x, stage):
        start_idx = self._encoder_block_index[stage]
        if stage == len(self._encoder_block_index) - 1:
            stop_idx = -1
        else: stop_idx = self._encoder_block_index[stage + 1]
        
        for idx, block in enumerate(self.encoder._blocks[start_idx:stop_idx]):
            drop_connect_rate = self.encoder._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float((start_idx + idx)) / len(self.encoder._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        return x
        
    def configure_optimizers(self):
        optimizer = self.construct_optimizer(self.parameters())
        return optimizer
    
    def construct_optimizer(self, params):
        # Get the parameters of an optimizer
        optim_params = self.optim_params[self.optim_names]
        
        if self.optim_names.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=optim_params['lr'],
                                  momentum=optim_params['momentum'], 
                                  dampening=optim_params['dampening'],
                                  weight_decay=optim_params['weight_decay'],
                                  nesterov=optim_params['nesterov'])
        elif self.optim_names.lower() == 'adamw':
            optimizer = optim.AdamW(params, lr=optim_params['lr'],
                                    betas=optim_params['betas'],
                                    eps=optim_params['eps'],
                                    weight_decay=optim_params['weight_decay'],
                                    amsgrad=optim_params['amsgrad'])
        
        return optimizer
    
    def configure_optimizers_parameters(self, params):
        for optim_name, optim_params in params.items():
            for optim_param_name, value in optim_params.items():
                    self.optim_params[optim_name][optim_param_name] = value
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
            
    def freeze_aspp(self):
        for param in self.aspp.parameters():
            param.requires_grad = False
            
    def unfreeze_aspp(self):
        for param in self.aspp.parameters():
            param.requires_grad = True
            
    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False
            
    def unfreeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = True