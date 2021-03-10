# This code is inspired from
# https://medium.com/analytics-vidhya/how-to-add-additional-layers-in-a-pre-trained-model-using-pytorch-5627002c75a5
# https://www.kaggle.com/shivanandmn/efficientnet-pytorch-lightning-train-inference

import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from .efficientnet import EfficientNet

class EffNetTL(pl.LightningModule):
    def __init__(self, backbone_name='efficientnet-b0', num_classes=10, pretrained=False,
                 head=None, optim_names={'head': 'sgd', 'backbone': 'sgd'}):
        super().__init__()
        
        if pretrained: self.backbone = EfficientNet.from_pretrained(backbone_name, num_classes=num_classes)
        else: self.backbone = EfficientNet.from_name(backbone_name, num_classes=num_classes)
        
        self.head = head if head else nn.Linear(self.backbone._fc.in_features, num_classes)
        
        # Optimizer parameters
        self.optim_names = optim_names
        self.optim_params = {
            'head': {
                'adamw': {
                    'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08,
                    'weight_decay': 0.01, 'amsgrad': False
                    },
                'sgd': {
                    'lr': 0.01, 'momentum': 0, 'dampening': 0,
                    'weight_decay': 0, 'nesterov': False
                    }
                },
            'backbone': {
                'adamw': {
                    'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08,
                    'weight_decay': 0.01, 'amsgrad': False
                    },
                'sgd': {
                    'lr': 0.01, 'momentum': 0, 'dampening': 0,
                    'weight_decay': 0, 'nesterov': False
                    }
                }
        }
            
    def forward(self, inputs):
        # EfficientNet feature extractor
        x = self.backbone.extract_features(inputs)       
        
        # Pooling, flatten, and dropout layer
        x = self.backbone._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.backbone._dropout(x)
        
        # Head block
        x = self.head(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(torch.argmax(y_hat, dim=1), y)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True),
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat,y)
        acc = accuracy(torch.argmax(y_hat, dim=1), y)
        self.log("val_acc",acc,prog_bar=True,logger=True),
        self.log("val_loss",loss,prog_bar=True,logger=True)
            
    def configure_optimizers(self):
        head_optim = self.construct_optimizer('head', self.head.parameters())
        backbone_optim = self.construct_optimizer('backbone', self.backbone.parameters())
        return head_optim, backbone_optim
            
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
            
    def construct_optimizer(self, name, params):
        # Get the parameters of an optimizer
        optim_params = self.optim_params[name][self.optim_names[name]]
        
        if self.optim_names[name].lower() == 'sgd':
            optimizer = optim.SGD(params, lr=optim_params['lr'],
                                  momentum=optim_params['momentum'], 
                                  dampening=optim_params['dampening'],
                                  weight_decay=optim_params['weight_decay'],
                                  nesterov=optim_params['nesterov'])
        elif self.optim_names[name].lower() == 'adamw':
            optimizer = optim.AdamW(params, lr=optim_params['lr'],
                                    betas=optim_params['betas'],
                                    eps=optim_params['eps'],
                                    weight_decay=optim_params['weight_decay'],
                                    amsgrad=optim_params['amsgrad'])
        
        return optimizer
    
    def configure_optimizers_parameters(self, params):
        
        for param_name, optimizers in params.items():
            for optim_name, optim_params in optimizers.items():
                for optim_param_name, value in optim_params.items():
                     self.optim_params[param_name][optim_name][optim_param_name] = value