# This code is inspired from
# https://medium.com/analytics-vidhya/how-to-add-additional-layers-in-a-pre-trained-model-using-pytorch-5627002c75a5

import torch.nn as nn
from .efficientnet import EfficientNet


class EffNetTL(nn.Module):
    def __init__(self, backbone_name='efficientnet-b0', num_classes=10, pretrained=False, head=None):
        super().__init__()
        
        if pretrained: self.backbone = EfficientNet.from_pretrained(backbone_name, num_classes=num_classes)
        else: self.backbone = EfficientNet.from_name(backbone_name, num_classes=num_classes)
        
        self.head = head if head else nn.Linear(self.backbone._fc.in_features, num_classes)
            
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
            
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
