import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

class SegmentationModel(pl.LightningModule):
    def __init__(self, encoder_name='efficientnet-b2', encoder_weights='imagenet',
                 activation='sigmoid', in_channels=3, classes=1,
                 loss=smp.utils.losses.DiceLoss(eps=1e-6),
                 metrics={'IoU': smp.utils.metrics.IoU(),
                          'dice_score': smp.utils.metrics.Fscore()}):
        super().__init__()
        self.model = smp.FPN(encoder_name=encoder_name,
                             encoder_weights=encoder_weights,
                             activation=activation, classes=classes
                             in_channels=in_channels)
        self.loss = loss
        self.metrics = metrics
        
    def forward(self, x):
        return self.model.forward(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4,
                                      weight_decay=2e-5, amsgrad=True)
        return optimizer
    
    def training_step(self, batch, batch_nb):
        img, mask = batch
        
        out = self.forward(img)        
        loss = self.loss(out, mask)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.metrics is not None:
            for name, func in self.metrics.items():
                self.log(f"train_{name}", func(out, mask), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_nb):
        img, mask = batch
        
        out = self.forward(img)        
        loss = self.loss(out, mask)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.metrics is not None:
            for name, func in self.metrics.items():
                self.log(f"val_{name}", func(out, mask), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
        
    def freeze_encoder(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        
    def unfreeze_encoder(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = True 
            
    def freeze_decoder(self):
        for param in self.model.decoder.parameters():
            param.requires_grad = False
    
    def unfreeze_decoder(self):
        for param in self.model.decoder.parameters():
            param.requires_grad = True