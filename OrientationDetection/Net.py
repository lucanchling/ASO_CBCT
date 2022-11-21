import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from monai.networks.nets.densenet import DenseNet121, DenseNet169, DenseNet201
import numpy as np
from icecream import ic
import pytorch_lightning as pl
import torchmetrics

# Different Network

class DenseNet(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.net = DenseNet169(spatial_dims=3, in_channels=1, out_channels=3)
        # self.net = EfficientNetBN('efficientnet-b2', spatial_dims=3, in_channels=1,num_classes=3, pretrained=False)
        self.CosSimLoss = nn.CosineSimilarity()
        self.Accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        return nn.functional.normalize(self.net(x),dim=1)

    def training_step(self, batch, batch_idx):
        scan, directionVector, scan_path = batch
        batch_size = scan.shape[0]

        directionVector_hat = self(scan)
        
        loss = (1 - self.CosSimLoss(directionVector_hat, directionVector))
        # Sum the loss over the batch
        loss = loss.sum()
        self.log('train_loss', loss, batch_size=batch_size)
                   
        return loss

    def validation_step(self, batch, batch_idx):
        scan, directionVector, scan_path = batch
        batch_size = scan.shape[0]
        directionVector_hat = self(scan)
        # ic(directionVector_hat)
        loss = (1 - self.CosSimLoss(directionVector_hat, directionVector))
        loss = loss.sum()
        self.log('val_loss', loss, batch_size=batch_size)

        return loss

    def test_step(self, batch, batch_idx):
        scan, directionVector, scan_path = batch
        batch_size = scan.shape[0]

        directionVector_hat = self(scan)
        
        loss = (1 - self.CosSimLoss(directionVector_hat, directionVector))
        loss = loss.sum()
        self.log('val_loss', loss, batch_size=batch_size)
        
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)