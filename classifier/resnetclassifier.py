
import sys
import warnings
from pathlib import Path
from argparse import ArgumentParser

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# torch and lightning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics import Accuracy

class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes, resnet_version,
                 optimizer='adam', lr=1e-6, batch_size=16,
                 transfer=False):
        super().__init__()

        self.__dict__.update(locals())
        resnets = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }
        self.num_classes = num_classes
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        # instantiate loss criterion
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        # Using a pretrained ResNet backbone
        self.resnet_model = resnets[resnet_version](pretrained=transfer)
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, num_classes)
        self.acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self._encoding = None  # Buffer to store encoding


        target_layer = list(self.resnet_model.children())[-2]  # Penultimate layer

        def hook_fn(module, input, output):
            # Flatten output and store it
            self._encoding = output.flatten(1)

        target_layer.register_forward_hook(hook_fn)
        self.latent_dim = self.get_encoding_size()


    def forward(self, X):
        return self.resnet_model(X)

    def get_encoding_size(self):
        dummy = torch.zeros((1,3,512,512))
        return self.get_encoding(dummy).flatten(1).shape[-1]

    def get_encoding(self, X):
        _ = self(X)  # This triggers forward and stores encoding via hook
        return self._encoding

    def compute_loss(self, x, y):
        return self.criterion(self(x), y)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,100, 2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        preds = self(x)
        loss = self.criterion(preds, y).mean()

        acc = self.acc(preds, y)
        # perform logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        preds = self(x)
        loss = self.criterion(preds,y).mean()
        acc = self.acc(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        preds = self(x)
        loss = self.criterion(preds, y).mean()
        acc = self.acc(preds, y)

        # perform logging
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)
