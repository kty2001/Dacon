import os
import shutil
import torch
from torch import nn
import torchvision.models as models
import lightning as L


class EfficientNet_b7Model(L.LightningModule):
    def __init__(self, num_classes=2):
        super().__init__()
        self.efficientnet_b7 = models.efficientnet_b7(pretrained=True)
        self.efficientnet_b7.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(2560, num_classes))
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()
        self.val_loss = 1

    def forward(self, x):
        x = self.efficientnet_b7(x)
        x = self.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
        self.val_loss = loss

    def on_validation_epoch_end(self):
        print(self.val_loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


class ResNet50Model(L.LightningModule):
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(num_ftrs, num_classes))
        self.softmax = nn.Softmax()
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        x = self.resnet50(x)
        x = self.softmax(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
        self.val_loss = loss

    def on_validation_epoch_end(self):
        print(self.val_loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
