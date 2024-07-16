import os
import shutil
import torch
from torch import nn
import torchvision.models as models
import lightning as L
import numpy as np

from src.utils import auc_brier_ece

class EfficientNet_b7Model(L.LightningModule):
    def __init__(self, num_classes=2, mode='real'):
        super().__init__()
        self.efficientnet_b7 = models.efficientnet_b7(pretrained=True)
        self.efficientnet_b7.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(2560, num_classes))
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()
        self.val_loss = 1
        self.auc, self.brier, self.ece = [], [], []
        self.mode = mode

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
        if self.mode == 'real':
            y_true = y.cpu().detach().numpy()[:, 1]
            y_prob = torch.sigmoid(y_hat).cpu().detach().numpy()[:, 1]
        elif self.mode == 'fake':
            y_true = y.cpu().detach().numpy()[:, 0]
            y_prob = torch.sigmoid(y_hat).cpu().detach().numpy()[:, 0]
        auc_scores, brier_scores, ece_scores = auc_brier_ece(y_true, y_prob)
        self.auc.extend(auc_scores)
        self.brier.extend(brier_scores)
        self.ece.extend(ece_scores)

    def on_validation_epoch_end(self):
        print(self.val_loss)
        mean_auc, mean_brier, mean_ece = np.mean(self.auc), np.mean(self.brier), np.mean(self.ece)
        print(f"auc: {mean_auc:.5f} / brier: {mean_brier:.5f} / ece: {mean_ece:.5f}\nCombi_Score: {0.5 * (1 - mean_auc) + 0.25 * mean_brier + 0.25 * mean_ece:.5f}")
        self.auc, self.brier, self.ece = [], [], []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

class EfficientNet_b6Model(L.LightningModule):
    def __init__(self, num_classes=2, mode='real'):
        super().__init__()
        self.efficientnet_b6 = models.efficientnet_b6(pretrained=True)
        self.efficientnet_b6.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(2304, num_classes))
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()
        self.val_loss = 1
        self.auc, self.brier, self.ece = [], [], []
        self.mode = mode

    def forward(self, x):
        x = self.efficientnet_b6(x)
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
        if self.mode == 'real':
            y_true = y.cpu().detach().numpy()[:, 1]
            y_prob = torch.sigmoid(y_hat).cpu().detach().numpy()[:, 1]
        elif self.mode == 'fake':
            y_true = y.cpu().detach().numpy()[:, 0]
            y_prob = torch.sigmoid(y_hat).cpu().detach().numpy()[:, 0]
        auc_scores, brier_scores, ece_scores = auc_brier_ece(y_true, y_prob)
        self.auc.extend(auc_scores)
        self.brier.extend(brier_scores)
        self.ece.extend(ece_scores)

    def on_validation_epoch_end(self):
        print(self.val_loss)
        mean_auc, mean_brier, mean_ece = np.mean(self.auc), np.mean(self.brier), np.mean(self.ece)
        print(f"auc: {mean_auc:.5f} / brier: {mean_brier:.5f} / ece: {mean_ece:.5f}\nCombi_Score: {0.5 * (1 - mean_auc) + 0.25 * mean_brier + 0.25 * mean_ece:.5f}")
        self.auc, self.brier, self.ece = [], [], []

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
        self.auc, self.brier, self.ece = [], [], []

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
        auc_scores, brier_scores, ece_scores = auc_brier_ece(y, y_hat)
        self.auc.extend(auc_scores)
        self.brier.extend(brier_scores)
        self.ece.extend(ece_scores)

    def on_validation_epoch_end(self):
        print(self.val_loss)
        print(f"auc: {np.mean(self.auc)} / brier: {np.mean(self.brier)} / ece: {np.mean(self.ece)}")
        self.auc, self.brier, self.ece = [], [], []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)