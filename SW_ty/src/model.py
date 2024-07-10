import os
import shutil
import torch
from torch import nn
import torchvision.models as models
import lightning as L

class DeldirCallback(L.Callback):
    def __init__(self, del_dir):
        self.del_dir = del_dir

    def on_fit_end(self, trainer, pl_module):
        print(f"fit end")
        # if os.path.exists(self.del_dir):
        #     shutil.rmtree(self.del_dir)
        # os.makedirs(self.del_dir)

class EfficientNetModel(L.LightningModule):
    def __init__(self, num_classes=2):
        super().__init__()
        self.efficientnet_b7 = models.efficientnet_b7(pretrained=True)
        self.efficientnet_b7.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(2560, num_classes))
        self.sigmoid = nn.Sigmoid()
        
        self.loss_fn = nn.BCELoss()

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

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


class EfficientNetB7Classifier(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5, lastconv_output_channels=2560):
        super(EfficientNetB7Classifier, self).__init__()
        self.efficientnet_b7 = models.efficientnet_b7(pretrained=True)
        self.efficientnet_b7.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.efficientnet_b7(x)
        x = self.sigmoid(x)
        return x
    
class ResNet50Classication(nn.Module):
    def __init__(self, num_class=1000):
        super(ResNet50Classication, self).__init__()
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_class)

    def forward(self, x):
        return self.model(x)