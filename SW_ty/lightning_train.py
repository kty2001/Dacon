import os
import shutil
import time
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import lightning as L

from src.dataset import VoiceDataset
from src.model import EfficientNet_b7Model, ResNet50Model, EfficientNet_b6Model
from src.utils import seed_everything, split_data, CONFIG

import warnings
warnings.filterwarnings('ignore')


def train(mode, kfold_num):
    train_image_path = 'data/train_mel'
    train_csv_path = 'data/train_answer.csv'
    val_image_path = 'data/val_mel'
    val_csv_path = 'data/val_answer.csv'
    with open('data/features/unlabeled_data_features.pkl', 'rb') as f:
        argu_data = pickle.load(f)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(CONFIG.IMAGE_SIZE),
    ])

    train_dataset = VoiceDataset(image_path=train_image_path, csv_path=train_csv_path, argu_data=argu_data, transform=transform, mode=mode)
    val_dataset = VoiceDataset(image_path=val_image_path, csv_path=val_csv_path, argu_data=argu_data, transform=transform, mode=mode)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)
    
    model1 = EfficientNet_b7Model(num_classes=CONFIG.N_CLASSES, mode=mode)
    # model2 = EfficientNet_b6Model(num_classes=CONFIG.N_CLASSES, mode=mode)
    model2 = ResNet50Model(num_classes=CONFIG.N_CLASSES)
    
    trainer1 = L.Trainer(max_epochs=5, accelerator='gpu', limit_train_batches=16)
    trainer2 = L.Trainer(max_epochs=5, accelerator='gpu', limit_train_batches=16)
    
    trainer1.fit(model1, train_loader, val_loader)
    trainer1.save_checkpoint(filepath=f'lightning_logs/kfold/effi7-K{kfold_num}-sche-bat16-{mode}.ckpt')
    time.sleep(1)

    trainer2.fit(model2, train_loader, val_loader)
    trainer2.save_checkpoint(filepath=f'lightning_logs/kfold/res50-K{kfold_num}-sche-bat16-{mode}.ckpt')
    time.sleep(1)

print("seed initialize to", CONFIG.SEED)
seed_everything(CONFIG.SEED)
print("data split")
split_data("data/train.csv", CONFIG.SEED)

def kfold_train(kfold_num):
    for i in range(kfold_num):
        print("current fold:", i)

        train(mode='real', kfold_num=i)
        train(mode='fake', kfold_num=i)
        time.sleep(1)

kfold_train(kfold_num=1)