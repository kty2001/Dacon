import os
import shutil
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import lightning as L
# from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from src.dataset import VoiceDataset
from src.model import EfficientNetB7Classifier, ResNet50Classication, EfficientNetModel, DeldirCallback
from src.utils import seed_everything, split_data, multiLabel_AUC, CONFIG

import warnings
warnings.filterwarnings('ignore')


def train(mode):
    if os.path.exists('data/argument'):
        shutil.rmtree('data/argument')
    os.makedirs('data/argument')
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

    model = EfficientNetModel(num_classes=CONFIG.N_CLASSES)
    trainer = L.Trainer(max_epochs=5, accelerator='gpu', callbacks=[DeldirCallback('./data/argument')])
    trainer.fit(model, train_loader, val_loader)

seed_everything(CONFIG.SEED) # Seed 고정
split_data("data/train.csv", CONFIG.SEED)

# if os.path.exists('data/argument'):
#     shutil.rmtree('data/argument')
# os.makedirs('data/argument')

train(mode='real')
# train(mode='fake')
