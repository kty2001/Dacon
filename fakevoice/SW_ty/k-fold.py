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
from src.model import EfficientNet_b7Model, ResNet50Model, ResNet152Model
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
    
    model1 = EfficientNet_b7Model(num_classes=CONFIG.N_CLASSES)
    # model2 = ResNet50Model(num_classes=CONFIG.N_CLASSES)
    model2 = ResNet152Model(num_classes=CONFIG.N_CLASSES)
    
    trainer1 = L.Trainer(max_epochs=5, accelerator='gpu', limit_train_batches=32)
    trainer2 = L.Trainer(max_epochs=5, accelerator='gpu', limit_train_batches=32)
    
    time.sleep(1)
    trainer1.fit(model1, train_loader, val_loader)
    time.sleep(1)
    trainer1.save_checkpoint(filepath=f'lightning_logs/kfold/effi-K{kfold_num}-argu50-bat32-{mode}.ckpt')
    time.sleep(1)
    trainer2.fit(model2, train_loader, val_loader)
    time.sleep(1)
    # trainer2.save_checkpoint(filepath=f'lightning_logs/kfold/res-K{kfold_num}-argu50-bat32-{mode}.ckpt')
    trainer2.save_checkpoint(filepath=f'lightning_logs/kfold/res152-K{kfold_num}-argu50-bat32-{mode}.ckpt')
    time.sleep(1)

def model_inference(test_loader, model, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in tqdm(iter(test_loader)):
            images = images.to(device)
            
            probs = model(images)

            probs = probs.cpu().detach().numpy()
            predictions += probs.tolist()
    return predictions

def inference(device, mode, kfold_num):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(CONFIG.IMAGE_SIZE),
    ])

    test_dataset = VoiceDataset(image_path='data/test_mel', csv_path='data/test.csv', argu_data=None, transform=transform, mode=mode) # need for modifing
    test_loader = DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE*16, shuffle=False)

    # load checkpoint
    checkpoint1 = f'lightning_logs/kfold/effi-K{kfold_num}-argu50-bat32-{mode}.ckpt'
    infer_model1 = EfficientNet_b7Model.load_from_checkpoint(checkpoint1, num_classes=CONFIG.N_CLASSES).to(device)
    # checkpoint2 = f'lightning_logs/kfold/res50-K{kfold_num}-argu50-bat32-{mode}.ckpt'
    checkpoint2 = f'lightning_logs/kfold/res152-K{kfold_num}-argu50-bat32-{mode}.ckpt'
    # infer_model2 = ResNet50Model.load_from_checkpoint(checkpoint2, num_classes=CONFIG.N_CLASSES).to(device)
    infer_model2 = ResNet152Model.load_from_checkpoint(checkpoint2, num_classes=CONFIG.N_CLASSES).to(device)

    return np.array(model_inference(test_loader, infer_model1, device)), np.array(model_inference(test_loader, infer_model2, device))


print("seed initialize to", CONFIG.SEED)
seed_everything(CONFIG.SEED)
print("data split")
split_data("data/train.csv", CONFIG.SEED)

def kfold(kfold_num):
    real_preds_list = []
    fake_preds_list = []

    for i in range(kfold_num):
        print("current fold:", i)

        train(mode='real', kfold_num=i)
        time.sleep(1)
        train(mode='fake', kfold_num=i)
        time.sleep(1)

        real_preds1, real_preds2 = inference(device='cuda', mode='real', kfold_num=i)
        time.sleep(1)
        fake_preds1, fake_preds2 = inference(device='cuda', mode='fake', kfold_num=i)
        time.sleep(1)
        real_preds_list.append((real_preds1 + real_preds2) / 2)
        fake_preds_list.append((fake_preds1 + fake_preds2) / 2)

    return np.mean(real_preds_list, axis=0), np.mean(fake_preds_list, axis=0)

real_preds, fake_preds = kfold(kfold_num=5)

csv_name = 'submitfile/ese_K5_bat32_argu50.csv'  # need for modifing
submit = pd.read_csv('submitfile/sample_submission.csv')
submit.iloc[:, 1] = fake_preds[:, 1]
submit.iloc[:, 2] = real_preds[:, 1]
submit.to_csv(csv_name, index=False)
print("make csv in", csv_name)
