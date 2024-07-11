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

from src.dataset import VoiceDataset
from src.model import EfficientNet_b7Model, ResNet50Model
from src.utils import seed_everything, split_data, CONFIG

import warnings
warnings.filterwarnings('ignore')


def train(mode):
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
    model2 = ResNet50Model(num_classes=CONFIG.N_CLASSES)
    
    trainer = L.Trainer(max_epochs=5, accelerator='gpu', limit_train_batches=32)

    trainer.fit(model1, train_loader, val_loader)
    trainer.fit(model2, train_loader, val_loader)

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

def inference(device, mode):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(CONFIG.IMAGE_SIZE),
    ])

    test_dataset = VoiceDataset(image_path='data/test_mel', csv_path='data/test.csv', argu_data=None, transform=transform, mode=mode) # need for modifing
    test_loader = DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE*16, shuffle=False)

    # load checkpoint
    checkpoint1 = f"./lightning_logs/effi-argu50-bat32-{mode}/checkpoints/epoch=4-step=160.ckpt"
    infer_model1 = EfficientNet_b7Model.load_from_checkpoint(checkpoint1, num_classes=CONFIG.N_CLASSES)
    checkpoint2 = f"./lightning_logs/res-argu50-bat32-{mode}/checkpoints/epoch=4-step=160.ckpt"
    infer_model2 = ResNet50Model.load_from_checkpoint(checkpoint2, num_classes=CONFIG.N_CLASSES)

    # return np.array(model_inference(test_loader, infer_model1, device))
    return np.array(model_inference(test_loader, infer_model1, device)), np.array(model_inference(test_loader, infer_model2, device))


print("seed initialize to", CONFIG.SEED)
seed_everything(CONFIG.SEED)
print("data split")
split_data("data/train.csv", CONFIG.SEED)

real_preds_list = []
fake_preds_list = []

def kfold(kfold_num):
    for i in range(kfold_num):
        train(mode='real')
        train(mode='fake')

        real_preds1, real_preds2 = inference(device='cuda', mode='real')
        fake_preds1, fake_preds2 = inference(device='cuda', mode='fake')
        real_preds_list.append((real_preds1 + real_preds2) / 2)
        fake_preds_list.append((fake_preds1 + fake_preds2) / 2)

    return np.mean(real_preds_list, axis=0), np.mean(fake_preds_list, axis=0)

real_preds, fake_preds = kfold(5)

csv_name = 'submitfile/K5_ese_bat32_argu50.csv'  # need for modifing
submit = pd.read_csv('submitfile/sample_submission.csv')
submit.iloc[:, 1] = fake_preds[:, 1]
submit.iloc[:, 2] = real_preds[:, 1]
submit.to_csv(csv_name, index=False)
