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
from src.utils import seed_everything, CONFIG

import warnings
warnings.filterwarnings('ignore')


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
    checkpoint1 = f'lightning_logs/kfold/effi7-K{kfold_num}-sche-bat16-{mode}.ckpt'
    infer_model1 = EfficientNet_b7Model.load_from_checkpoint(checkpoint1, num_classes=CONFIG.N_CLASSES, mode=mode).to(device)
    checkpoint2 = f'lightning_logs/kfold/res50-K{kfold_num}-sche-bat16-{mode}.ckpt'
    # infer_model2 = EfficientNet_b6Model.load_from_checkpoint(checkpoint2, num_classes=CONFIG.N_CLASSES, mode=mode).to(device)
    infer_model2 = ResNet50Model.load_from_checkpoint(checkpoint2, num_classes=CONFIG.N_CLASSES).to(device)

    return np.array(model_inference(test_loader, infer_model1, device)), np.array(model_inference(test_loader, infer_model2, device))

def kfold_infer(kfold_num):
    real_preds_list = []
    fake_preds_list = []

    for i in range(kfold_num):
        print("current fold:", i)

        real_preds1, real_preds2 = inference(device='cuda', mode='real', kfold_num=i)
        time.sleep(1)
        fake_preds1, fake_preds2 = inference(device='cuda', mode='fake', kfold_num=i)
        time.sleep(1)
        real_preds_list.append((real_preds1 + real_preds2) / 2)
        fake_preds_list.append((fake_preds1 + fake_preds2) / 2)

    return np.mean(real_preds_list, axis=0), np.mean(fake_preds_list, axis=0)

print("seed initialize to", CONFIG.SEED)
seed_everything(CONFIG.SEED)

real_preds, fake_preds = kfold_infer(kfold_num=1)

csv_name = 'submitfile/effi7_res50_sche_K1_bat16.csv'  # need for modifing
submit = pd.read_csv('submitfile/sample_submission.csv')
submit.iloc[:, 1] = fake_preds[:, 1]
submit.iloc[:, 2] = real_preds[:, 1]
submit.to_csv(csv_name, index=False)
print("make csv in", csv_name)
