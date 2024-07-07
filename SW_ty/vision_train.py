import os
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
# from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from src.dataset import VoiceDataset
from src.model import EfficientNetB7Classifier, ResNet50Classication
from src.utils import seed_everything, split_data, multiLabel_AUC, CONFIG

import warnings
warnings.filterwarnings('ignore')
    

if os.path.exists('data/argument'):
    shutil.rmtree('data/argument')
os.makedirs('data/argument')

def train_one_epoch(train_loader, device, model, criterion, optimizer):    
    model.train()
    train_loss = []

    for images, labels in tqdm(iter(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        
        output = model(images)
        loss = criterion(output, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        train_loss.append(loss.item())

    train_loss = np.mean(train_loss)
    
    return train_loss

def valid_one_epoch(val_loader, device, model, criterion):
    model.eval()
    val_loss, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for images, labels in tqdm(iter(val_loader)):
            images = images.to(device)
            labels = labels.to(device)
            
            probs = model(images)
            
            loss = criterion(probs, labels)

            val_loss.append(loss.item())

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        val_loss = np.mean(val_loss)

        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        
        # Calculate AUC score
        auc_score = multiLabel_AUC(all_labels, all_probs)
    
    return val_loss, auc_score

def train(device, mode):
    # need for modifing
    train_image_path = 'data/train_melspec'
    train_csv_path = 'data/train_answer.csv'
    val_image_path = 'data/val_melspec'
    val_csv_path = 'data/val_answer.csv'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(CONFIG.IMAGE_SIZE),
    ])

    train_dataset = VoiceDataset(image_path=train_image_path, csv_path=train_csv_path, transform=transform, mode=mode)
    val_dataset = VoiceDataset(image_path=val_image_path, csv_path=val_csv_path, transform=transform, mode=mode)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

    model = EfficientNetB7Classifier(num_classes=CONFIG.N_CLASSES).to(device)
    # model.load_state_dict(torch.load(f'weights/effi_5epoch_melspec_{mode}.pth')) # need for modifing
    # model = ResNet50Classication(num_classes=CONFIG.N_CLASSES).to(device)
    # model.load_state_dict(torch.load(f'weights/effi_5epoch_melspec_{mode}.pth')) # need for modifing
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(params = model.parameters(), lr = CONFIG.LR)  # AdamW, Lion-pytorch
    # scheduler = ReduceLROnPlateau(optimizer, mode='min, patience=2) 
    # scheduler = CosineAnnealingLR(optimizer, T_max=5)

    best_epoch = 0
    best_val_score = 0
    best_model = None
    
    for epoch in range(1, CONFIG.N_EPOCHS+1):
        train_loss = train_one_epoch(train_loader, device, model, criterion, optimizer)
        val_loss, val_score = valid_one_epoch(val_loader, device, model, criterion)
        print(f'Epoch [{epoch}], Train Loss : [{train_loss:.6f}] Val Loss : [{val_loss:.6f}] Val AUC : [{val_score:.6f}]')
            
        if best_val_score < val_score:
            best_val_score = val_score
            best_model = model
            best_epoch = epoch

    torch.save(best_model.state_dict(), f'weights/effi_5epoch_argu_{mode}.pth') # need for modify
    print(f"The best model is {best_epoch} epoch model")
    print(f'save the best model to effi_5epoch_argu_{mode}.pth')

    return best_model

seed_everything(CONFIG.SEED) # Seed 고정
split_data("data/train.csv", CONFIG.SEED)

real_model = train(device='cuda', mode='real')
fake_model = train(device='cuda', mode='fake')