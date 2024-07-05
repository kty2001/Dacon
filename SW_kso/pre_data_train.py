# train.py

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from src.model import VGG19, BiLSTM
from src.dataset import CustomDatasetMel, CustomDatasetMFCC, Config
from src.utils import multiLabel_AUC
from torchvision import transforms


import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Config:
    SR = 32000
    N_MFCC = 13
    N_CLASSES = 2
    BATCH_SIZE = 128
    N_EPOCHS = 50
    LR = 0.0001
    SEED = 42

CONFIG = Config()

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CONFIG.SEED)


def train(model, optimizer, train_loader, val_loader, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    best_val_score = 0
    best_model = None
    
    for epoch in range(1, CONFIG.N_EPOCHS + 1):
        model.train()
        train_loss = []
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            features = features.to(device)
            labels = labels.to(device).squeeze()
            
            optimizer.zero_grad()
            
            output_logits = model(features)
            
            loss = criterion(output_logits, labels.squeeze())  # labels를 1차원 벡터로 변환하여 전달
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val AUC : [{_val_score:.5f}]')
            
        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model
    
    return best_model

def train_mel(model, optimizer, train_loader, val_loader, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    best_val_score = 0
    best_model = None
    
    for epoch in range(1, CONFIG.N_EPOCHS + 1):
        model.train()
        train_loss = []
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):            
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            output_logits = model(features)
            
            loss = criterion(output_logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val AUC : [{_val_score:.5f}]')
            
        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model
    
    return best_model


def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for features, labels in tqdm(iter(val_loader)):
            features = features.float().to(device)
            labels = labels.float().to(device)
            
            probs = model(features)
            
            loss = criterion(probs, labels)

            val_loss.append(loss.item())

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        _val_loss = np.mean(val_loss)

        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        
        # Calculate AUC score
        auc_score = multiLabel_AUC(all_labels, all_probs)
    
    return _val_loss, auc_score


if __name__ == '__main__':
    # # Load preprocessed data
    train_mfcc = np.load('./data/train_mfcc.npy', allow_pickle=True)
    val_mfcc = np.load('./data/val_mfcc.npy', allow_pickle=True)
    train_labels_mfcc = np.load('./data/train_labels_mfcc.npy', allow_pickle=True)
    val_labels_mfcc = np.load('./data/val_labels_mfcc.npy', allow_pickle=True)
    
    train_mel_spec = np.load('./data/train_mel_spec.npy', allow_pickle=True)
    val_mel_spec = np.load('./data/val_mel_spec.npy', allow_pickle=True)
    train_labels_mel = np.load('./data/train_labels_mel.npy', allow_pickle=True)
    val_labels_mel = np.load('./data/val_labels_mel.npy', allow_pickle=True)

    # # Create datasets and data loaders
    train_dataset_mfcc = CustomDatasetMFCC(train_mfcc, train_labels_mfcc)
    val_dataset_mfcc = CustomDatasetMFCC(val_mfcc, val_labels_mfcc)
    train_loader_mfcc = DataLoader(train_dataset_mfcc, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
    val_loader_mfcc = DataLoader(val_dataset_mfcc, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

    train_dataset_mel = CustomDatasetMel(train_mel_spec, train_labels_mel)
    val_dataset_mel = CustomDatasetMel(val_mel_spec, val_labels_mel)
    train_loader_mel = DataLoader(train_dataset_mel, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
    val_loader_mel = DataLoader(val_dataset_mel, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

    model_mfcc = BiLSTM(input_dim=CONFIG.N_MFCC, hidden_dim=64, output_dim=CONFIG.N_CLASSES)
    optimizer_mfcc = torch.optim.Adam(params=model_mfcc.parameters(), lr=CONFIG.LR)

    best_model_mfcc = train(model_mfcc, optimizer_mfcc, train_loader_mfcc, val_loader_mfcc, device)
    
    torch.save(best_model_mfcc.state_dict(), './data/checkpoints/best_model_mfcc01.pth')
    
    
    model_mel = VGG19(num_classes= CONFIG.N_CLASSES)
    optimizer_mel = torch.optim.Adam(params=model_mel.parameters(), lr=CONFIG.LR)

    best_model_mel = train_mel(model_mel, optimizer_mel, train_loader_mel, val_loader_mel, device)
    torch.save(best_model_mel.state_dict(), './data/checkpoints/best_model_mel01.pth')


