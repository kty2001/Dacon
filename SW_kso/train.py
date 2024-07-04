# train.py

import torch
from torch import nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import CustomDatasetMFCC, CustomDatasetMel, get_mfcc_feature, get_mel_spectrogram
from src.model import VGG19, BiLSTM
from src.utils import multiLabel_AUC
from torchmetrics import AUROC

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Config:
    SR = 32000
    N_MFCC = 13
    N_CLASSES = 2
    BATCH_SIZE = 128
    N_EPOCHS = 20
    LR = 3e-4
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

df = pd.read_csv('./data/train.csv')
train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CONFIG.SEED)

train_mfcc, train_labels = get_mfcc_feature(train, True)
val_mfcc, val_labels = get_mfcc_feature(val, True)

train_mel_spec, train_labels = get_mel_spectrogram(train, True)
val_mel_spec, val_labels = get_mel_spectrogram(val, True)

train_dataset_mfcc = CustomDatasetMFCC(train_mfcc, train_labels)
val_dataset_mfcc = CustomDatasetMFCC(val_mfcc, val_labels)

train_dataset_mel = CustomDatasetMel(train_mel_spec, train_labels)
val_dataset_mel = CustomDatasetMel(val_mel_spec, val_labels)

train_loader_mfcc = DataLoader(
    train_dataset_mfcc,
    batch_size=CONFIG.BATCH_SIZE,
    shuffle=True
)

val_loader_mfcc = DataLoader(
    val_dataset_mfcc,
    batch_size=CONFIG.BATCH_SIZE,
    shuffle=False
)

train_loader_mel = DataLoader(
    train_dataset_mel,
    batch_size=CONFIG.BATCH_SIZE,
    shuffle=True
)

val_loader_mel = DataLoader(
    val_dataset_mel,
    batch_size=CONFIG.BATCH_SIZE,
    shuffle=False
)

def train(model, optimizer, train_loader, val_loader, device):
    model.to(device)
    criterion = nn.BCELoss().to(device)
    best_val_score = 0
    best_model = None
    
    for epoch in range(1, CONFIG.N_EPOCHS + 1):
        model.train()
        train_loss = []
        for features, labels in tqdm(iter(train_loader)):
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
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
    val_loss = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in tqdm(iter(val_loader)):
            features = features.to(device)
            labels = labels.to(device)
            probs = model(features)
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

# Training CNN model with Mel spectrograms
model_mel = VGG19(num_classes=CONFIG.N_CLASSES)
optimizer_mel = optim.Adam(params=model_mel.parameters(), lr=CONFIG.LR)
trained_model_mel = train(model_mel, optimizer_mel, train_loader_mel, val_loader_mel, device)

# Training BiLSTM model with MFCCs
model_mfcc = BiLSTM(input_dim=CONFIG.N_MFCC, hidden_dim=64, output_dim=CONFIG.N_CLASSES)
optimizer_mfcc = optim.Adam(params=model_mfcc.parameters(), lr=CONFIG.LR)
trained_model_mfcc = train(model_mfcc, optimizer_mfcc, train_loader_mfcc, val_loader_mfcc, device)


