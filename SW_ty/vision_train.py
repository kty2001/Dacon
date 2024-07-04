from tqdm import tqdm

import numpy as np
import pandas as pd
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torchvision.models as models
import torchmetrics

from src.dataset import VoiceDataset
from src.model import EfficientNetB7Classifier
from src.utils import seed_everything, split_data, multiLabel_AUC


import warnings
warnings.filterwarnings('ignore')

class Config:
    SR = 32000
    N_MFCC = 13
    # Dataset
    ROOT_FOLDER = './'
    # Training
    N_CLASSES = 2
    BATCH_SIZE = 16
    N_EPOCHS = 5
    LR = 3e-4
    # Others
    SEED = 42
    
CONFIG = Config()

seed_everything(CONFIG.SEED) # Seed 고정
split_data("data/train.csv", CONFIG.SEED)

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

def train(device):
    train_image_path = 'data/train_mfcc'
    train_csv_path = 'data/train_answer.csv'
    val_image_path = 'data/val_mfcc'
    val_csv_path = 'data/val_answer.csv'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])

    train_dataset = VoiceDataset(image_path=train_image_path, csv_path=train_csv_path, transform=transform)
    val_dataset = VoiceDataset(image_path=val_image_path, csv_path=val_csv_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

    model = EfficientNetB7Classifier(num_classes=2).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = CONFIG.LR)
    
    best_val_score = 0
    best_model = None
    
    for epoch in range(1, CONFIG.N_EPOCHS+1):
        train_loss = train_one_epoch(train_loader, device, model, criterion, optimizer)
        val_loss, val_score = valid_one_epoch(val_loader, device, model, criterion)
        print(f'Epoch [{epoch}], Train Loss : [{train_loss:.5f}] Val Loss : [{val_loss:.5f}] Val AUC : [{val_score:.5f}]')
            
        # if best_val_score < val_score:
        #     best_val_score = val_score
        #     best_model = model
    
    torch.save(model.state_dict(), 'visionver.pth')
    print('save the best model to visionver.pth')

    return best_model

infer_model = train(device='cuda')
print(type(infer_model))

# test = pd.read_csv('./data/test.csv')
# test_mfcc = get_mfcc_feature(test, False)
# test_dataset = CustomDataset(test_mfcc, None)
# test_loader = DataLoader(
#     test_dataset,
#     batch_size=CONFIG.BATCH_SIZE,
#     shuffle=False
# )

# def inference(model, test_loader, device):
#     model.to(device)
#     model.eval()
#     predictions = []
#     with torch.no_grad():
#         for features in tqdm(iter(test_loader)):
#             features = features.float().to(device)
            
#             probs = model(features)

#             probs  = probs.cpu().detach().numpy()
#             predictions += probs.tolist()
#     return predictions

# preds = inference(infer_model, test_loader, device)

# submit = pd.read_csv('./sample_submission.csv')
# submit.iloc[:, 1:] = preds
# submit.head()
# submit.to_csv('./baseline_submit.csv', index=False)