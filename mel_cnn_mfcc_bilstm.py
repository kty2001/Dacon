import librosa

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random

from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torch
import torchmetrics
import os


import warnings
warnings.filterwarnings('ignore')


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Config:
    SR = 32000
    N_MFCC = 13
    # Dataset
    ROOT_FOLDER = './'
    # Training
    N_CLASSES = 2
    BATCH_SIZE = 128
    N_EPOCHS = 20
    LR = 3e-4
    # Others
    SEED = 42
    
CONFIG = Config()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CONFIG.SEED) # Seed 고정

df = pd.read_csv('./train.csv')
train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CONFIG.SEED)

def get_mfcc_feature(df, train_mode=True):
    features = []
    labels = []
    for _, row in tqdm(df.iterrows()):
        # librosa패키지를 사용하여 wav 파일 load
        y, sr = librosa.load(row['path'], sr=CONFIG.SR)
        
        # librosa패키지를 사용하여 mfcc 추출
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CONFIG.N_MFCC)
        mfcc = np.mean(mfcc.T, axis=0)
        features.append(mfcc)

        if train_mode:
            label = row['label']
            label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
            label_vector[0 if label == 'fake' else 1] = 1
            labels.append(label_vector)

    if train_mode:
        return features, labels
    return features

train_mfcc, train_labels = get_mfcc_feature(train, True)
val_mfcc, val_labels = get_mfcc_feature(val, True)

def get_mel_spectrogram(df, train_mode=True):
    features = []
    labels = []
    max_length = 0  # Track the maximum length of Mel spectrograms
    
    for _, row in tqdm(df.iterrows()):
        # Load audio file using librosa
        y, sr = librosa.load(row['path'], sr=CONFIG.SR)
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB scale
        
        # Ensure all spectrograms have the same shape
        if mel_spec.shape[1] > max_length:
            max_length = mel_spec.shape[1]
        
        features.append(mel_spec)

        if train_mode:
            label = row['label']
            label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
            label_vector[0 if label == 'fake' else 1] = 1
            labels.append(label_vector)

    # Pad all spectrograms to the maximum length
    for i in range(len(features)):
        features[i] = np.pad(features[i], ((0, 0), (0, max_length - features[i].shape[1])), mode='constant')
    
    if train_mode:
        return features, labels
    return features

train_mel_spec, train_labels = get_mel_spectrogram(train, True)
val_mel_spec, val_labels = get_mel_spectrogram(val, True)


class CustomDatasetMFCC(Dataset):
    def __init__(self, mfcc, label):
        self.mfcc = mfcc
        self.label = label

    def __len__(self):
        return len(self.mfcc)

    def __getitem__(self, index):
        if self.label is not None:
            return self.mfcc[index], self.label[index]
        return self.mfcc[index]

train_dataset_mfcc = CustomDatasetMFCC(train_mfcc, train_labels)
val_dataset_mfcc = CustomDatasetMFCC(val_mfcc, val_labels)

class CustomDatasetMel(Dataset):
    def __init__(self, mel_spec, label):
        self.mel_spec = mel_spec
        self.label = label

    def __len__(self):
        return len(self.mel_spec)

    def __getitem__(self, index):
        if self.label is not None:
            return self.mel_spec[index], self.label[index]
        return self.mel_spec[index]

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

class CNN(nn.Module):
    def __init__(self, input_channels=128, num_classes= CONFIG.N_CLASSES):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Adjust the size for fully connected layers based on the output of convolutional layers
        self.fc_input_size = 64 * 16 * 16
        
        self.fc1 = nn.Linear(self.fc_input_size, 128)  # Adjusted to match the input size
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        
        # Flatten the tensor before passing to fully connected layers
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

# Mel spectrogram 데이터의 shape을 맞추기 위해 input_channels를 수정합니다.
model_mel = CNN(input_channels=128)

optimizer_mel = torch.optim.Adam(params=model_mel.parameters(), lr=CONFIG.LR)


class BiLSTM(nn.Module):
    def __init__(self, input_dim=CONFIG.N_MFCC, hidden_dim=64, output_dim=CONFIG.N_CLASSES):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(self.relu(lstm_out[:, -1, :]))  # Take the last time-step's output
        out = torch.sigmoid(out)
        return out

model_mfcc = BiLSTM()
optimizer_mfcc = torch.optim.Adam(params=model_mfcc.parameters(), lr=CONFIG.LR)

from sklearn.metrics import roc_auc_score

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


def multiLabel_AUC(y_true, y_scores):
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        auc_scores.append(auc)
    mean_auc_score = np.mean(auc_scores)
    return mean_auc_score
    


# Training CNN model with Mel spectrograms
model_mel = CNN()
optimizer_mel = torch.optim.Adam(params=model_mel.parameters(), lr=CONFIG.LR)
infer_model_mel = train(model_mel, optimizer_mel, train_loader_mel, val_loader_mel, device)

# Training BiLSTM model with MFCCs
model_mfcc = BiLSTM()
optimizer_mfcc = torch.optim.Adam(params=model_mfcc.parameters(), lr=CONFIG.LR)
infer_model_mfcc = train(model_mfcc, optimizer_mfcc, train_loader_mfcc, val_loader_mfcc, device)

test_df = pd.read_csv('./test.csv')
test_mel_spec = get_mel_spectrogram(test_df, train_mode=False)

class CustomDatasetMelTest(Dataset):
    def __init__(self, mel_spec):
        self.mel_spec = mel_spec

    def __len__(self):
        return len(self.mel_spec)

    def __getitem__(self, index):
        return self.mel_spec[index]

# Create test dataset and DataLoader
test_dataset_mel = CustomDatasetMelTest(test_mel_spec)
test_loader_mel = DataLoader(
    test_dataset_mel,
    batch_size=CONFIG.BATCH_SIZE,
    shuffle=False
)

test_mfcc = get_mfcc_feature(test_df, train_mode=False)
class CustomDatasetMFCC(Dataset):
    def __init__(self, mfcc):
        self.mfcc = mfcc

    def __len__(self):
        return len(self.mfcc)

    def __getitem__(self, index):
        return self.mfcc[index]

# Create test dataset and DataLoader
test_dataset_mfcc = CustomDatasetMFCC(test_mfcc)
test_loader_mfcc = DataLoader(
    test_dataset_mfcc,
    batch_size=CONFIG.BATCH_SIZE,
    shuffle=False
)



def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in tqdm(iter(test_loader)):
            features = features.to(device)
            
            probs = model(features)

            probs  = probs.cpu().detach().numpy()
            predictions += probs.tolist()
    return predictions

# Inference with CNN model (Mel spectrograms)
preds_mel = inference(infer_model_mel, test_loader_mel, device)

# Inference with BiLSTM model (MFCCs)
preds_mfcc = inference(infer_model_mfcc, test_loader_mfcc, device)


# Convert predictions to numpy arrays
preds_mel = np.array(preds_mel)
preds_mfcc = np.array(preds_mfcc)

# Ensemble by averaging predictions
ensemble_preds = (preds_mel + preds_mfcc) / 2.0

submit = pd.read_csv('./sample_submission.csv')
# Assuming 'submit' is your DataFrame for submission
submit.iloc[:, 1:] = ensemble_preds
submit.to_csv('./ensemble_submission.csv', index=False)
