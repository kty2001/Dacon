# dataset.py

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class Config:
    SR = 32000
    N_MFCC = 13
    N_CLASSES = 2
    BATCH_SIZE = 128
    SEED = 42

CONFIG = Config()

def get_mfcc_feature(df, train_mode=True):
    features = []
    labels = []
    for _, row in tqdm(df.iterrows()):
        y, sr = librosa.load(row['path'], sr=CONFIG.SR)
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

def get_mel_spectrogram(df, train_mode=True):
    features = []
    labels = []
    max_length = 0
    
    for _, row in tqdm(df.iterrows(), desc="Extracting Mel Spectrogram"):
        y, sr = librosa.load(row['path'], sr=CONFIG.SR)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        if mel_spec.shape[1] > max_length:
            max_length = mel_spec.shape[1]
        
        features.append(mel_spec)
        
        if train_mode:
            label = row['label']
            label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
            label_vector[0 if label == 'fake' else 1] = 1
            labels.append(label_vector)

    for i in range(len(features)):
        # Apply random padding to ensure translation invariance
        features[i] = time_shift(features[i])
        # Apply zero-padding to make all spectrograms of the same length
        features[i] = np.pad(features[i], ((0, 0), (0, max_length - features[i].shape[1])), mode='constant')
        
        # Resize spectrogram to (224, 224) for VGG input
        features[i] = resize_spectrogram(features[i])

    if train_mode:
        return features, labels
    return features

def time_shift(mel_spec):
    """Apply random time shift to the mel spectrogram"""
    shift = np.random.randint(low=0, high=mel_spec.shape[1])
    shifted = np.roll(mel_spec, shift, axis=1)
    return shifted

def resize_spectrogram(mel_spec):
    """Resize mel spectrogram to (224, 224) for VGG input"""
    mel_spec = mel_spec[:224, :224]  # Crop or pad to (224, 224)
    mel_spec = np.expand_dims(mel_spec, axis=0)  # Add channel dimension
    return mel_spec


class CustomDatasetMFCC(Dataset):
    def __init__(self, mfcc, labels=None):
        self.mfcc = mfcc
        self.labels = labels

    def __len__(self):
        return len(self.mfcc)

    def __getitem__(self, index):
        mfcc_data = torch.tensor(self.mfcc[index], dtype=torch.float32)
        if self.labels is not None:
            label = torch.tensor(self.labels[index], dtype=torch.float32)
            return mfcc_data, label
        else:
            return mfcc_data

class CustomDatasetMel(Dataset):
    def __init__(self, mel_spec, labels=None):
        self.mel_spec = mel_spec
        self.labels = labels

    def __len__(self):
        return len(self.mel_spec)

    def __getitem__(self, index):
        mel_data = torch.tensor(self.mel_spec[index], dtype=torch.float32)
        if self.labels is not None:
            label = torch.tensor(self.labels[index], dtype=torch.float32)
            return mel_data, label
        else:
            return mel_data
