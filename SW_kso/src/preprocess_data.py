# preprocess_data.py

import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Config:
    SR = 32000
    N_MFCC = 13
    N_CLASSES = 2
    ROOT_FOLDER = './'
    SEED = 42

CONFIG = Config()

def get_mfcc_feature(df, train_mode=True):
    features = []
    labels = []
    for _, row in tqdm(df.iterrows(), desc="Extracting MFCC"):
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

def preprocess_and_save_data():
    # df = pd.read_csv('./data/train.csv')
    # train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CONFIG.SEED)

    # # Extract and save MFCC features
    # train_mfcc, train_labels_mfcc = get_mfcc_feature(train, True)
    # val_mfcc, val_labels_mfcc = get_mfcc_feature(val, True)
    # np.save('./data/train_mfcc.npy', train_mfcc)
    # np.save('./data/val_mfcc.npy', val_mfcc)
    # np.save('./data/train_labels_mfcc.npy', train_labels_mfcc)
    # np.save('./data/val_labels_mfcc.npy', val_labels_mfcc)

    # # Extract and save Mel spectrogram features
    # train_mel_spec, train_labels_mel = get_mel_spectrogram(train, True)
    # val_mel_spec, val_labels_mel = get_mel_spectrogram(val, True)
    # np.save('./data/train_mel_spec.npy', train_mel_spec)
    # np.save('./data/val_mel_spec.npy', val_mel_spec)
    # np.save('./data/train_labels_mel.npy', train_labels_mel)
    # np.save('./data/val_labels_mel.npy', val_labels_mel)
    test = pd.read_csv('./data/test.csv')

    test_mel_spec = get_mel_spectrogram(test, False)
    np.save('./data/test_mel_spec.npy', test_mel_spec)
    

    test_mfcc = get_mfcc_feature(test, False)
    np.save('./data/test_mfcc.npy', test_mfcc)
    
    
    

if __name__ == '__main__':
    preprocess_and_save_data()
