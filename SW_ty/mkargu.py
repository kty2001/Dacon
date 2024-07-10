import os
import shutil
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from src.utils import seed_everything, split_data, CONFIG
import librosa
import matplotlib.pyplot as plt
import pickle

seed_everything(CONFIG.SEED)
split_data("data/train.csv", CONFIG.SEED)

def argument_visualize(csv_path, save_path):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    df = pd.read_csv(csv_path)

    with open('data/features/unlabeled_data_features.pkl', 'rb') as f:
        argu_data = pickle.load(f)

    for i in tqdm(range(len(df))):
        image_data = df.loc[i]

        y1, _ = librosa.load(os.path.join('data', image_data['path']), sr=CONFIG.SR)
        y2 = argu_data[np.random.randint(0, 1264)]

        max_length = max(len(y1), len(y2))
        if np.random.choice([True, False]):
            y1 = np.pad(y1, (0, max_length - len(y1)), 'constant')
            y2 = np.pad(y2, (0, max_length - len(y2)), 'constant')
        else:
            y1 = np.pad(y1, (max_length - len(y1), 0), 'constant')
            y2 = np.pad(y2, (max_length - len(y2), 0), 'constant')
        y = y1 + y2

        melspecs = librosa.feature.melspectrogram(y=y, sr=CONFIG.SR, n_mels=128, fmax=8192)  # n_mels: 멜 필터의 개수, fmax: 주파수 최대값
        melspecs_db = librosa.power_to_db(melspecs, ref=np.max)

        plt.figure(figsize=(4, 2))
        librosa.display.specshow(melspecs_db, x_axis='time', y_axis='mel', sr=CONFIG.SR, fmax=8192)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{save_path}/argu_{image_data['id']}.png", dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

argument_visualize('data/train_answer.csv', 'data/argu')