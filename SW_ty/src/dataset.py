import os
import glob
import shutil
import pickle

import librosa
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

from src.utils import CONFIG


class VoiceDataset(Dataset):
    def __init__(self, image_path, csv_path, argu_data, transform, mode):
        self.image_path = image_path
        self.csv_data = pd.read_csv(csv_path)
        self.argu_data = argu_data
        self.transform = transform
        self.mode = mode
        self.train = True if 'label' in self.csv_data.columns else False

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, index):
        image_data = self.csv_data.loc[index]

        if self.train == True:
            argument = np.random.choice([0, 1], p=[0.5, 0.5])

            if argument == 0:
                image = Image.open(os.path.join(self.image_path, f"{image_data['id']}.png")).convert('RGB')
            else:
                save_fig = os.path.join(f'data/argument/argument_{image_data["id"]}.png')
                argument_image(image_data, self.argu_data, save_fig)
                image = Image.open(save_fig).convert('RGB')
                # shutil.rmtree(save_fig)

            image = np.array(image, dtype=np.float32)
            image = self.transform(image)

            if self.mode == 'real':
                label = np.array([0, 1] if image_data['label'] == 'real' else [1, 0], dtype=np.float32)
            elif self.mode == 'fake':
                label = np.array([0, 1] if image_data['label'] == 'fake' else [1, 0], dtype=np.float32)

            return image, label

        else:
            image = Image.open(os.path.join(self.image_path, f"{image_data['id']}.png")).convert('RGB')
            image = np.array(image, dtype=np.float32)
            image = self.transform(image)
            
            return image

def argument_image(image_data, argu_data, save_fig):
    y1, _ = librosa.load(os.path.join("data", image_data['path']), sr=CONFIG.SR)
    y2 = argu_data[np.random.randint(0, 1264)]
    
    max_length = max(len(y1), len(y2))
    if np.random.choice([True, False]):
        y1 = np.pad(y1, (0, max_length - len(y1)), 'constant')
        y2 = np.pad(y2, (0, max_length - len(y2)), 'constant')
    else:
        y1 = np.pad(y1, (max_length - len(y1), 0), 'constant')
        y2 = np.pad(y2, (max_length - len(y2), 0), 'constant')
    y = y1 + y2

    S = librosa.feature.melspectrogram(y=y, sr=CONFIG.SR, n_mels=CONFIG.N_MELS, fmax=8192)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # plt.figure(figsize=(8, 4))
    plt.figure(figsize=(4, 2))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=CONFIG.SR, fmax=8192)
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(save_fig, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()