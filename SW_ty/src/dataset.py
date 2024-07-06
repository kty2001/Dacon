import os

import pandas as pd
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


class VoiceDataset(Dataset):
    def __init__(self, image_path, csv_path, transform, mode):
        self.image_path = image_path
        self.csv_data = pd.read_csv(csv_path)
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, index):
        image_data = self.csv_data.loc[index]

        image = Image.open(os.path.join(self.image_path, f"{image_data['id']}.png")).convert('RGB')
        image = np.array(image, dtype=np.float32)
        image = self.transform(image)
        
        if self.mode == "real":
            if 'label' in self.csv_data.columns:
                label = np.array([0, 1] if image_data['label'] == 'real' else [1, 0], dtype=np.float32)
                return image, label
            else:
                return image
        
        elif self.mode == 'fake':
            if 'label' in self.csv_data.columns:
                label = np.array([0, 1] if image_data['label'] == 'fake' else [1, 0], dtype=np.float32)
                return image, label
            else:
                return image
