from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import VoiceDataset
from src.model import EfficientNetB7Classifier
from src.utils import CONFIG

import warnings
warnings.filterwarnings('ignore')

def model_inference(test_loader, model, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in tqdm(iter(test_loader)):
            images = images.to(device)
            
            probs = model(images)

            probs = probs.cpu().detach().numpy()
            predictions += probs.tolist()
    return predictions

def inference(device, mode):
    test = pd.read_csv('data/test.csv')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(CONFIG.IMAGE_SIZE),
    ])

    test_dataset = VoiceDataset(image_path='data/test_melspec', csv_path='data/test.csv', transform=transform, mode=mode) # need for modifing
    test_loader = DataLoader( test_dataset, batch_size=CONFIG.BATCH_SIZE*16, shuffle=False)


    infer_model = EfficientNetB7Classifier(num_classes=CONFIG.N_CLASSES).to(device)
    infer_model.load_state_dict(torch.load(f'weights/effi_5epoch_melspec_{mode}.pth')) # need for modifing

    return model_inference(test_loader, infer_model, device)

real_preds = np.array(inference(device='cuda', mode='real'))
fake_preds = np.array(inference(device='cuda', mode='fake'))

submit = pd.read_csv('submitfile/sample_submission.csv')
submit.iloc[:, 1] = fake_preds[:, 1]
submit.iloc[:, 2] = real_preds[:, 1]
submit.to_csv('submitfile/effi_5epoch_melspec_sepa.csv', index=False) # need for modifing
