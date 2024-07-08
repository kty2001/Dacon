from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import VoiceDataset
from src.model import EfficientNetB7Classifier, ResNet50Classication
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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(CONFIG.IMAGE_SIZE),
    ])

    test_dataset = VoiceDataset(image_path='data/test_melspec', csv_path='data/test.csv', transform=transform, mode=mode) # need for modifing
    test_loader = DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE*16, shuffle=False)

    infer_model = EfficientNetB7Classifier(num_classes=CONFIG.N_CLASSES).to(device)
    infer_model.load_state_dict(torch.load(f'weights/effi_5epoch_argu50_{mode}.pth')) # need for modifing
    # infer_model2 = ResNet50Classication(num_classes=CONFIG.N_CLASSES).to(device)
    # infer_model2.load_state_dict(torch.load(f'weights/resnet_5epoch_melspec_{mode}.pth')) # need for modifing

    return np.array(model_inference(test_loader, infer_model, device))
    # return np.array(model_inference(test_loader, infer_model, device)), np.array(model_inference(test_loader, infer_model2, device))

real_preds = inference(device='cuda', mode='real')
fake_preds = inference(device='cuda', mode='fake')
# real_preds1, real_preds2 = inference(device='cuda', mode='real')
# fake_preds1, fake_preds2 = inference(device='cuda', mode='fake')

# real_preds = (real_preds1 + real_preds2) / 2
# fake_preds = (fake_preds1 + fake_preds2) / 2

csv_name = 'submitfile/effi_5epoch_argu50.csv'  # need for modifing
submit = pd.read_csv('submitfile/sample_submission.csv')
submit.iloc[:, 1] = fake_preds[:, 1]
submit.iloc[:, 2] = real_preds[:, 1]
submit.to_csv(csv_name, index=False)
