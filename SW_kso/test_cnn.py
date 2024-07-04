# inference.py
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import CustomDatasetMel
from src.model import VGG19
from src.utils import multiLabel_AUC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG = {
    'SR': 32000,
    'N_MFCC': 13,
    'N_CLASSES': 2,
    'BATCH_SIZE': 128
}

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in tqdm(iter(test_loader)):
            features = features.to(device)
            probs = model(features)
            probs = torch.softmax(probs, dim=1)
            probs  = probs.cpu().detach().numpy()
            predictions += probs.tolist()
    return predictions

test_mel_spec = np.load('./data/test_mel_spec.npy', allow_pickle=True)

# Create test datasets and DataLoaders
test_dataset_mel = CustomDatasetMel(test_mel_spec, None)
test_loader_mel = DataLoader(
    test_dataset_mel,
    batch_size=CONFIG['BATCH_SIZE'],
    shuffle=False
)

model_mel = VGG19(num_classes=CONFIG['N_CLASSES'])
model_mel.load_state_dict(torch.load('./data/checkpoints/best_model_mel.pth')) 
preds_mel = inference(model_mel, test_loader_mel, device)


preds_mel = np.array(preds_mel)




submit = pd.read_csv('./data/sample_submission.csv')
# submit.iloc[:, 1:] = ensemble_preds

submit.iloc[:, 1:] = preds_mel

submit.to_csv('./data/submission/cnn_submission.csv', index=False)
