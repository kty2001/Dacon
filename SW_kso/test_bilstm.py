# inference.py
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import CustomDatasetMFCC
from src.model import BiLSTM
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


test_mfcc = np.load('./data/test_mfcc.npy', allow_pickle=True)
test_dataset_mfcc = CustomDatasetMFCC(test_mfcc, None)
test_loader_mfcc = DataLoader(
    test_dataset_mfcc,
    batch_size=CONFIG['BATCH_SIZE'],
    shuffle=False
)

model_mfcc = BiLSTM(input_dim=CONFIG['N_MFCC'], hidden_dim=64, output_dim=CONFIG['N_CLASSES'])
model_mfcc.load_state_dict(torch.load('./data/checkpoints/best_model_mfcc.pth'))  # Load trained model state dict
preds_mfcc = inference(model_mfcc, test_loader_mfcc, device)

preds_mfcc = np.array(preds_mfcc)

submit = pd.read_csv('./data/sample_submission.csv')
# submit.iloc[:, 1:] = ensemble_preds

submit.iloc[:, 1:] = preds_mfcc

submit.to_csv('./data/submission/Bilstm_submission.csv', index=False)
