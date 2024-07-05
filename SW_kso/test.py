# inference.py
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import CustomDatasetMFCC, CustomDatasetMel, get_mel_spectrogram, get_mfcc_feature
from src.model import VGG19, BiLSTM
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

# Load test data
# test_df = pd.read_csv('./data/test.csv')
# test_mel_spec = get_mel_spectrogram(test_df, train_mode=False)
# test_mfcc = get_mfcc_feature(test_df, train_mode=False)

test_mfcc = np.load('./data/test_mfcc.npy', allow_pickle=True)
test_mel_spec = np.load('./data/test_mel_spec.npy', allow_pickle=True)
    

# Create test datasets and DataLoaders
test_dataset_mel = CustomDatasetMel(test_mel_spec, None)
test_loader_mel = DataLoader(
    test_dataset_mel,
    batch_size=CONFIG['BATCH_SIZE'],
    shuffle=False
)

test_dataset_mfcc = CustomDatasetMFCC(test_mfcc, None)
test_loader_mfcc = DataLoader(
    test_dataset_mfcc,
    batch_size=CONFIG['BATCH_SIZE'],
    shuffle=False
)


model_mel = VGG19(num_classes=CONFIG['N_CLASSES'])
model_mel.load_state_dict(torch.load('./data/checkpoints/best_model_mel01.pth')) 
preds_mel = inference(model_mel, test_loader_mel, device)

# Inference with BiLSTM model (MFCCs)
model_mfcc = BiLSTM(input_dim=CONFIG['N_MFCC'], hidden_dim=64, output_dim=CONFIG['N_CLASSES'])
model_mfcc.load_state_dict(torch.load('./data/checkpoints/best_model_mfcc01.pth'))  # Load trained model state dict
preds_mfcc = inference(model_mfcc, test_loader_mfcc, device)

preds_mel = np.array(preds_mel)
preds_mfcc = np.array(preds_mfcc)


ensemble_preds = (preds_mel + preds_mfcc) / 2.0

submit = pd.read_csv('./data/sample_submission.csv')
# submit.iloc[:, 1:] = ensemble_preds

submit.iloc[:, 1:] = ensemble_preds

submit.to_csv('./data/submission/ensemble_submission01.csv', index=False)
