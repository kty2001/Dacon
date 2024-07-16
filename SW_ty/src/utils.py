import os
import random

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score

class Config:
    # Data
    SR = 32000
    N_MFCC = 13
    N_MELS = 128
    # Dataset
    ROOT_FOLDER = './'
    # Training
    N_CLASSES = 2
    BATCH_SIZE = 16
    IMAGE_SIZE = (256, 256)
    N_EPOCHS = 5
    LR = 1e-4
    # Others
    SEED = 42

CONFIG = Config()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def split_data(csv_path, seed):
    df = pd.read_csv(csv_path)
    train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=seed)

    train.to_csv('data/train_answer.csv', index=False)
    val.to_csv('data/val_answer.csv', index=False)

def multiLabel_AUC(y_true, y_scores):
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        auc_scores.append(auc)
    mean_auc_score = np.mean(auc_scores)

    return mean_auc_score

def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    bin_totals = np.histogram(y_prob, bins=np.linspace(0, 1, n_bins + 1), density=False)[0]
    non_empty_bins = bin_totals > 0
    bin_weights = bin_totals / len(y_prob)
    bin_weights = bin_weights[non_empty_bins]
    prob_true = prob_true[:len(bin_weights)]
    prob_pred = prob_pred[:len(bin_weights)]
    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
    return ece

def auc_brier_ece(y_true, y_prob):    
    # Calculate AUC for each class
    auc_scores = []
    auc = roc_auc_score(y_true, y_prob)
    auc_scores.append(auc)

    # Brier Score
    brier_scores = []
    brier = mean_squared_error(y_true, y_prob)
    brier_scores.append(brier)
    
    # ECE
    ece_scores = []
    ece = expected_calibration_error(y_true, y_prob)
    ece_scores.append(ece)
    
    return auc_scores, brier_scores, ece_scores