import os
import random

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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