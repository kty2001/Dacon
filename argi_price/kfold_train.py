import os
import warnings
import argparse
import random
import torch.utils
import torch.utils.data
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

from src.model import PricePredictionLSTM, model_train, model_valid
from src.dataset import AgriculturePriceDataset
from src.utils import CFG, PRODUCT_LIST, process_data, result


warnings.filterwarnings('ignore')


seed = 3636
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str)
args = parser.parse_args()


rootname = args.filename
products_predictions = {}
products_scalers = {}
train_result = {}

kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

pbar_outer = tqdm(PRODUCT_LIST, desc="품목 처리 중", position=0)
for product_name in pbar_outer:
    pbar_outer.set_description(f"품목별 전처리 및 모델 학습 -> {product_name}")
    product_df, scaler = process_data("./data/train/train.csv", 
                              "./data/train/meta/TRAIN_산지공판장_2018-2021.csv", 
                              "./data/train/meta/TRAIN_전국도매_2018-2021.csv", 
                              product_name, scaler=None)
    products_scalers[product_name] = scaler
    dataset = AgriculturePriceDataset(product_df)

    best_val_losses = []

    for i, (train_index, valid_index) in enumerate(kfold.split(dataset)):
        train_data = torch.utils.data.Subset(dataset, train_index)
        val_data = torch.utils.data.Subset(dataset, valid_index)        

        train_loader = DataLoader(train_data, CFG.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, CFG.batch_size, shuffle=False)

        input_size = len(dataset.numeric_columns)
        model = PricePredictionLSTM(input_size, CFG.hidden_size, CFG.num_layers, CFG.output_size)
        criterion = nn.L1Loss()
        optimizer = torch.optim.AdamW(model.parameters(), CFG.learning_rate)
        
        best_val_loss = float('inf')
        os.makedirs(f'weights/{rootname}', exist_ok=True)

        print(f"---------- kfold {i} ----------")
        for epoch in range(CFG.epoch):
            train_loss = model_train(model, train_loader, criterion, optimizer, epoch)
            val_loss = model_valid(model, val_loader, criterion)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'weights/{rootname}/best_{product_name}_fold{i}.pth')
            
            if (epoch+1) % 30 == 0:
                print(f'Epoch {epoch+1}/{CFG.epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        best_val_losses.append(best_val_loss)
    
    average_val_loss = sum(best_val_losses) / len(best_val_losses)
    train_result[product_name] = round(average_val_loss, 5)
    print(f'Best Validation Loss for {product_name}: {average_val_loss:.4f}')
    
    fold_prediction = []

    for fold_num in range(kfold.n_splits):
        product_prediction = []
        model.load_state_dict(torch.load(f'weights/{rootname}/best_{product_name}_fold{fold_num}.pth'))

        pbar_inner = tqdm(range(25), desc="테스트 파일 추론 중", position=1, leave=False)
        for i in pbar_inner:
            test_file = f"./data/test/TEST_{i:02d}.csv"
            산지공판장_file = f"./data/test/meta/TEST_산지공판장_{i:02d}.csv"
            전국도매_file = f"./data/test/meta/TEST_전국도매_{i:02d}.csv"
            
            test_data, _ = process_data(test_file, 산지공판장_file, 전국도매_file, product_name, scaler=products_scalers[product_name])
            test_dataset = AgriculturePriceDataset(test_data, is_test=True)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

            model.eval()
            predictions = []
            with torch.no_grad():
                for batch in test_loader:
                    output = model(batch)
                    predictions.append(output.numpy())
            
            predictions_array = np.concatenate(predictions)

            # 예측값을 원래 스케일로 복원
            price_column_index = test_data.columns.get_loc(test_dataset.price_column)
            predictions_reshaped = predictions_array.reshape(-1, 1)
            
            # 가격 열에 대해서만 inverse_transform 적용
            price_scaler = MinMaxScaler()
            price_scaler.min_ = products_scalers[product_name].min_[price_column_index]
            price_scaler.scale_ = products_scalers[product_name].scale_[price_column_index]
            predictions_original_scale = price_scaler.inverse_transform(predictions_reshaped)
            
            if np.isnan(predictions_original_scale).any():
                pbar_inner.set_postfix({"상태": "NaN"})
            else:
                pbar_inner.set_postfix({"상태": "정상"})
                product_prediction.extend(predictions_original_scale.flatten())
        
        fold_prediction.append(product_prediction)
            
    products_predictions[product_name] = np.mean(fold_prediction, axis=0)
    pbar_outer.update(1)

result(filename=f"{rootname}.csv", product_predictions=products_predictions)

print(train_result)
print()