import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import lightning as L

from src.model import PricePredictionLSTM
from src.dataset import AgriculturePriceDataset
from src.utils import CFG, process_data


class DatasetModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.trainset = AgriculturePriceDataset(dataframe, window_size=9, prediction_length=3, is_test=False)
            self.validset = AgriculturePriceDataset(dataframe, window_size=9, prediction_length=3, is_test=True)
            
            pbar_outer.set_description(f"품목별 전처리 및 모델 학습 -> {품목명}")
            train_data, scaler = process_data("./data/train/train.csv", 
                                    "./data/train/meta/TRAIN_산지공판장_2018-2021.csv", 
                                    "./data/train/meta/TRAIN_전국도매_2018-2021.csv", 
                                    품목명)
            품목별_scalers[품목명] = scaler

            dataset = AgriculturePriceDataset(train_data)

            train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=True, pin_memory=True)


품목별_predictions = {}
품목별_scalers = {}

pbar_outer = tqdm(품목_리스트, desc="품목 처리 중", position=0)
for 품목명 in pbar_outer:
    pbar_outer.set_description(f"품목별 전처리 및 모델 학습 -> {품목명}")
    train_data, scaler = process_data("./data/train/train.csv", 
                              "./data/train/meta/TRAIN_산지공판장_2018-2021.csv", 
                              "./data/train/meta/TRAIN_전국도매_2018-2021.csv", 
                              품목명)
    품목별_scalers[품목명] = scaler
    dataset = AgriculturePriceDataset(train_data)

    # 데이터를 train과 validation으로 분할
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(train_data, CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, CFG.batch_size, shuffle=False)

    input_size = len(dataset.numeric_columns)
    
    model = PricePredictionLSTM(input_size, CFG.hidden_size, CFG.num_layers, CFG.output_size)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), CFG.learning_rate)
    
    best_val_loss = float('inf')
    os.makedirs('models', exist_ok=True)

    for epoch in range(CFG.epoch):
        train_loss = train_model(model, train_loader, criterion, optimizer, CFG.epoch)
        val_loss = evaluate_model(model, val_loader, criterion)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'models/best_model_{품목명}.pth')
        
        print(f'Epoch {epoch+1}/{CFG.epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    print(f'Best Validation Loss for {품목명}: {best_val_loss:.4f}')
    
    품목_predictions = []

    ### 추론 
    pbar_inner = tqdm(range(25), desc="테스트 파일 추론 중", position=1, leave=False)
    for i in pbar_inner:
        test_file = f"./data/test/TEST_{i:02d}.csv"
        산지공판장_file = f"./data/test/meta/TEST_산지공판장_{i:02d}.csv"
        전국도매_file = f"./data/test/meta/TEST_전국도매_{i:02d}.csv"
        
        test_data, _ = process_data(test_file, 산지공판장_file, 전국도매_file, 품목명, scaler=품목별_scalers[품목명])
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
        price_scaler.min_ = 품목별_scalers[품목명].min_[price_column_index]
        price_scaler.scale_ = 품목별_scalers[품목명].scale_[price_column_index]
        predictions_original_scale = price_scaler.inverse_transform(predictions_reshaped)
        #print(predictions_original_scale)
        
        if np.isnan(predictions_original_scale).any():
            pbar_inner.set_postfix({"상태": "NaN"})
        else:
            pbar_inner.set_postfix({"상태": "정상"})
            품목_predictions.extend(predictions_original_scale.flatten())

            
    품목별_predictions[품목명] = 품목_predictions
    pbar_outer.update(1)
