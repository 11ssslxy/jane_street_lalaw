import pandas as pd
import numpy as np
import polars as pl

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna

from tqdm import tqdm

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

TARGET = 'responder_6'
TIME_COLS = ['date_id', 'time_id']
LEAD_COLS = ['symbol_id', 'weight']
RESPONDER_COLS = [f"responder_{i}" for i in range(9)]
RESPONDER_COLS_LAG = [f"responder_{idx}_lag_1" for idx in range(9)]
FEAT_COLS = [f"feature_{i:02d}" for i in range(79)]

def read_data(path = f'/kaggle/input/train-test-data241218v2/train_data.parquet',slice_num = 0):
    df = pl.read_parquet(path)[-slice_num:]
    return df
    
class TimeSeriesDataset(Dataset):
    def __init__(self, data, target, weight):
        self.data = data
        self.target = target
        self.weight = weight

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx], self.weight[idx]

def get_train_loader(path, slice_num, batch_size = 8196, input_size = 88):
    df = read_data(path = path, slice_num = slice_num)
    
    data = df[FEAT_COLS+RESPONDER_COLS_LAG]
    target = df[[TARGET]]
    weights = df[['weight']]
    
    # 转换为PyTorch张量
    data = data.to_torch(dtype=pl.Float32)
    target = target.to_torch(dtype=pl.Float32)
    weights = weights.to_torch(dtype = pl.Float32)
    
    data = data.reshape(slice_num,1,input_size)
    target = target.reshape(slice_num,1)
    weights = weights.reshape(slice_num,1)
    
    dataset = TimeSeriesDataset(data, target, weights)
    data_loader = DataLoader(dataset, batch_size, shuffle=True)
    return data_loader
    
class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(AttentionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM层
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Attention机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        # LSTM输出
        out, _ = self.lstm(x, (h0, c0))  # out: [batch_size, seq_length, hidden_dim]

        # Attention机制
        attn_weights = self.attention(out).squeeze(-1)  # [batch_size, seq_length]
        attn_weights = torch.softmax(attn_weights, dim=1)  # 归一化
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), out).squeeze(1)  # [batch_size, hidden_dim]

        # 最终输出
        output = self.fc(attn_applied)  # [batch_size, output_dim]
        return output

class WCriterion(nn.Module):
    def __init__(self):
        super(WCriterion, self).__init__()

    def forward(self, y_pred, y_true, w):
        """
        自定义加权损失函数：
        1 - 加权均方误差 / 加权真实值平方和
        """

        numerator = torch.sum((y_true - y_pred)**2 * w)  # 加权均方误差
        denominator = torch.sum(y_true**2 * w)           # 加权真实值平方和
        # print("numerator:"  , numerator.item(), "denominator:" , denominator.item() )
        # 避免 denominator 为 0 的情况
        if denominator == 0:
            raise ValueError("Denominator in custom loss is zero, check your data or weights.")
        
        loss = 1 - numerator / denominator
        return loss
        
def train_model(model, optimizer, criterion, train_loader, device):
    model.train()
    running_loss = 0.0
    for i, (X_batch, y_batch, w) in tqdm(enumerate(train_loader)):
        t = X_batch.shape[0]
        w, X_batch, y_batch = w.to(device), X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch, w)
        # print('?loss:',loss.item(), ' t:', t, '  ', loss.item() * t)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * t
        # print('loss:',loss.item(), ' t:', t, '  ', loss.item() * t)
        # del w, X_batch, y_batch
    # print('!!!!',running_loss / len(train_loader.dataset))
    return running_loss / len(train_loader.dataset)

def validate_model(model, criterion, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for i, (X_batch, y_batch, w) in tqdm(enumerate(val_loader)):
            t = X_batch.shape[0]
            w, X_batch, y_batch = w.to(device), X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch, w)
            running_loss += loss.item() * t
            # del w, X_batch, y_batch
    return running_loss / len(val_loader.dataset)

def objective(trial):
    global local_best_value, best_model
    # 获取超参数
    hidden_dim = trial.suggest_int("hidden_dim", 32, 2048)
    layer_dim = trial.suggest_int("layer_dim", 1,6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建模型，并将其移动到 GPU（如果可用）
    model = AttentionLSTM(input_dim, hidden_dim, layer_dim, output_dim).to(device)
    criterion = WCriterion()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, maximize=True)
    # 训练过程
    for epoch in range(num_epochs):
        train_loss = train_model(model, optimizer, criterion, train_dataset, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}')
    # test
    val_loss = validate_model(model, criterion, val_dataset, device)
    print('train_loss:', train_loss, '\nval_loss:', val_loss)
    if val_loss > local_best_value:
        best_model = model
        local_best_value = val_loss
        torch.save(best_model, '/kaggle/working/best_model.model')
    del model
    return val_loss

input_dim = 88
output_dim = 1
learning_rate = 0.001
num_epochs = 1
batch_size = 8196
train_size = 30000000
best_model = None
local_best_value = -float("inf")

# GPU运行


train_dataset  = get_train_loader(path = f'/kaggle/input/train-test-data241218v2/train_data.parquet',\
                                  slice_num = train_size, batch_size = batch_size, input_size = input_dim)
val_dataset = get_train_loader(path = f'/kaggle/input/train-test-data241218v2/test_data.parquet',\
                               slice_num = int(train_size * 0.2), batch_size = batch_size, input_size = input_dim)



study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
print("Best trial:")
print(study.best_trial.params)



