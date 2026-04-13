# -*- coding: utf-8 -*-
"""
DNN Channel Prediction (Magnitude Version): A comprehensive script for training
and evaluating various deep learning models for predicting wireless channel magnitude.

[V4 - Final Audited Version with Feature Engineering]
- This version adds two auxiliary features to the model input:
    1. The mean magnitude across all subcarriers.
    2. The deviation of each subcarrier's magnitude from the mean.
- The data processing pipeline and model definitions have been updated accordingly.
- A critical dimension mismatch bug in the Transformer's decoder has been fixed.
- All function calls have been updated to pass necessary dimension arguments.
- This version uses MSELoss and ReduceLROnPlateau.
"""

import os
import time
import math
import scipy.io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import random
import contextlib
from typing import Tuple, List, Dict, Any, Optional

# ==============================================================================
# 1. 模型定义 (Model Definitions)
# ==============================================================================

class PositionalEncoder(nn.Module):
    def __init__(self, dropout: float = 0.1, max_seq_len: int = 5000, d_model: int = 512, batch_first: bool = True):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(self.x_dim)]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size: int, output_size: int, dim_val: int, n_encoder_layers: int, n_decoder_layers: int,
                 n_heads: int, dropout_encoder: float, dropout_decoder: float, dropout_pos_enc: float,
                 dim_feedforward_encoder: int, dim_feedforward_decoder: int,
                 max_seq_len: int, dec_seq_len: int, out_seq_len: int):
        super().__init__()
        self.dec_seq_len = dec_seq_len
        self.out_seq_len = out_seq_len
        self.encoder_input_layer = nn.Linear(in_features=input_size, out_features=dim_val)
        self.decoder_input_layer = nn.Linear(in_features=output_size, out_features=dim_val)
        self.positional_encoding_layer = PositionalEncoder(d_model=dim_val, dropout=dropout_pos_enc, max_seq_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_val, nhead=n_heads, 
                                                  dim_feedforward=dim_feedforward_encoder, 
                                                  dropout=dropout_encoder, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_encoder_layers, norm=None)
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_val, nhead=n_heads, 
                                                  dim_feedforward=dim_feedforward_decoder, 
                                                  dropout=dropout_decoder, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_decoder_layers, norm=None)
        self.linear_mapping = nn.Linear(in_features=dim_val, out_features=output_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        src = self.encoder(src=src)
        decoder_output = self.decoder_input_layer(tgt)
        decoder_output = self.decoder(tgt=decoder_output, memory=src, tgt_mask=tgt_mask, memory_mask=src_mask)
        decoder_output = self.linear_mapping(decoder_output)
        return decoder_output[:, -self.out_seq_len:, :]

class GRUNet_Seq2Seq(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, drop_prob: float = 0.2):
        super(GRUNet_Seq2Seq, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, h = self.gru(x, h)
        out = self.fc(out)
        return out, h

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

class LSTMNet_Seq2Seq(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, drop_prob: float = 0.2):
        super(LSTMNet_Seq2Seq, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, h: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out, h = self.lstm(x, h)
        out = self.fc(out)
        return out, h

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

class CNNNet(nn.Module):
    def __init__(self, lookback: int, input_channels: int, output_channels: int):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, lookback)
            dummy_output = self.pool(F.relu(self.conv1(dummy_input)))
            conv_output_size = dummy_output.numel()
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLPNet(nn.Module):
    def __init__(self, lookback: int, input_channels: int, output_channels: int):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(lookback * input_channels, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==============================================================================
# 2. 数据处理函数 (Data Processing Functions)
# ==============================================================================

def generate_samples_from_block_m2o(block_2d: np.ndarray, lookback: int, horizon: int, step: int, target_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    num_timesteps, num_features = block_2d.shape
    X, y = [], []
    required_len = (lookback - 1) * step + horizon
    num_samples = num_timesteps - required_len
    if num_samples <= 0: 
        return np.array([]), np.array([])
    for i in range(num_samples):
        input_end_idx = i + (lookback - 1) * step
        label_idx = input_end_idx + horizon
        indices = np.arange(i, input_end_idx + 1, step)
        X.append(block_2d[indices])
        y.append(block_2d[label_idx, :target_dim])
    return np.array(X), np.array(y)

def generate_samples_from_block_s2s(block_2d: np.ndarray, lookback: int, horizon: int, step: int, target_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    num_timesteps, num_features = block_2d.shape
    X, y = [], []
    required_len = (lookback - 1) * step + horizon 
    num_samples = num_timesteps - required_len
    if num_samples <= 0:
        return np.array([]), np.array([])
    for i in range(num_samples):
        input_end_idx = i + (lookback - 1) * step
        indices_in = np.arange(i, input_end_idx + 1, step)
        X.append(block_2d[indices_in])
        indices_out = np.arange(input_end_idx + 1, input_end_idx + 1 + horizon)
        y.append(block_2d[indices_out, :target_dim])
    return np.array(X), np.array(y)

def generate_square_subsequent_mask(dim1: int, dim2: int) -> torch.Tensor:
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

# ==============================================================================
# 3. 训练与评估函数 (Training & Evaluation Functions)
# ==============================================================================

@contextlib.contextmanager
def evaluating(model: nn.Module):
    model.eval()
    try:
        yield model
    finally:
        model.train()

def transformer_autoregressive_prediction(model: TimeSeriesTransformer, x: torch.Tensor, horizon: int, target_dim: int) -> torch.Tensor:
    """
    统一的、修正过的 Transformer 自回归预测函数。
    修复了维度不匹配问题，并消除了代码重复。
    """
    device = x.device
    
    src = model.encoder_input_layer(x)
    src = model.positional_encoding_layer(src)
    memory = model.encoder(src=src)
    
    # 关键修复：使用历史数据 x 的最后一步，并截取为目标维度，作为解码器初始输入
    decoder_input = x[:, -1:, :target_dim]
    outputs = []
    
    for _ in range(horizon):
        tgt_mask = generate_square_subsequent_mask(decoder_input.size(1), decoder_input.size(1)).to(device)
        tgt = model.decoder_input_layer(decoder_input)
        # 注意：此处 position_encoding_layer 可能不需要对 decoder_input 再次使用，
        # 但为保持与原始逻辑一致，暂且保留。在纯自回归中，位置信息通过 mask 和序列长度隐式处理。
        # tgt = model.positional_encoding_layer(tgt) 
        output = model.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        output = model.linear_mapping(output)
        
        next_input = output[:, -1:, :]
        decoder_input = torch.cat([decoder_input, next_input], dim=1)
        outputs.append(next_input)
    
    return torch.cat(outputs, dim=1)


def train(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
          optimizer: torch.optim.Optimizer, device: torch.device, horizon: int, target_dim: int) -> float:
    model.train()
    running_loss = 0.0
    for x, y, _ in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if isinstance(model, (GRUNet_Seq2Seq, LSTMNet_Seq2Seq)):
            h = model.init_hidden(x.size(0), device)
            out, h = model(x.float(), h)
        elif isinstance(model, TimeSeriesTransformer):
            out = transformer_autoregressive_prediction(model, x.float(), horizon, target_dim)
        else:
            out = model(x.float())
        y_for_loss = y[:, horizon-1:horizon, :] if out.dim() == 3 else y
        out_for_loss = out[:, -1:, :] if out.dim() == 3 else out
        loss = criterion(out_for_loss, y_for_loss.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device, 
             scaler_y: StandardScaler, horizon: int, target_dim: int) -> float:
    model.eval()
    total_loss_real = 0
    with torch.no_grad():
        for i, (x, y, y_clean) in enumerate(test_loader):
            x, y, y_clean = x.to(device), y.to(device), y_clean.to(device)
            if isinstance(model, (GRUNet_Seq2Seq, LSTMNet_Seq2Seq)):
                h = model.init_hidden(x.size(0), device)
                out, h = model(x.float(), h)
            elif isinstance(model, TimeSeriesTransformer):
                out = transformer_autoregressive_prediction(model, x.float(), horizon, target_dim)
            else:
                out = model(x.float())
            out_denorm = scaler_y.inverse_transform(
                out.cpu().numpy().reshape(-1, out.shape[-1])
            ).reshape(out.shape)
            y_clean_denorm = y_clean.cpu().numpy()
            if out_denorm.ndim == 3:
                pred_point = out_denorm[:, -1, :]
                true_point = y_clean_denorm[:, horizon-1, :]
            else:
                pred_point = out_denorm
                true_point = y_clean_denorm
            loss_real = np.mean((pred_point - true_point)**2)
            total_loss_real += loss_real * x.size(0)
    return total_loss_real / len(test_loader.dataset)

def predict(model: nn.Module, test_loader: DataLoader, device: torch.device, 
            scaler_y: StandardScaler, horizon: int, target_dim: int) -> Tuple[np.ndarray, float]:
    model.eval()
    all_predictions = []
    start_time = time.time()
    with torch.no_grad():
        for x, y, _ in test_loader:
            x, y = x.to(device), y.to(device)
            if isinstance(model, (GRUNet_Seq2Seq, LSTMNet_Seq2Seq)):
                h = model.init_hidden(x.size(0), device)
                out, h = model(x.float(), h)
            elif isinstance(model, TimeSeriesTransformer):
                out = transformer_autoregressive_prediction(model, x.float(), horizon, target_dim)
            else:
                out = model(x.float())
            out_denorm = scaler_y.inverse_transform(
                out.cpu().numpy().reshape(-1, out.shape[-1])
            ).reshape(out.shape)
            if out_denorm.ndim == 3:
                all_predictions.append(out_denorm[:, -1, :])
            else:
                all_predictions.append(out_denorm)
    end_time = time.time()
    total_time = end_time - start_time
    return np.concatenate(all_predictions, axis=0), total_time

def epoch_loop(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, 
               epochs: int, lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau, 
               criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, 
               scalers: Dict[str, StandardScaler], model_name: str, results_dir: str, horizon: int, target_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    training_loss, validation_loss = [], []
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 20
    
    scaler_y = scalers['y_m2o'] if model_name in ['CNN', 'MLP'] else scalers['y_s2s']
    
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train(model, train_loader, criterion, optimizer, device, horizon, target_dim)
        test_loss = evaluate(model, test_loader, device, scaler_y, horizon, target_dim)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | Train Loss (Norm.): {train_loss:.8f} | "
              f"Test Loss (Real): {test_loss:.6f} | LR: {current_lr:.6f} | "
              f"Time: {time.time()-start_time:.2f}s")

        if test_loss < best_loss:
            best_loss = test_loss
            epochs_no_improve = 0
            model_save_path = os.path.join(results_dir, f'{model_name}_best_model_magnitude.pth')
            torch.save(model.state_dict(), model_save_path)
            joblib.dump(scalers, os.path.join(results_dir, f'{model_name}_scalers_magnitude.gz'))
            print(f" -> Validation loss improved to {best_loss:.6f}. Saving {model_name} model to {model_save_path}")
        else:
            epochs_no_improve += 1
            
        lr_scheduler.step(test_loss)
        
        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break

    return np.array(training_loss), np.array(validation_loss)

def plot_losses(training_loss: np.ndarray, validation_loss: np.ndarray, model_name: str):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(training_loss, label='Training Loss (Normalized)', color='blue')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Normalized MSE Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2 = ax1.twinx()
    ax2.plot(validation_loss, label='Validation Loss (Real Scale)', color='orange')
    ax2.set_ylabel('Real Scale MSE Loss', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    plt.title(f'Loss Curve for {model_name} (Magnitude Prediction)')
    fig.tight_layout()
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.grid(True)
    plt.show()

def plot_predictions(true_vals: np.ndarray, preds: np.ndarray, model_name: str, 
                     naive_preds: np.ndarray, num_timesteps: int = 150, subcarrier_idx: int = 0):
    plt.figure(figsize=(15, 7))
    min_length = min(len(true_vals), len(preds), len(naive_preds))
    true_vals, preds, naive_preds = true_vals[:min_length], preds[:min_length], naive_preds[:min_length]
    time_steps = np.arange(min(min_length, num_timesteps))
    if subcarrier_idx >= true_vals.shape[1]:
        print(f"警告: 子载波索引 {subcarrier_idx} 超出范围，使用第一个子载波")
        subcarrier_idx = 0
    plt.plot(time_steps, true_vals[:num_timesteps, subcarrier_idx], label='True Magnitude', color='blue', linewidth=2)
    plt.plot(time_steps, preds[:num_timesteps, subcarrier_idx], label=f'{model_name} Predicted Magnitude', color='red', linestyle='--')
    if len(naive_preds) > 1:
        naive_shifted = np.roll(naive_preds[:num_timesteps, subcarrier_idx], 1)
        naive_shifted[0] = np.nan
        plt.plot(time_steps, naive_shifted, label='Naive Predicted Magnitude', color='green', linestyle=':')
    else:
        plt.plot(time_steps, naive_preds[:num_timesteps, subcarrier_idx], label='Naive Predicted Magnitude', color='green', linestyle=':')
    plt.title(f'Subcarrier #{subcarrier_idx} Predicted Magnitude Comparison (first {min(min_length, num_timesteps)} timesteps)')
    plt.xlabel('Time Step')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 4. 主执行流程 (Main Execution Flow)
# ==============================================================================

def main():
    print("--- 1. 配置实验参数 ---")
    project_root = '.'
    data_dir = os.path.join(project_root, 'data')
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    combined_data_file = os.path.join(data_dir, 'my_H_clean_combined_padded_optimized_test_autopilot_40m_for_training.mat')
    results_file = os.path.join(results_dir, 'performance_results_magnitude.csv')
    
    if not os.path.exists(combined_data_file):
        raise FileNotFoundError(f"数据文件不存在: {combined_data_file}")

    print(f"数据文件路径: {combined_data_file}")
    print(f"结果文件路径: {results_file}")

    NUM_SUBCARRIERS = 576
    INPUT_FEATURES = NUM_SUBCARRIERS * 2 + 1 
    OUTPUT_FEATURES = NUM_SUBCARRIERS 

    horizons = [1]
    lookback = 5
    ts_step = 1
    epochs = 500
    batch_size = 512
    learning_rates = {'GRU': 0.000025, 'LSTM': 0.0005, 'CNN': 0.0005, 'MLP': 0.0005, 'Transformer': 0.000025}

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备已设置为: {device}")

    print("\n--- 2. 加载并处理合并后的数据 ---")
    mat = scipy.io.loadmat(combined_data_file)
    combined_1d_array = mat['H'].flatten().astype(np.float32)
    complex_data = combined_1d_array.reshape(-1, NUM_SUBCARRIERS, 2)
    real_part = complex_data[:, :, 0]
    imag_part = complex_data[:, :, 1]
    H_2d_matrix = np.sqrt(real_part**2 + imag_part**2)
    print("数据转换完成。")

    print("正在创建并添加均值和差值辅助特征...")
    mean_magnitude = H_2d_matrix.mean(axis=1, keepdims=True)
    deviation_from_mean = H_2d_matrix - mean_magnitude
    H_augmented = np.concatenate([H_2d_matrix, mean_magnitude, deviation_from_mean], axis=1)
    print(f"辅助特征添加完成。数据维度从 {H_2d_matrix.shape} 增加到 {H_augmented.shape}")

    is_zero_row = ~H_2d_matrix.any(axis=1)
    block_change_indices = np.where(np.diff(is_zero_row))[0] + 1
    data_blocks_2d = np.split(H_augmented, block_change_indices)
    continuous_blocks = [block for block in data_blocks_2d if not np.all(block[:, :NUM_SUBCARRIERS] == 0)]
    print(f"成功恢复出 {len(continuous_blocks)} 个独立的连续数据块 (增强数据)。")

    print("\n--- 3. 开始实验循环 ---")
    performance_df = pd.DataFrame()
    if os.path.exists(results_file):
        performance_df = pd.read_csv(results_file, index_col=0)

    models_to_run = ['Naive Outdated','GRU','Transformer']

    for horizon in horizons:
        print(f"\n{'='*25} 正在处理 Horizon = {horizon} {'='*25}")

        split_idx = int(len(continuous_blocks) * 0.8)
        train_blocks, test_blocks = continuous_blocks[:split_idx], continuous_blocks[split_idx:]

        print("正在为 '多对一' 和 '序列到序列' 任务准备数据...")
        train_x_m2o_list, train_y_m2o_list = [], []
        train_x_s2s_list, train_y_s2s_list = [], []
        for block in train_blocks:
            x_m2o, y_m2o = generate_samples_from_block_m2o(block, lookback, horizon, ts_step, OUTPUT_FEATURES)
            if x_m2o.size > 0: train_x_m2o_list.append(x_m2o); train_y_m2o_list.append(y_m2o)
            x_s2s, y_s2s = generate_samples_from_block_s2s(block, lookback, horizon, ts_step, OUTPUT_FEATURES)
            if x_s2s.size > 0: train_x_s2s_list.append(x_s2s); train_y_s2s_list.append(y_s2s)
        train_x_m2o, train_y_m2o = np.concatenate(train_x_m2o_list), np.concatenate(train_y_m2o_list)
        train_x_s2s, train_y_s2s = np.concatenate(train_x_s2s_list), np.concatenate(train_y_s2s_list)

        test_x_m2o_list, test_y_m2o_list = [], []
        test_x_s2s_list, test_y_s2s_list = [], []
        for block in test_blocks:
            x_m2o, y_m2o = generate_samples_from_block_m2o(block, lookback, horizon, ts_step, OUTPUT_FEATURES)
            if x_m2o.size > 0: test_x_m2o_list.append(x_m2o); test_y_m2o_list.append(y_m2o)
            x_s2s, y_s2s = generate_samples_from_block_s2s(block, lookback, horizon, ts_step, OUTPUT_FEATURES)
            if x_s2s.size > 0: test_x_s2s_list.append(x_s2s); test_y_s2s_list.append(y_s2s)
        test_x_m2o, test_y_m2o = np.concatenate(test_x_m2o_list), np.concatenate(test_y_m2o_list)
        test_x_s2s, test_y_s2s = np.concatenate(test_x_s2s_list), np.concatenate(test_y_s2s_list)

        print("正在创建独立的归一化Scaler...")
        scaler_x_m2o = StandardScaler().fit(train_x_m2o.reshape(-1, INPUT_FEATURES))
        scaler_y_m2o = StandardScaler().fit(train_y_m2o.reshape(-1, OUTPUT_FEATURES))
        scaler_x_s2s = StandardScaler().fit(train_x_s2s.reshape(-1, INPUT_FEATURES))
        scaler_y_s2s = StandardScaler().fit(train_y_s2s.reshape(-1, OUTPUT_FEATURES))
        print("Scaler创建完成。")
        
        naive_predictions = test_x_m2o[:, -1, :NUM_SUBCARRIERS]
        naive_targets = test_y_m2o
        
        for model_name in models_to_run:
            print(f"\n--- 正在处理: {model_name} ---")

            if model_name == 'Naive Outdated':
                mse = np.mean((naive_predictions - naive_targets)**2)
                print(f" -> Naive Outdated 基准的 MSE: {mse:.6f}")
                performance_df.loc[horizon, model_name] = mse
                performance_df.to_csv(results_file)
                continue

            if model_name == 'Transformer':
                scaler_x, scaler_y = scaler_x_s2s, scaler_y_s2s
                train_x_norm = scaler_x.transform(train_x_s2s.reshape(-1, INPUT_FEATURES)).reshape(train_x_s2s.shape)
                train_y_norm = scaler_y.transform(train_y_s2s.reshape(-1, OUTPUT_FEATURES)).reshape(train_y_s2s.shape)
                test_x_norm = scaler_x.transform(test_x_s2s.reshape(-1, INPUT_FEATURES)).reshape(test_x_s2s.shape)
                test_y_norm = scaler_y.transform(test_y_s2s.reshape(-1, OUTPUT_FEATURES)).reshape(test_y_s2s.shape)
                train_dataset = TensorDataset(torch.from_numpy(train_x_norm), torch.from_numpy(train_y_norm), torch.from_numpy(train_y_s2s))
                test_dataset = TensorDataset(torch.from_numpy(test_x_norm), torch.from_numpy(test_y_norm), torch.from_numpy(test_y_s2s))
                model = TimeSeriesTransformer(
                    input_size=INPUT_FEATURES, output_size=OUTPUT_FEATURES, dim_val=512, 
                    n_encoder_layers=2, n_decoder_layers=2, n_heads=8,
                    dropout_encoder=0.1, dropout_decoder=0.1, dropout_pos_enc=0.1,
                    dim_feedforward_encoder=512, dim_feedforward_decoder=512,
                    max_seq_len=lookback, dec_seq_len=lookback, out_seq_len=horizon
                )

            elif model_name in ['GRU', 'LSTM']:
                scaler_x, scaler_y = scaler_x_s2s, scaler_y_s2s
                train_x_norm = scaler_x.transform(train_x_s2s.reshape(-1, INPUT_FEATURES)).reshape(train_x_s2s.shape)
                train_y_norm = scaler_y.transform(train_y_s2s.reshape(-1, OUTPUT_FEATURES)).reshape(train_y_s2s.shape)
                test_x_norm = scaler_x.transform(test_x_s2s.reshape(-1, INPUT_FEATURES)).reshape(test_x_s2s.shape)
                test_y_norm = scaler_y.transform(test_y_s2s.reshape(-1, OUTPUT_FEATURES)).reshape(test_y_s2s.shape)
                train_dataset = TensorDataset(torch.from_numpy(train_x_norm), torch.from_numpy(train_y_norm), torch.from_numpy(train_y_s2s))
                test_dataset = TensorDataset(torch.from_numpy(test_x_norm), torch.from_numpy(test_y_norm), torch.from_numpy(test_y_s2s))
                if model_name == 'GRU':
                    model = GRUNet_Seq2Seq(input_dim=INPUT_FEATURES, hidden_dim=512, output_dim=OUTPUT_FEATURES, n_layers=2, drop_prob=0.2)
                else:
                    model = LSTMNet_Seq2Seq(input_dim=INPUT_FEATURES, hidden_dim=256, output_dim=OUTPUT_FEATURES, n_layers=2)

            else:
                scaler_x, scaler_y = scaler_x_m2o, scaler_y_m2o
                train_x_norm = scaler_x.transform(train_x_m2o.reshape(-1, INPUT_FEATURES)).reshape(train_x_m2o.shape)
                train_y_norm = scaler_y.transform(train_y_m2o.reshape(-1, OUTPUT_FEATURES)).reshape(train_y_m2o.shape)
                test_x_norm = scaler_x.transform(test_x_m2o.reshape(-1, INPUT_FEATURES)).reshape(test_x_m2o.shape)
                test_y_norm = scaler_y.transform(test_y_m2o.reshape(-1, OUTPUT_FEATURES)).reshape(test_y_m2o.shape)
                train_dataset = TensorDataset(torch.from_numpy(train_x_norm), torch.from_numpy(train_y_norm), torch.from_numpy(train_y_m2o))
                test_dataset = TensorDataset(torch.from_numpy(test_x_norm), torch.from_numpy(test_y_norm), torch.from_numpy(test_y_m2o))
                if model_name == 'CNN': 
                    model = CNNNet(lookback, input_channels=INPUT_FEATURES, output_channels=OUTPUT_FEATURES)
                elif model_name == 'MLP': 
                    model = MLPNet(lookback, input_channels=INPUT_FEATURES, output_channels=OUTPUT_FEATURES)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            scalers_to_save = {'x_m2o': scaler_x_m2o, 'y_m2o': scaler_y_m2o, 'x_s2s': scaler_x_s2s, 'y_s2s': scaler_y_s2s}
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[model_name], weight_decay=1e-4)
            criterion = nn.MSELoss()
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

            training_loss, validation_loss = epoch_loop(
                model, train_loader, test_loader, epochs, lr_scheduler, 
                criterion, optimizer, device, scalers_to_save, model_name, results_dir, horizon, OUTPUT_FEATURES
            )

            if len(validation_loss) > 0:
                final_loss = min(validation_loss)
                performance_df.loc[horizon, model_name] = final_loss
                print(f" -> {model_name} 完成训练。最佳验证 MSE: {final_loss:.6f}")

            print(f" -> 正在计算 {model_name} 的推理延迟和性能...")
            scaler_y_for_pred = scaler_y_m2o if model_name in ['CNN', 'MLP'] else scaler_y_s2s
            preds, total_pred_time = predict(model, test_loader, device, scaler_y_for_pred, horizon, OUTPUT_FEATURES)

            num_samples = len(test_loader.dataset)
            avg_time = total_pred_time / num_samples
            print(f" -> 在 {num_samples} 个测试样本上总耗时: {total_pred_time:.4f} 秒")
            print(f" -> 平均单步预测时间: {avg_time * 1000:.4f} ms / {avg_time * 1e6:.2f} µs")

            if model_name in ['GRU', 'LSTM', 'Transformer']:
                true_vals = test_y_s2s[:, horizon-1, :]
            else:
                true_vals = test_y_m2o
            
            plot_predictions(true_vals, preds, model_name, naive_predictions, num_timesteps=150, subcarrier_idx=0)
            performance_df.to_csv(results_file)
            plot_losses(training_loss, validation_loss, model_name)

    print("\n--- 5. 实验完成 ---")
    print("最终性能对比 (Magnitude MSE):")
    print(performance_df)

if __name__ == '__main__':
    main()