# -*- coding: utf-8 -*-
"""
DNN Channel Prediction (Magnitude Version): A comprehensive script for training
and evaluating various deep learning models for predicting wireless channel magnitude.

This "Final Showdown" version pits each model architecture in its strongest
configuration against the others for a definitive performance comparison
on the magnitude prediction task.

版本说明 (快速训练版):
应用户要求，此版本恢复了将所有数据预先加载到内存的策略，以牺牲内存效率为代价，
换取更快的训练迭代速度。适用于数据集大小在内存容量可接受范围内的场景。
保留了所有关键的训练和评估逻辑修复。

[模型增强 V2]：进一步调整超参数，包括为Transformer设置更小的学习率，
并增加学习率调度器和早停的“耐心”，给予增强后的模型更充分的时间进行收敛，
以在“教师强制”训练模式下取得更优的性能。
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
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import random
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
    def __init__(self, input_size: int, dim_val: int, n_encoder_layers: int, n_decoder_layers: int,
                 n_heads: int, dropout_encoder: float, dropout_decoder: float, dropout_pos_enc: float,
                 dim_feedforward_encoder: int, dim_feedforward_decoder: int,
                 out_seq_len: int):
        super().__init__()
        self.out_seq_len = out_seq_len
        self.encoder_input_layer = nn.Linear(in_features=input_size, out_features=dim_val)
        self.decoder_input_layer = nn.Linear(in_features=input_size, out_features=dim_val)
        self.positional_encoding_layer = PositionalEncoder(d_model=dim_val, dropout=dropout_pos_enc)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_val, nhead=n_heads,
                                                   dim_feedforward=dim_feedforward_encoder,
                                                   dropout=dropout_encoder, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_encoder_layers, norm=None)
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_val, nhead=n_heads,
                                                   dim_feedforward=dim_feedforward_decoder,
                                                   dropout=dropout_decoder, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_decoder_layers, norm=None)
        self.linear_mapping = nn.Linear(in_features=dim_val, out_features=input_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        src = self.encoder(src=src, mask=src_mask)
        
        decoder_output = self.decoder_input_layer(tgt)
        decoder_output = self.positional_encoding_layer(decoder_output)
        decoder_output = self.decoder(tgt=decoder_output, memory=src, tgt_mask=tgt_mask, memory_mask=None)
        
        decoder_output = self.linear_mapping(decoder_output)
        return decoder_output

class GRUNet_Seq2Seq(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, drop_prob: float = 0.2):
        super().__init__()
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
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)

class LSTMNet_Seq2Seq(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, drop_prob: float = 0.2):
        super().__init__()
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
    def __init__(self, lookback: int, input_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self._to_linear = self._get_conv_output_size(lookback, input_channels)
        
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, input_channels)

    def _get_conv_output_size(self, lookback, input_channels):
        with torch.no_grad():
            dummy_input = torch.rand(1, input_channels, lookback)
            output = self.pool(self.conv1(dummy_input))
            return int(np.prod(output.size()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLPNet(nn.Module):
    def __init__(self, lookback: int, input_channels: int):
        super().__init__()
        self.fc1 = nn.Linear(lookback * input_channels, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, input_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==============================================================================
# 2. 数据处理与加载 (Data Processing & Loading)
# ==============================================================================

def generate_samples_from_block(block_2d: np.ndarray, lookback: int, horizon: int, step: int, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    num_timesteps, _ = block_2d.shape
    X, y = [], []
    required_len = (lookback - 1) * step + horizon
    num_samples = num_timesteps - required_len
    if num_samples <= 0: return np.array([]), np.array([])
    
    for i in range(num_samples):
        input_end_idx = i + (lookback - 1) * step
        indices_in = np.arange(i, input_end_idx + 1, step)
        X.append(block_2d[indices_in])
        
        if mode == 'm2o':
            label_idx = input_end_idx + horizon
            y.append(block_2d[label_idx])
        else:  # s2s
            indices_out = np.arange(i + 1, input_end_idx + horizon + 1, step)
            y.append(block_2d[indices_out])
            
    return np.array(X), np.array(y)

def generate_square_subsequent_mask(dim1: int, dim2: int) -> torch.Tensor:
    return torch.triu(torch.ones(dim1, dim2, dtype=torch.bool), diagonal=1)

# ==============================================================================
# 3. 训练与评估函数 (Training & Evaluation Functions)
# ==============================================================================

def train(model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
          optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device).float(), y.to(device).float()
        optimizer.zero_grad()

        if isinstance(model, (GRUNet_Seq2Seq, LSTMNet_Seq2Seq)):
            h = model.init_hidden(x.size(0), device)
            out, _ = model(x, h)
        elif isinstance(model, TimeSeriesTransformer):
            tgt_mask = generate_square_subsequent_mask(y.size(1), y.size(1)).to(device)
            out = model(x, y, tgt_mask=tgt_mask)
        else:
            out = model(x)
        
        if out.dim() == 3:
            y_target = y[:, :out.size(1), :]
            loss_fn = nn.MSELoss(reduction='none')
            per_element_loss = loss_fn(out, y_target)
            seq_len = out.size(1)
            weights = torch.linspace(0.1, 1.0, seq_len).pow(2).to(device)
            weights = weights.view(1, -1, 1)
            weighted_loss = per_element_loss.mean(dim=2) * weights.squeeze(-1)
            loss = weighted_loss.mean()
        else:
            loss = criterion(out, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def predict_autoregressive(model: nn.Module, x: torch.Tensor, horizon: int, device: torch.device) -> torch.Tensor:
    model.eval()
    if isinstance(model, TimeSeriesTransformer):
        src = model.encoder_input_layer(x)
        src = model.positional_encoding_layer(src)
        memory = model.encoder(src=src)
        y_input = x[:, -1:, :]
        for _ in range(horizon):
            tgt_mask = generate_square_subsequent_mask(y_input.size(1), y_input.size(1)).to(device)
            decoder_input = model.decoder_input_layer(y_input)
            decoder_input = model.positional_encoding_layer(decoder_input)
            out = model.decoder(tgt=decoder_input, memory=memory, tgt_mask=tgt_mask)
            out = model.linear_mapping(out)
            y_input = torch.cat((y_input, out[:, -1:, :]), dim=1)
        return y_input[:, 1:, :]
    else: # GRU/LSTM
        outputs = []
        h = model.init_hidden(x.size(0), device)
        out, h = model(x, h)
        last_output = out[:, -1:, :]
        outputs.append(last_output)
        for _ in range(horizon - 1):
            out, h = model(last_output, h)
            outputs.append(out)
            last_output = out
        return torch.cat(outputs, dim=1)

def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device,
             scaler: StandardScaler, horizon: int) -> float:
    model.eval()
    total_loss_real = 0
    with torch.no_grad():
        for x, y_clean in test_loader:
            x, y_clean = x.to(device).float(), y_clean.to(device).float()

            if isinstance(model, (GRUNet_Seq2Seq, LSTMNet_Seq2Seq, TimeSeriesTransformer)):
                out = predict_autoregressive(model, x, horizon, device)
            else:
                out = model(x)

            out_denorm = scaler.inverse_transform(out.cpu().numpy().reshape(-1, out.shape[-1])).reshape(out.shape)
            y_clean_denorm = y_clean.cpu().numpy()

            if out_denorm.ndim == 3:
                pred_point = out_denorm[:, -1, :]
                true_point = y_clean_denorm[:, -1, :]
            else:
                pred_point = out_denorm
                true_point = y_clean_denorm

            loss_real = np.mean((pred_point - true_point)**2)
            total_loss_real += loss_real * x.size(0)
            
    return total_loss_real / len(test_loader.dataset)

def predict(model: nn.Module, test_loader: DataLoader, device: torch.device,
            scaler: StandardScaler, horizon: int) -> Tuple[np.ndarray, float]:
    model.eval()
    all_predictions = []
    start_time = time.time()
    
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device).float()

            if isinstance(model, (GRUNet_Seq2Seq, LSTMNet_Seq2Seq, TimeSeriesTransformer)):
                out = predict_autoregressive(model, x, horizon, device)
            else:
                out = model(x)
                
            out_denorm = scaler.inverse_transform(out.cpu().numpy().reshape(-1, out.shape[-1])).reshape(out.shape)
            
            if out_denorm.ndim == 3:
                all_predictions.append(out_denorm[:, -1, :])
            else:
                all_predictions.append(out_denorm)

    end_time = time.time()
    total_time = end_time - start_time
    return np.concatenate(all_predictions, axis=0), total_time

def epoch_loop(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
               epochs: int, lr_scheduler, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device,
               scaler: StandardScaler, model_name: str, results_dir: str, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    training_loss, validation_loss = [], []
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 20 # [超参数调整] 增加早停耐心

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, device, scaler, horizon)

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
            scaler_save_path = os.path.join(results_dir, f'{model_name}_scaler_magnitude.gz')
            torch.save(model.state_dict(), model_save_path)
            joblib.dump(scaler, scaler_save_path)
            print(f" -> Validation loss improved to {best_loss:.6f}. Saving {model_name} model.")
        else:
            epochs_no_improve += 1

        if current_lr < 1e-7:
            print("Learning rate too small, stopping training.")
            break
            
        lr_scheduler.step(test_loss)
        
        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break

    return np.array(training_loss), np.array(validation_loss)

def plot_losses(training_loss: np.ndarray, validation_loss: np.ndarray, model_name: str):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(training_loss, label='Training Loss (Normalized)', color='blue')
    ax1.set_xlabel('Epochs'); ax1.set_ylabel('Normalized MSE Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2 = ax1.twinx()
    ax2.plot(validation_loss, label='Validation Loss (Real Scale)', color='orange')
    ax2.set_ylabel('Real Scale MSE Loss', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    plt.title(f'Loss Curve for {model_name} (Magnitude Prediction)'); fig.tight_layout()
    lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.grid(True); plt.show()

def plot_predictions(true_vals: np.ndarray, preds: np.ndarray, model_name: str,
                     naive_preds: np.ndarray, num_timesteps: int = 150, subcarrier_idx: int = 0):
    plt.figure(figsize=(15, 7))
    plt.plot(true_vals[:num_timesteps, subcarrier_idx], label='真实幅度 (True Magnitude)', color='blue', linewidth=2)
    plt.plot(preds[:num_timesteps, subcarrier_idx], label=f'{model_name} 预测幅度', color='red', linestyle='--')
    plt.plot(naive_preds[:num_timesteps, subcarrier_idx], label='Naive 预测幅度', color='green', linestyle=':')
    plt.title(f'子载波 #{subcarrier_idx} 幅度预测对比 (前 {num_timesteps} 个时间步)')
    plt.xlabel('时间步 (Time Step)'); plt.ylabel('信道系数幅度')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# ==============================================================================
# 4. 主执行流程 (Main Execution Flow)
# ==============================================================================

def main():
    # --- 4.1. 参数配置 ---
    print("--- 1. 配置实验参数 ---")
    project_root = '.'
    data_dir = os.path.join(project_root, 'data')
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    combined_data_file = os.path.join(data_dir, 'my_H_clean_combined_padded_optimized_test_autopilot_40m.mat')
    results_file = os.path.join(results_dir, 'performance_results_magnitude.csv')
    if not os.path.exists(combined_data_file):
        raise FileNotFoundError(f"数据文件不存在: {combined_data_file}")

    NUM_ACTIVE_SUBCARRIERS = 576
    horizons = [1]
    lookback = 5
    ts_step = 1
    epochs = 500
    batch_size = 512
    # [超参数调整] 为Transformer设置更小的学习率
    learning_rates = {'GRU': 0.0005, 'LSTM': 0.0005, 'CNN': 0.0005, 'MLP': 0.0005, 'Transformer': 0.0001}

    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备已设置为: {device}")

    # --- 4.2. 数据加载与预处理 ---
    print("\n--- 2. 加载并处理数据 ---")
    mat = scipy.io.loadmat(combined_data_file)
    combined_1d_array = mat['H'].flatten().astype(np.float32)

    print("正在将复数信道数据转换为幅度...")
    complex_data = combined_1d_array.reshape(-1, NUM_ACTIVE_SUBCARRIERS, 2)
    H_2d_matrix = np.sqrt(complex_data[:, :, 0]**2 + complex_data[:, :, 1]**2)
    del complex_data, combined_1d_array, mat 
    print("数据转换完成。")

    is_zero_row = ~H_2d_matrix.any(axis=1)
    block_change_indices = np.where(np.diff(is_zero_row))[0] + 1
    data_blocks_2d = np.split(H_2d_matrix, block_change_indices)
    continuous_blocks = [block for block in data_blocks_2d if not np.all(block == 0)]
    del H_2d_matrix, data_blocks_2d 
    print(f"成功恢复出 {len(continuous_blocks)} 个独立的连续数据块。")

    # --- 4.3. 实验循环 ---
    print("\n--- 3. 开始实验循环 ---")
    performance_df = pd.DataFrame()
    if os.path.exists(results_file):
        performance_df = pd.read_csv(results_file, index_col=0)

    #models_to_run = ['Naive Outdated', 'GRU', 'LSTM', 'CNN', 'MLP', 'Transformer']
    models_to_run = ['Naive Outdated', 'Transformer', 'GRU']

    for horizon in horizons:
        print(f"\n{'='*25} 正在处理 Horizon = {horizon} {'='*25}")

        split_idx = int(len(continuous_blocks) * 0.8)
        train_blocks, test_blocks = continuous_blocks[:split_idx], continuous_blocks[split_idx:]

        print("正在生成所有训练和测试样本...")
        
        train_x_m2o_list, train_y_m2o_list = [], []
        for b in train_blocks:
            x, y = generate_samples_from_block(b, lookback, horizon, ts_step, 'm2o')
            if x.size > 0:
                train_x_m2o_list.append(x)
                train_y_m2o_list.append(y)
        
        test_x_m2o_list, test_y_m2o_list = [], []
        for b in test_blocks:
            x, y = generate_samples_from_block(b, lookback, horizon, ts_step, 'm2o')
            if x.size > 0:
                test_x_m2o_list.append(x)
                test_y_m2o_list.append(y)

        train_x_m2o, train_y_m2o = np.concatenate(train_x_m2o_list), np.concatenate(train_y_m2o_list)
        test_x_m2o, test_y_m2o = np.concatenate(test_x_m2o_list), np.concatenate(test_y_m2o_list)

        train_x_s2s_list, train_y_s2s_list = [], []
        for b in train_blocks:
            x, y = generate_samples_from_block(b, lookback, horizon, ts_step, 's2s')
            if x.size > 0:
                train_x_s2s_list.append(x)
                train_y_s2s_list.append(y)
        
        test_x_s2s_list, test_y_s2s_list = [], []
        for b in test_blocks:
            x, y = generate_samples_from_block(b, lookback, horizon, ts_step, 's2s')
            if x.size > 0:
                test_x_s2s_list.append(x)
                test_y_s2s_list.append(y)

        train_x_s2s, train_y_s2s = np.concatenate(train_x_s2s_list), np.concatenate(train_y_s2s_list)
        test_x_s2s, test_y_s2s = np.concatenate(test_x_s2s_list), np.concatenate(test_y_s2s_list)
        
        print("数据生成完毕。")

        print("正在训练归一化模型 (Scaler)...")
        scaler_m2o = StandardScaler().fit(train_x_m2o.reshape(-1, NUM_ACTIVE_SUBCARRIERS))
        scaler_s2s = StandardScaler().fit(train_x_s2s.reshape(-1, NUM_ACTIVE_SUBCARRIERS))
        print("归一化模型训练完成。")

        for model_name in models_to_run:
            print(f"\n--- 正在处理: {model_name} ---")

            if model_name == 'Naive Outdated':
                predictions = test_x_m2o[:, -1, :]
                target = test_y_s2s[:, -1, :]
                mse = np.mean((predictions - target)**2)
                print(f" -> Naive Outdated 基准的 MSE (S2S可比): {mse:.6f}")
                performance_df.loc[horizon, model_name] = mse
                performance_df.to_csv(results_file)
                continue

            if model_name in ['GRU', 'LSTM', 'Transformer']:
                scaler = scaler_s2s
                train_x, train_y = train_x_s2s, train_y_s2s
                test_x, test_y = test_x_s2s, test_y_s2s
                
                # [模型增强] 使用更大容量的模型
                if model_name == 'GRU': 
                    model = GRUNet_Seq2Seq(NUM_ACTIVE_SUBCARRIERS, 256, NUM_ACTIVE_SUBCARRIERS, 3, drop_prob=0.3)
                elif model_name == 'LSTM': 
                    model = LSTMNet_Seq2Seq(NUM_ACTIVE_SUBCARRIERS, 256, NUM_ACTIVE_SUBCARRIERS, 3, drop_prob=0.3)
                else: # Transformer
                    model = TimeSeriesTransformer(NUM_ACTIVE_SUBCARRIERS, 256, 3, 3, 8, 0.2, 0.2, 0.1, 1024, 1024, horizon)
            else: 
                scaler = scaler_m2o
                train_x, train_y = train_x_m2o, train_y_m2o
                test_x, test_y = test_x_m2o, test_y_m2o
                if model_name == 'CNN': model = CNNNet(lookback, NUM_ACTIVE_SUBCARRIERS)
                else: model = MLPNet(lookback, NUM_ACTIVE_SUBCARRIERS)

            # 数据归一化
            train_x_norm = scaler.transform(train_x.reshape(-1, NUM_ACTIVE_SUBCARRIERS)).reshape(train_x.shape)
            train_y_norm = scaler.transform(train_y.reshape(-1, NUM_ACTIVE_SUBCARRIERS)).reshape(train_y.shape)
            test_x_norm = scaler.transform(test_x.reshape(-1, NUM_ACTIVE_SUBCARRIERS)).reshape(test_x.shape)
            
            train_dataset = TensorDataset(torch.from_numpy(train_x_norm), torch.from_numpy(train_y_norm))
            test_dataset = TensorDataset(torch.from_numpy(test_x_norm), torch.from_numpy(test_y))
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates.get(model_name, 0.0005))
            criterion = nn.MSELoss()
            # [超参数调整] 增加调度器耐心
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5) 

            training_loss, validation_loss = epoch_loop(model, train_loader, test_loader, epochs, lr_scheduler, 
                                                        criterion, optimizer, device, scaler, model_name, results_dir, horizon)
            
            if len(validation_loss) > 0:
                final_loss = min(validation_loss)
                performance_df.loc[horizon, model_name] = final_loss
                print(f" -> {model_name} 完成训练。最佳验证 MSE: {final_loss:.6f}")

            print(f" -> 正在计算 {model_name} 的推理延迟和性能...")
            preds, total_pred_time = predict(model, test_loader, device, scaler, horizon)
            
            num_samples = len(test_loader.dataset)
            avg_time = total_pred_time / num_samples
            print(f" -> 在 {num_samples} 个测试样本上总耗时: {total_pred_time:.4f} 秒")
            print(f" -> 平均单步预测时间: {avg_time * 1000:.4f} ms")

            naive_preds = test_x_m2o[:, -1, :]
            if model_name in ['GRU', 'LSTM', 'Transformer']:
                true_vals = test_y_s2s[:, -1, :]
            else:
                true_vals = test_y_m2o
            
            plot_predictions(true_vals, preds, model_name, naive_preds)
            performance_df.to_csv(results_file)
            if len(training_loss) > 0:
                plot_losses(training_loss, validation_loss, model_name)

    print("\n--- 5. 实验完成 ---")
    print("最终性能对比 (Magnitude MSE):")
    print(performance_df)

if __name__ == '__main__':
    main()

