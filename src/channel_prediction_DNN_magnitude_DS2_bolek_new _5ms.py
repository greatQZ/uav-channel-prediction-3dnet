# -*- coding: utf-8 -*-
"""
DNN Channel Prediction (Magnitude Version): A comprehensive script for training
and evaluating various deep learning models for predicting wireless channel magnitude.

v4: 修正了一个在归一化部分的小笔误。代码结构和逻辑经过全面审核。
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
from typing import Tuple, Optional

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
                 max_seq_len: int, dec_seq_len: int, out_seq_len: int):
        super().__init__()
        self.dec_seq_len = dec_seq_len
        self.out_seq_len = out_seq_len
        self.encoder_input_layer = nn.Linear(in_features=input_size, out_features=dim_val)
        self.decoder_input_layer = nn.Linear(in_features=input_size, out_features=dim_val)
        self.positional_encoding_layer = PositionalEncoder(d_model=dim_val, dropout=dropout_pos_enc, max_seq_len=max_seq_len)
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
        src = self.encoder(src=src)
        decoder_output = self.decoder_input_layer(tgt)
        decoder_output = self.positional_encoding_layer(decoder_output)
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
    def __init__(self, lookback: int, input_channels: int):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        conv_output_size = 128 * math.floor((lookback-1)/2)
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, input_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLPNet(nn.Module):
    def __init__(self, lookback: int, input_channels: int):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(lookback * input_channels, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, input_channels)
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

def generate_samples_from_block_m2o(block_2d: np.ndarray, lookback: int, horizon: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    num_timesteps, num_features = block_2d.shape
    X, y = [], []
    required_len = (lookback - 1) * step + horizon
    num_samples = num_timesteps - required_len + 1
    if num_samples <= 0: 
        return np.array([]), np.array([])
    for i in range(num_samples):
        input_end_idx = i + (lookback - 1) * step
        label_idx = input_end_idx + horizon 
        if label_idx >= num_timesteps: break
        indices = np.arange(i, input_end_idx + 1, step)
        X.append(block_2d[indices])
        y.append(block_2d[label_idx])
    return np.array(X), np.array(y)

def generate_samples_from_block_s2s(block_2d: np.ndarray, lookback: int, horizon: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    num_timesteps, num_features = block_2d.shape
    X, y = [], []
    required_len = (lookback - 1) * step + horizon
    num_samples = num_timesteps - required_len + 1
    if num_samples <= 0:
        return np.array([]), np.array([])
    for i in range(num_samples):
        input_end_idx = i + (lookback - 1) * step
        indices_in = np.arange(i, input_end_idx + 1, step)
        output_start_idx = input_end_idx + 1
        output_end_idx = output_start_idx + horizon
        if output_end_idx > num_timesteps: break
        indices_out = np.arange(output_start_idx, output_end_idx)
        X.append(block_2d[indices_in])
        y.append(block_2d[indices_out])
    return np.array(X), np.array(y)

def generate_square_subsequent_mask(dim1: int, dim2: int) -> torch.Tensor:
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

# ==============================================================================
# 3. 训练与评估函数 (Training & Evaluation Functions)
# ==============================================================================

def train(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
          optimizer: torch.optim.Optimizer, device: torch.device, horizon: int) -> float:
    model.train()
    running_loss = 0.0
    for x, y, _ in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        is_s2s = isinstance(model, (GRUNet_Seq2Seq, LSTMNet_Seq2Seq, TimeSeriesTransformer))

        if isinstance(model, (GRUNet_Seq2Seq, LSTMNet_Seq2Seq)):
            h = model.init_hidden(x.size(0), device)
            out, h = model(x.float(), h)
            out = out[:, -horizon:, :]
        elif isinstance(model, TimeSeriesTransformer):
            tgt_in = torch.cat([x[:, -1:, :], y[:, :-1, :]], dim=1)
            tgt_mask = generate_square_subsequent_mask(tgt_in.size(1), tgt_in.size(1)).to(device)
            out = model(x.float(), tgt_in.float(), tgt_mask=tgt_mask)
        else: # MLP, CNN
            out = model(x.float())

        if is_s2s:
            loss = criterion(out, y.float())
        else: # Many-to-One
            loss = criterion(out, y.float().unsqueeze(1).squeeze(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def run_inference(model: nn.Module, loader: DataLoader, device: torch.device, horizon: int) -> np.ndarray:
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            is_s2s = isinstance(model, (GRUNet_Seq2Seq, LSTMNet_Seq2Seq, TimeSeriesTransformer))

            if isinstance(model, (GRUNet_Seq2Seq, LSTMNet_Seq2Seq)):
                h = model.init_hidden(x.size(0), device)
                _, h = model(x.float(), h)
                decoder_input = x[:, -1:, :]
                predictions = []
                for _ in range(horizon):
                    out_step, h = model(decoder_input.float(), h)
                    predictions.append(out_step)
                    decoder_input = out_step
                out = torch.cat(predictions, dim=1)
            elif isinstance(model, TimeSeriesTransformer):
                src = model.encoder_input_layer(x.float())
                src = model.positional_encoding_layer(src)
                memory = model.encoder(src=src)
                decoder_input = x[:, -1:, :].float()
                for _ in range(horizon):
                    tgt_len = decoder_input.size(1)
                    tgt_mask = generate_square_subsequent_mask(tgt_len, tgt_len).to(device)
                    tgt_encoded = model.decoder_input_layer(decoder_input)
                    tgt_with_pe = model.positional_encoding_layer(tgt_encoded)
                    output = model.decoder(tgt=tgt_with_pe, memory=memory, tgt_mask=tgt_mask)
                    output = model.linear_mapping(output)
                    next_step_pred = output[:, -1:, :]
                    decoder_input = torch.cat([decoder_input, next_step_pred], dim=1)
                out = decoder_input[:, 1:, :]
            else: # MLP, CNN
                out = model(x.float())
            
            all_outputs.append(out.cpu().numpy())
            
    return np.concatenate(all_outputs, axis=0)

def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device, 
             scaler: StandardScaler, horizon: int) -> float:
    predictions_norm = run_inference(model, test_loader, device, horizon)
    y_clean_list = [y_clean for _, _, y_clean in test_loader]
    y_clean = np.concatenate(y_clean_list, axis=0)
    is_s2s = isinstance(model, (GRUNet_Seq2Seq, LSTMNet_Seq2Seq, TimeSeriesTransformer))
    if is_s2s:
        pred_shape = predictions_norm.shape
        predictions_denorm = scaler.inverse_transform(predictions_norm.reshape(-1, pred_shape[-1])).reshape(pred_shape)
    else: # Many-to-One
        predictions_denorm = scaler.inverse_transform(predictions_norm)
    loss_real = np.mean((predictions_denorm - y_clean)**2)
    return loss_real

def predict(model: nn.Module, test_loader: DataLoader, device: torch.device, 
            scaler: StandardScaler, horizon: int) -> Tuple[np.ndarray, float]:
    start_time = time.time()
    predictions_norm = run_inference(model, test_loader, device, horizon)
    total_time = time.time() - start_time
    is_s2s = isinstance(model, (GRUNet_Seq2Seq, LSTMNet_Seq2Seq, TimeSeriesTransformer))
    if is_s2s:
        pred_shape = predictions_norm.shape
        predictions_denorm = scaler.inverse_transform(predictions_norm.reshape(-1, pred_shape[-1])).reshape(pred_shape)
        final_preds = predictions_denorm[:, -1, :]
    else: # Many-to-One
        final_preds = scaler.inverse_transform(predictions_norm)
    return final_preds, total_time

def epoch_loop(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, 
               epochs: int, lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau, 
               criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, 
               scaler: StandardScaler, model_name: str, results_dir: str, 
               horizon: int, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    training_loss, validation_loss = [], []
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 20    

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train(model, train_loader, criterion, optimizer, device, horizon)
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
            model_save_path = os.path.join(results_dir, f'{model_name}_lookback{lookback}_horizon{horizon}_best_model_magnitude_bolek_para2_5ms.pth')
            scaler_save_path = os.path.join(results_dir, f'{model_name}_lookback{lookback}_horizon{horizon}_scaler_magnitude_bolek_para2_5ms.gz')
            torch.save(model.state_dict(), model_save_path)
            joblib.dump(scaler, scaler_save_path)
            print(f" -> Validation loss improved to {best_loss:.6f}. Saving {model_name} model.")
        else:
            epochs_no_improve += 1

        if current_lr < 1e-6:
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
                     naive_preds: np.ndarray, num_timesteps: int = 500, subcarrier_idx: int = 0):
    plt.figure(figsize=(15, 7))
    min_length = min(len(true_vals), len(preds), len(naive_preds))
    true_vals = true_vals[:min_length]
    preds = preds[:min_length]
    naive_preds = naive_preds[:min_length]
    time_steps = np.arange(min(min_length, num_timesteps))
    if subcarrier_idx >= true_vals.shape[1]:
        print(f"警告: 子载波索引 {subcarrier_idx} 超出范围，使用第一个子载波")
        subcarrier_idx = 0
    plt.plot(time_steps, true_vals[:num_timesteps, subcarrier_idx], 
             label='True Magnitude', color='blue', linewidth=2)
    plt.plot(time_steps, preds[:num_timesteps, subcarrier_idx], 
             label=f'{model_name} Predicted Magnitude', color='red', linestyle='--')
    plt.plot(time_steps, naive_preds[:num_timesteps, subcarrier_idx], 
             label='Naive Predicted Magnitude', color='green', linestyle=':')
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
    # --- 4.1. 参数配置 ---
    NUM_ACTIVE_SUBCARRIERS = 576
    horizons = [20]
    lookback = 10
    ts_step = 2
    epochs = 500
    batch_size = 512
    learning_rates = {'GRU': 0.000025, 'LSTM': 0.0001, 'CNN': 0.0001, 'MLP': 0.0001, 'Transformer': 0.00001}

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)    
    
    print("--- 1. 配置实验参数 ---")
    project_root = '.'
    data_dir = os.path.join(project_root, 'data')
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    combined_data_file = os.path.join(data_dir, 'my_H_real_data_bolek_for_5ms_40m_10mps_for_training.mat')
    results_file = os.path.join(results_dir, f'performance_results_magnitude_lookback{lookback}_bolek_para2_5ms.csv')
    
    if not os.path.exists(combined_data_file):
        raise FileNotFoundError(f"数据文件不存在: {combined_data_file}")

    print(f"数据文件路径: {combined_data_file}")
    print(f"结果文件路径: {results_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备已设置为: {device}")

    print("\n--- 2. 加载并处理合并后的数据 ---")
    mat = scipy.io.loadmat(combined_data_file)
    combined_1d_array = mat['H'].flatten().astype(np.float32)
    print("正在将复数信道数据转换为幅度...")
    complex_data = combined_1d_array.reshape(-1, NUM_ACTIVE_SUBCARRIERS, 2)
    real_part = complex_data[:, :, 0]
    imag_part = complex_data[:, :, 1]
    H_2d_matrix = np.sqrt(real_part**2 + imag_part**2)
    print("数据转换完成。")

    is_zero_row = ~H_2d_matrix.any(axis=1)
    block_change_indices = np.where(np.diff(is_zero_row))[0] + 1
    data_blocks_2d = np.split(H_2d_matrix, block_change_indices)
    continuous_blocks = [block for block in data_blocks_2d if not np.all(block == 0)]
    print(f"成功恢复出 {len(continuous_blocks)} 个独立的连续数据块 (幅度数据)。")

    print("\n--- 3. 开始实验循环 ---")
    performance_df = pd.DataFrame()
    if os.path.exists(results_file):
        performance_df = pd.read_csv(results_file, index_col=0)

    models_to_run = ['Naive Outdated', 'Transformer']

    for horizon in horizons:
        print(f"\n{'='*25} 正在处理 Horizon = {horizon} {'='*25}")

        all_data = np.concatenate(continuous_blocks, axis=0)
        print(f"所有数据块已合并成一个长度为 {len(all_data)} 的连续序列。")
        split_point = int(len(all_data) * 0.8)
        
        if split_point > 0 and len(all_data) - split_point > 0:
            train_blocks = [all_data[:split_point]]
            test_blocks = [all_data[split_point:]]
            print(f"数据分割结果: {len(train_blocks[0])} 个训练时间步, {len(test_blocks[0])} 个测试时间步。")
        else:
            train_blocks, test_blocks = [], [all_data]
            print("警告: 数据总量过短，无法有效分割，已将所有数据放入测试集。")
        
        print("正在为 '多对一' 和 '序列到序列' 任务准备数据...")
        train_x_m2o_list, train_y_m2o_list = [], []
        train_x_s2s_list, train_y_s2s_list = [], []
        for block in train_blocks:
            x_m2o, y_m2o = generate_samples_from_block_m2o(block, lookback, horizon, ts_step)
            if x_m2o.size > 0: train_x_m2o_list.append(x_m2o); train_y_m2o_list.append(y_m2o)
            x_s2s, y_s2s = generate_samples_from_block_s2s(block, lookback, horizon, ts_step)
            if x_s2s.size > 0: train_x_s2s_list.append(x_s2s); train_y_s2s_list.append(y_s2s)
        
        if not train_x_m2o_list or not train_x_s2s_list:
            print(f"警告: 未能为 horizon={horizon} 和 lookback={lookback} 生成足够的训练样本。将跳过此 horizon 的训练。")
            continue

        train_x_m2o, train_y_m2o = np.concatenate(train_x_m2o_list), np.concatenate(train_y_m2o_list)
        train_x_s2s, train_y_s2s = np.concatenate(train_x_s2s_list), np.concatenate(train_y_s2s_list)

        test_x_m2o_list, test_y_m2o_list = [], []
        test_x_s2s_list, test_y_s2s_list = [], []
        for block in test_blocks:
            x_m2o, y_m2o = generate_samples_from_block_m2o(block, lookback, horizon, ts_step)
            if x_m2o.size > 0: test_x_m2o_list.append(x_m2o); test_y_m2o_list.append(y_m2o)
            x_s2s, y_s2s = generate_samples_from_block_s2s(block, lookback, horizon, ts_step)
            if x_s2s.size > 0: test_x_s2s_list.append(x_s2s); test_y_s2s_list.append(y_s2s)
        
        if not test_x_m2o_list or not test_x_s2s_list:
             print(f"警告: 未能为 horizon={horizon} 和 lookback={lookback} 生成测试样本。跳过此 horizon。")
             continue

        test_x_m2o, test_y_m2o = np.concatenate(test_x_m2o_list), np.concatenate(test_y_m2o_list)
        test_x_s2s, test_y_s2s = np.concatenate(test_x_s2s_list), np.concatenate(test_y_s2s_list)
        
        # 【勘误】修正此处的笔误
        scaler_m2o = StandardScaler().fit(train_x_m2o.reshape(-1, NUM_ACTIVE_SUBCARRIERS))
        scaler_s2s = StandardScaler().fit(train_x_s2s.reshape(-1, NUM_ACTIVE_SUBCARRIERS))
        print("数据准备和归一化完成。")

        naive_predictions = test_x_m2o[:, -1, :]
        naive_targets = test_y_m2o
        
        for model_name in models_to_run:
            print(f"\n--- 正在处理: {model_name} ---")

            if model_name == 'Naive Outdated':
                start_time = time.time()
                mse = np.mean((naive_predictions - naive_targets)**2)
                num_samples = len(test_x_m2o)
                total_time = time.time() - start_time
                avg_time = total_time / num_samples if num_samples > 0 else 0
                print(f" -> Naive Outdated 基准的 MSE: {mse:.6f}")
                print(f" -> 平均单步预测时间: {avg_time * 1e6:.2f} µs")
                performance_df.loc[horizon, model_name] = mse
                performance_df.to_csv(results_file)
                continue

            if model_name in ['GRU', 'LSTM', 'Transformer']:
                train_x, train_y_s2s_clean, test_x, test_y_s2s_clean = train_x_s2s, train_y_s2s, test_x_s2s, test_y_s2s
                scaler = scaler_s2s
            else:
                train_x, train_y_m2o_clean, test_x, test_y_m2o_clean = train_x_m2o, train_y_m2o, test_x_m2o, test_y_m2o
                scaler = scaler_m2o

            train_x_norm = scaler.transform(train_x.reshape(-1, NUM_ACTIVE_SUBCARRIERS)).reshape(train_x.shape)
            test_x_norm = scaler.transform(test_x.reshape(-1, NUM_ACTIVE_SUBCARRIERS)).reshape(test_x.shape)

            if model_name in ['GRU', 'LSTM', 'Transformer']:
                train_y_norm = scaler.transform(train_y_s2s_clean.reshape(-1, NUM_ACTIVE_SUBCARRIERS)).reshape(train_y_s2s_clean.shape)
                train_dataset = TensorDataset(torch.from_numpy(train_x_norm), torch.from_numpy(train_y_norm), torch.from_numpy(train_y_s2s_clean))
                test_dataset = TensorDataset(torch.from_numpy(test_x_norm), torch.zeros_like(torch.from_numpy(test_y_s2s_clean)), torch.from_numpy(test_y_s2s_clean))
            else:
                train_y_norm = scaler.transform(train_y_m2o_clean.reshape(-1, NUM_ACTIVE_SUBCARRIERS)).reshape(train_y_m2o_clean.shape)
                train_dataset = TensorDataset(torch.from_numpy(train_x_norm), torch.from_numpy(train_y_norm), torch.from_numpy(train_y_m2o_clean))
                test_dataset = TensorDataset(torch.from_numpy(test_x_norm), torch.zeros_like(torch.from_numpy(test_y_m2o_clean)), torch.from_numpy(test_y_m2o_clean))

            if model_name == 'Transformer':
                model = TimeSeriesTransformer(input_size=NUM_ACTIVE_SUBCARRIERS, dim_val=512, n_encoder_layers=2, n_decoder_layers=2, n_heads=16,
                                              dropout_encoder=0.1, dropout_decoder=0.1, dropout_pos_enc=0.1,
                                              dim_feedforward_encoder=1024, dim_feedforward_decoder=1024,
                                              max_seq_len=lookback+horizon, dec_seq_len=horizon, out_seq_len=horizon)
            elif model_name == 'GRU': model = GRUNet_Seq2Seq(input_dim=NUM_ACTIVE_SUBCARRIERS, hidden_dim=512, output_dim=NUM_ACTIVE_SUBCARRIERS, n_layers=2)
            elif model_name == 'LSTM': model = LSTMNet_Seq2Seq(input_dim=NUM_ACTIVE_SUBCARRIERS, hidden_dim=512, output_dim=NUM_ACTIVE_SUBCARRIERS, n_layers=2)
            elif model_name == 'CNN': model = CNNNet(lookback, input_channels=NUM_ACTIVE_SUBCARRIERS)
            elif model_name == 'MLP': model = MLPNet(lookback, input_channels=NUM_ACTIVE_SUBCARRIERS)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[model_name], weight_decay=1e-5)
            criterion = nn.MSELoss()
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

            training_loss, validation_loss = epoch_loop(model, train_loader, test_loader, epochs, lr_scheduler, 
                                                        criterion, optimizer, device, scaler, model_name, results_dir, 
                                                        horizon, lookback)

            if len(validation_loss) > 0:
                final_loss = min(validation_loss)
                performance_df.loc[horizon, model_name] = final_loss
                print(f" -> {model_name} 完成训练。最佳验证 MSE: {final_loss:.6f}")
                
            print(f" -> 正在加载最佳模型并进行最终预测...")
            model_path = os.path.join(results_dir, f'{model_name}_lookback{lookback}_horizon{horizon}_best_model_magnitude_bolek_para2_5ms.pth')
            model.load_state_dict(torch.load(model_path))
            
            preds, total_pred_time = predict(model, test_loader, device, scaler, horizon)

            num_samples = len(test_loader.dataset)
            avg_time = total_pred_time / num_samples
            print(f" -> 在 {num_samples} 个测试样本上总耗时: {total_pred_time:.4f} 秒")
            print(f" -> 平均单步预测时间: {avg_time * 1e6:.2f} µs")

            if model_name in ['GRU', 'LSTM', 'Transformer']:
                true_vals = test_y_s2s[:, horizon-1, :]
            else:
                true_vals = test_y_m2o
            
            plot_predictions(true_vals, preds, model_name, naive_predictions, num_timesteps=500, subcarrier_idx=1)
            performance_df.to_csv(results_file)
            plot_losses(training_loss, validation_loss, model_name)

    print("\n--- 5. 实验完成 ---")
    print("最终性能对比 (Magnitude MSE):")
    print(performance_df)

if __name__ == '__main__':
    main()