# -*- coding: utf-8 -*-
"""
DNN Channel Prediction (Magnitude Version): A comprehensive script for training
and evaluating various deep learning models for predicting wireless channel magnitude.

This "Final Showdown" version pits each model architecture in its strongest
configuration against the others for a definitive performance comparison
on the magnitude prediction task.
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler # 引入StandardScaler
import matplotlib.pyplot as plt
import joblib
import random

# ==============================================================================
# 1. 模型定义 (Model Definitions)
# ==============================================================================

class PositionalEncoder(nn.Module):
    def __init__(self, dropout: float=0.1, max_seq_len: int=5000, d_model: int=512, batch_first: bool=True):
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
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_val, nhead=n_heads, dim_feedforward=dim_feedforward_encoder, dropout=dropout_encoder, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_encoder_layers, norm=None)
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_val, nhead=n_heads, dim_feedforward=dim_feedforward_decoder, dropout=dropout_decoder, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_decoder_layers, norm=None)
        self.linear_mapping = nn.Linear(in_features=dim_val, out_features=input_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        src = self.encoder(src=src)
        decoder_output = self.decoder_input_layer(tgt)
        decoder_output = self.decoder(tgt=decoder_output, memory=src, tgt_mask=tgt_mask, memory_mask=src_mask)
        decoder_output = self.linear_mapping(decoder_output)
        return decoder_output[:, -self.out_seq_len:, :]

class GRUNet_Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet_Seq2Seq, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(out)
        return out, h

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

class LSTMNet_Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(LSTMNet_Seq2Seq, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(out)
        return out, h

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

class CNNNet(nn.Module):
    def __init__(self, lookback, input_channels):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(128 * math.floor((lookback-1)/2), 128)
        self.fc2 = nn.Linear(128, input_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLPNet(nn.Module):
    def __init__(self, lookback, input_channels):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(lookback * input_channels, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, input_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==============================================================================
# 2. 数据处理函数 (Data Processing Functions)
# ==============================================================================

def generate_samples_from_block_m2o(block_2d, lookback, horizon, step):
    num_timesteps, num_features = block_2d.shape
    X, y = [], []
    required_len = (lookback - 1) * step + horizon
    num_samples = num_timesteps - required_len
    if num_samples <= 0: return np.array([]), np.array([])
    for i in range(num_samples):
        input_end_idx = i + (lookback - 1) * step
        label_idx = input_end_idx + horizon
        indices = np.arange(i, input_end_idx + 1, step)
        X.append(block_2d[indices])
        y.append(block_2d[label_idx])
    return np.array(X), np.array(y)

def generate_samples_from_block_s2s(block_2d, lookback, horizon, step):
    num_timesteps, num_features = block_2d.shape
    X, y = [], []
    required_len = (lookback - 1) * step + horizon
    num_samples = num_timesteps - required_len
    if num_samples <= 0: return np.array([]), np.array([])
    for i in range(num_samples):
        input_end_idx = i + (lookback - 1) * step
        indices_in = np.arange(i, input_end_idx + 1, step)
        indices_out = np.arange(i + 1, input_end_idx + horizon + 1, step)
        X.append(block_2d[indices_in])
        y.append(block_2d[indices_out])
    return np.array(X), np.array(y)

def generate_square_subsequent_mask(dim1, dim2):
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

# ==============================================================================
# 3. 训练与评估函数 (Training & Evaluation Functions)
# ==============================================================================

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for x, y, _ in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if isinstance(model, (GRUNet_Seq2Seq, LSTMNet_Seq2Seq)):
            h = model.init_hidden(x.size(0), device)
            out, h = model(x.float(), h)
        elif isinstance(model, TimeSeriesTransformer):
            teacher_forcing_ratio = 0.8
            if random.random() < teacher_forcing_ratio:
                tgt_mask = generate_square_subsequent_mask(y.size(1), y.size(1)).to(device)
                out = model(x.float(), y.float(), tgt_mask=tgt_mask)
            else:
                #decoder_input = x[:, -1:, :].repeat(1, y.size(1), 1)
                #tgt_mask = generate_square_subsequent_mask(decoder_input.size(1), decoder_input.size(1)).to(device)
                #out = model(x.float(), decoder_input.float(), tgt_mask=tgt_mask)
                decoder_input = x[:, -1:, :].repeat(1, model.out_seq_len, 1) # model.out_seq_len 就是 horizon
                tgt_mask = generate_square_subsequent_mask(model.out_seq_len, model.out_seq_len).to(device)
                out = model(x.float(), decoder_input.float(), tgt_mask=tgt_mask)
        else:
            out = model(x.float())

        if out.dim() == 3:
            #y_for_loss = y[:, -out.size(1):, :]
            y_for_loss = y[:, :out.size(1), :]
        else:
            y_for_loss = y

        loss = criterion(out, y_for_loss.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def evaluate(model, test_loader, device, scaler, horizon):
    model.eval()
    total_loss_real = 0
    with torch.no_grad():
        for i, (x, y, y_clean) in enumerate(test_loader):
            x, y, y_clean = x.to(device), y.to(device), y_clean.to(device)

            # Use autoregressive prediction for evaluation to avoid data leakage
            if isinstance(model, (GRUNet_Seq2Seq, LSTMNet_Seq2Seq)):
                h = model.init_hidden(x.size(0), device)
                out, h = model(x.float(), h)
            #elif isinstance(model, TimeSeriesTransformer):
            #    decoder_input = x[:, -1:, :].repeat(1, y.size(1), 1)
            #    tgt_mask = generate_square_subsequent_mask(decoder_input.size(1), decoder_input.size(1)).to(device)
            #    out = model(x.float(), decoder_input.float(), tgt_mask=tgt_mask)
            elif isinstance(model, TimeSeriesTransformer):
                 # 解码器的输入序列长度应该是 horizon
                decoder_input = x[:, -1:, :].repeat(1, horizon, 1) 
                # 掩码的大小也应该是 horizon x horizon
                tgt_mask = generate_square_subsequent_mask(horizon, horizon).to(device)
                out = model(x.float(), decoder_input.float(), tgt_mask=tgt_mask)
            else:
                out = model(x.float())

            out_denorm = scaler.inverse_transform(out.cpu().numpy().reshape(-1, out.shape[-1])).reshape(out.shape)
            y_clean_denorm = y_clean.cpu().numpy()

            if out_denorm.ndim == 3:
            # 预测值是输出序列的最后一个点 (对于Transformer是第horizon个点)
            # 真实值应该是目标序列的第 (horizon-1) 个索引
                loss_real = np.mean((out_denorm[:, -1, :] - y_clean_denorm[:, horizon-1, :])**2)
            else:
                loss_real = np.mean((out_denorm - y_clean_denorm)**2)

            total_loss_real += loss_real * x.size(0)
    return total_loss_real / len(test_loader.dataset)


def predict(model, test_loader, device, scaler,horizon):
    model.eval()
    all_predictions = []
    start_time = time.time()
    with torch.no_grad():
        for x, y, _ in test_loader:
            x, y = x.to(device), y.to(device)

            if isinstance(model, (GRUNet_Seq2Seq, LSTMNet_Seq2Seq)):
                h = model.init_hidden(x.size(0), device)
                out, h = model(x.float(), h)
            #elif isinstance(model, TimeSeriesTransformer):
            #    decoder_input = x[:, -1:, :].repeat(1, y.size(1), 1)
            #    tgt_mask = generate_square_subsequent_mask(decoder_input.size(1), decoder_input.size(1)).to(device)
            #    out = model(x.float(), decoder_input.float(), tgt_mask=tgt_mask)
            elif isinstance(model, TimeSeriesTransformer):
                 # 解码器的输入序列长度应该是 horizon
                decoder_input = x[:, -1:, :].repeat(1, horizon, 1) 
                # 掩码的大小也应该是 horizon x horizon
                tgt_mask = generate_square_subsequent_mask(horizon, horizon).to(device)
                out = model(x.float(), decoder_input.float(), tgt_mask=tgt_mask)
            else:
                out = model(x.float())

            out_denorm = scaler.inverse_transform(out.cpu().numpy().reshape(-1, out.shape[-1])).reshape(out.shape)

            if out_denorm.ndim == 3:
                all_predictions.append(out_denorm[:, -1, :])
            else:
                all_predictions.append(out_denorm)

    end_time = time.time()
    total_time = end_time - start_time

    return np.concatenate(all_predictions, axis=0), total_time


def epoch_loop(model, train_loader, test_loader, epochs, lr_scheduler, criterion, optimizer, device, scaler, model_name, results_dir, horizon):
    training_loss, validation_loss = [], []
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 10

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, device, scaler,horizon)

        training_loss.append(train_loss)
        validation_loss.append(test_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | Train Loss (Norm.): {train_loss:.8f} | Test Loss (Real): {test_loss:.6f} | LR: {current_lr:.6f} | Time: {time.time()-start_time:.2f}s")

        if test_loss < best_loss:
            best_loss = test_loss
            epochs_no_improve = 0

            # Only save the best Transformer model, as it's the focus
            if model_name == 'Transformer':
                model_save_path = os.path.join(results_dir, f'{model_name}_best_model_magnitude.pth')
                scaler_save_path = os.path.join(results_dir, f'{model_name}_scaler_magnitude.gz')

                torch.save(model.state_dict(), model_save_path)
                joblib.dump(scaler, scaler_save_path)

                print(f" -> Validation loss improved to {best_loss:.6f}. Saving Transformer model to {model_save_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break

        lr_scheduler.step(test_loss)

    return np.array(training_loss), np.array(validation_loss)


def plot_losses(training_loss, validation_loss, model_name):
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


def plot_predictions(true_vals, preds, model_name, naive_preds, num_timesteps=150, subcarrier_idx=0):
    """绘制真实值与预测值的对比图 (只针对幅度)。"""
    plt.figure(figsize=(15, 7))

    plt.plot(true_vals[:num_timesteps, subcarrier_idx], label='真实幅度 (True Magnitude)', color='blue', linewidth=2)
    plt.plot(preds[:num_timesteps, subcarrier_idx], label=f'{model_name} 预测幅度', color='red', linestyle='--')
    plt.plot(naive_preds[:num_timesteps, subcarrier_idx], label='Naive 预测幅度', color='green', linestyle=':')
    plt.title(f'子载波 #{subcarrier_idx} 幅度预测对比 (前 {num_timesteps} 个时间步)')
    plt.xlabel('时间步 (Time Step)')
    plt.ylabel('信道系数幅度')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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

    print(f"数据文件路径: {combined_data_file}")
    print(f"结果文件路径: {results_file}")

    NUM_ACTIVE_SUBCARRIERS = 576
    horizons = [1]
    lookback = 5
    ts_step = 1
    epochs = 500
    batch_size = 512
    learning_rates = {'GRU': 0.0005, 'LSTM': 0.0005, 'CNN': 0.0005, 'MLP': 0.0005, 'Transformer': 0.0005}

    torch.manual_seed(42); np.random.seed(42)

    # --- 4.2. 设备设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备已设置为: {device}")

    # --- 4.3. 数据加载与分割 ---
    print("\n--- 2. 加载并处理合并后的数据 ---")
    mat = scipy.io.loadmat(combined_data_file)
    combined_1d_array = mat['H'].flatten().astype(np.float32)

    # ### 修改点 1: 更新特征维度，只使用幅度 ###
    total_features = NUM_ACTIVE_SUBCARRIERS

    # ### 修改点 2: 将复数数据转换为幅度数据 ###
    print("正在将复数信道数据转换为幅度...")
    # 1. Reshape to (timesteps, subcarriers, 2) to separate real/imag parts
    complex_data = combined_1d_array.reshape(-1, NUM_ACTIVE_SUBCARRIERS, 2)
    # 2. Calculate magnitude: sqrt(real^2 + imag^2)
    real_part = complex_data[:, :, 0]
    imag_part = complex_data[:, :, 1]
    H_2d_matrix = np.sqrt(real_part**2 + imag_part**2)
    print("数据转换完成。")

    is_zero_row = ~H_2d_matrix.any(axis=1)
    block_change_indices = np.where(np.diff(is_zero_row))[0] + 1
    data_blocks_2d = np.split(H_2d_matrix, block_change_indices)
    continuous_blocks = [block for block in data_blocks_2d if not np.all(block == 0)]
    print(f"成功恢复出 {len(continuous_blocks)} 个独立的连续数据块 (幅度数据)。")

    # --- 4.4. 实验循环 ---
    print("\n--- 3. 开始实验循环 ---")
    performance_df = pd.DataFrame()
    if os.path.exists(results_file):
        performance_df = pd.read_csv(results_file, index_col=0)

    # ### 修改点 3: 恢复所有模型进行训练 ###
    #models_to_run = ['Naive Outdated', 'GRU', 'LSTM', 'CNN', 'MLP', 'Transformer']
    models_to_run = ['Naive Outdated', 'GRU', 'Transformer']

    for horizon in horizons:
        print(f"\n{'='*25} 正在处理 Horizon = {horizon} {'='*25}")

        split_idx = int(len(continuous_blocks) * 0.8)
        train_blocks, test_blocks = continuous_blocks[:split_idx], continuous_blocks[split_idx:]

        print("正在为 '多对一' 和 '序列到序列' 任务准备数据...")

        train_x_m2o_list, train_y_m2o_list = [], []
        for block in train_blocks:
            x, y = generate_samples_from_block_m2o(block, lookback, horizon, ts_step)
            if x.size > 0: train_x_m2o_list.append(x); train_y_m2o_list.append(y)
        train_x_m2o, train_y_m2o = np.concatenate(train_x_m2o_list), np.concatenate(train_y_m2o_list)

        test_x_m2o_list, test_y_m2o_list = [], []
        for block in test_blocks:
            x, y = generate_samples_from_block_m2o(block, lookback, horizon, ts_step)
            if x.size > 0: test_x_m2o_list.append(x); test_y_m2o_list.append(y)
        test_x_m2o, test_y_m2o = np.concatenate(test_x_m2o_list), np.concatenate(test_y_m2o_list)

        train_x_s2s_list, train_y_s2s_list = [], []
        for block in train_blocks:
            x, y = generate_samples_from_block_s2s(block, lookback, horizon, ts_step)
            if x.size > 0: train_x_s2s_list.append(x); train_y_s2s_list.append(y)
        train_x_s2s, train_y_s2s = np.concatenate(train_x_s2s_list), np.concatenate(train_y_s2s_list)

        test_x_s2s_list, test_y_s2s_list = [], []
        for block in test_blocks:
            x, y = generate_samples_from_block_s2s(block, lookback, horizon, ts_step)
            if x.size > 0: test_x_s2s_list.append(x); test_y_s2s_list.append(y)
        test_x_s2s, test_y_s2s = np.concatenate(test_x_s2s_list), np.concatenate(test_y_s2s_list)

        #scaler_m2o = MinMaxScaler().fit(train_x_m2o.reshape(-1, total_features))
        #scaler_s2s = MinMaxScaler().fit(train_x_s2s.reshape(-1, total_features))
        scaler_m2o = StandardScaler().fit(train_x_m2o.reshape(-1, total_features))
        scaler_s2s = StandardScaler().fit(train_x_s2s.reshape(-1, total_features))

        print("数据准备完成。")

        for model_name in models_to_run:
            print(f"\n--- 正在处理: {model_name} ---")

            if model_name == 'Naive Outdated':
                start_time = time.time()
                predictions = test_x_m2o[:, -1, :]
                end_time = time.time()
                total_time = end_time - start_time

                targets = test_y_m2o
                mse = np.mean((predictions - targets)**2)
                num_samples = len(test_x_m2o)
                avg_time = total_time / num_samples

                print(f" -> Naive Outdated 基准的 MSE: {mse:.6f}")
                print(f" -> 在 {num_samples} 个测试样本上总耗时: {total_time:.4f} 秒")
                print(f" -> 平均单步预测时间: {avg_time * 1000:.4f} ms / {avg_time * 1e6:.2f} µs")

                performance_df.loc[horizon, model_name] = mse
                performance_df.to_csv(results_file)
                continue

            # ### 修改点 4: 恢复所有模型的训练和评估逻辑 ###
            if model_name == 'Transformer':
                train_x_norm = scaler_s2s.transform(train_x_s2s.reshape(-1, total_features)).reshape(train_x_s2s.shape)
                train_y_norm = scaler_s2s.transform(train_y_s2s.reshape(-1, total_features)).reshape(train_y_s2s.shape)
                test_x_norm = scaler_s2s.transform(test_x_s2s.reshape(-1, total_features)).reshape(test_x_s2s.shape)
                test_y_norm = scaler_s2s.transform(test_y_s2s.reshape(-1, total_features)).reshape(test_y_s2s.shape)

                train_dataset = TensorDataset(torch.from_numpy(train_x_norm), torch.from_numpy(train_y_norm), torch.from_numpy(train_y_s2s))
                test_dataset = TensorDataset(torch.from_numpy(test_x_norm), torch.from_numpy(test_y_norm), torch.from_numpy(test_y_s2s))
                scaler = scaler_s2s

                model = TimeSeriesTransformer(input_size=total_features, dim_val=128, n_encoder_layers=2, n_decoder_layers=2, n_heads=8,
                                              dropout_encoder=0.1, dropout_decoder=0.1, dropout_pos_enc=0.1,
                                              dim_feedforward_encoder=512, dim_feedforward_decoder=512,
                                              max_seq_len=lookback, dec_seq_len=lookback, out_seq_len=horizon)

            elif model_name in ['GRU', 'LSTM']:
                print(f" -> {model_name} 正在使用 Sequence-to-Sequence 策略...")
                train_x_norm = scaler_s2s.transform(train_x_s2s.reshape(-1, total_features)).reshape(train_x_s2s.shape)
                train_y_norm = scaler_s2s.transform(train_y_s2s.reshape(-1, total_features)).reshape(train_y_s2s.shape)
                test_x_norm = scaler_s2s.transform(test_x_s2s.reshape(-1, total_features)).reshape(test_x_s2s.shape)
                test_y_norm = scaler_s2s.transform(test_y_s2s.reshape(-1, total_features)).reshape(test_y_s2s.shape)

                train_dataset = TensorDataset(torch.from_numpy(train_x_norm), torch.from_numpy(train_y_norm), torch.from_numpy(train_y_s2s))
                test_dataset = TensorDataset(torch.from_numpy(test_x_norm), torch.from_numpy(test_y_norm), torch.from_numpy(test_y_s2s))
                scaler = scaler_s2s

                if model_name == 'GRU':
                    model = GRUNet_Seq2Seq(input_dim=total_features, hidden_dim=128, output_dim=total_features, n_layers=2)
                else: # LSTM
                    model = LSTMNet_Seq2Seq(input_dim=total_features, hidden_dim=256, output_dim=total_features, n_layers=2)

            else: # CNN, MLP
                print(f" -> {model_name} 正在使用 Many-to-One 策略...")
                train_x_norm = scaler_m2o.transform(train_x_m2o.reshape(-1, total_features)).reshape(train_x_m2o.shape)
                train_y_norm = scaler_m2o.transform(train_y_m2o.reshape(-1, total_features)).reshape(train_y_m2o.shape)
                test_x_norm = scaler_m2o.transform(test_x_m2o.reshape(-1, total_features)).reshape(test_x_m2o.shape)
                test_y_norm = scaler_m2o.transform(test_y_m2o.reshape(-1, total_features)).reshape(test_y_m2o.shape)

                train_dataset = TensorDataset(torch.from_numpy(train_x_norm), torch.from_numpy(train_y_norm), torch.from_numpy(train_y_m2o))
                test_dataset = TensorDataset(torch.from_numpy(test_x_norm), torch.from_numpy(test_y_norm), torch.from_numpy(test_y_m2o))
                scaler = scaler_m2o

                if model_name == 'CNN': model = CNNNet(lookback, input_channels=total_features)
                elif model_name == 'MLP': model = MLPNet(lookback, input_channels=total_features)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[model_name])
            criterion = nn.MSELoss()
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

            training_loss, validation_loss = epoch_loop(model, train_loader, test_loader, epochs, lr_scheduler, criterion, optimizer, device, scaler, model_name, results_dir,horizon)

            if len(validation_loss) > 0:
                final_loss = min(validation_loss)
                performance_df.loc[horizon, model_name] = final_loss
                print(f" -> {model_name} 完成训练。最佳验证 MSE: {final_loss:.6f}")


            print(f" -> 正在计算 {model_name} 的推理延迟和性能...")
            preds, total_pred_time = predict(model, test_loader, device, scaler, horizon)

            num_samples = len(test_loader.dataset)
            avg_time = total_pred_time / num_samples

            print(f" -> 在 {num_samples} 个测试样本上总耗时: {total_pred_time:.4f} 秒")
            print(f" -> 平均单步预测时间: {avg_time * 1000:.4f} ms / {avg_time * 1e6:.2f} µs")

            # 使用正确的 target 数据进行绘图 (m2o)
            #naive_preds = test_x_m2o[:, -1, :]
            #true_vals = test_y_m2o
            #plot_predictions(true_vals, preds, model_name, naive_preds)
            # --- 修改建议 ---
            naive_preds = test_x_m2o[:, -1, :]

            # 根据模型类型，选择正确的真实值进行绘图和最终比较
            if model_name in ['GRU', 'LSTM', 'Transformer']:
            # 对于S2S模型，真实值是S2S目标序列的最后一个点
                #true_vals = test_y_s2s[:, -1, :]
                # 对于S2S模型，真实值是S2S目标序列中对应horizon的点
                true_vals = test_y_s2s[:, horizon-1, :] # <--- 修正后的代码
            else:
            # 对于M2O模型，真实值就是M2O的目标
                true_vals = test_y_m2o

            plot_predictions(true_vals, preds, model_name, naive_preds)

            performance_df.to_csv(results_file)
            plot_losses(training_loss, validation_loss, model_name)

    print("\n--- 5. 实验完成 ---")
    print("最终性能对比 (Magnitude MSE):")
    print(performance_df)

if __name__ == '__main__':
    main()