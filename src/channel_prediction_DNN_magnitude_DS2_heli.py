# -*- coding: utf-8 -*-
"""
DNN Channel Prediction (Magnitude Version): A comprehensive script for training
and evaluating various deep learning models for predicting wireless channel magnitude.

修复了Naive预测向左偏移的问题，确保预测值与真实值正确对齐。
整合了针对单一数据块和样本生成失败的稳健性修复。
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
from torch.utils.data import TensorDataset, DataLoader, IterableDataset
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
    # 修正: required_len 计算的是生成一个样本所需的总时间跨度
    required_len = (lookback - 1) * step + horizon
    # 可生成的样本数
    num_samples = num_timesteps - required_len + 1

    if num_samples <= 0: 
        return np.array([]), np.array([])
    
    for i in range(num_samples):
        # 输入序列的结束点
        input_end_idx = i + (lookback - 1) * step
        # 标签（目标）点
        label_idx = input_end_idx + horizon 
        
        # 确保标签索引在数组范围内
        if label_idx >= num_timesteps:
            break
            
        indices = np.arange(i, input_end_idx + 1, step)
        X.append(block_2d[indices])
        y.append(block_2d[label_idx])
        
    return np.array(X), np.array(y)


def generate_samples_from_block_s2s(block_2d: np.ndarray, lookback: int, horizon: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    num_timesteps, num_features = block_2d.shape
    X, y = [], []
    
    # 修正: required_len 计算的是生成一个样本所需的总时间跨度
    required_len = (lookback - 1) * step + horizon
    # 可生成的样本数
    num_samples = num_timesteps - required_len + 1
    
    if num_samples <= 0:
        return np.array([]), np.array([])
        
    for i in range(num_samples):
        input_end_idx = i + (lookback - 1) * step
        
        # 输入序列的索引
        indices_in = np.arange(i, input_end_idx + 1, step)
        
        # 输出/目标序列的索引 (从输入结束后的下一个点开始，连续取horizon个点)
        output_start_idx = input_end_idx + 1
        output_end_idx = output_start_idx + horizon
        
        # 确保输出序列在数组范围内
        if output_end_idx > num_timesteps:
            break
            
        indices_out = np.arange(output_start_idx, output_end_idx)
        
        X.append(block_2d[indices_in])
        y.append(block_2d[indices_out])
        
    return np.array(X), np.array(y)

def generate_square_subsequent_mask(dim1: int, dim2: int) -> torch.Tensor:
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

# ==============================================================================
# 3. 训练与评估函数 (Training & Evaluation Functions)
# ==============================================================================

@contextlib.contextmanager
def evaluating(model: nn.Module):
    """上下文管理器，用于设置模型为评估模式"""
    model.eval()
    try:
        yield model
    finally:
        model.train()

def train_transformer_autoregressive(model: nn.Module, x: torch.Tensor, horizon: int, device: torch.device) -> torch.Tensor:
    """使用自回归方式进行Transformer的预测（用于训练和评估）"""
    # 编码器处理整个输入序列
    src = model.encoder_input_layer(x)
    src = model.positional_encoding_layer(src)
    memory = model.encoder(src=src)
    
    # 初始化解码器输入（使用最后一个真实输入作为起点）
    decoder_input = x[:, -1:, :]
    outputs = []
    
    # 自回归生成预测序列
    for _ in range(horizon):
        # 每次只解码一步，所以不需要复杂的mask
        tgt = model.decoder_input_layer(decoder_input)
        # 注意：这里的位置编码可能需要更复杂的处理，但为简化，我们每次都重新编码
        tgt = model.positional_encoding_layer(tgt)
        
        output = model.decoder(tgt=tgt, memory=memory) # 在生成模式下，不需要tgt_mask
        output = model.linear_mapping(output)
        
        # 取序列的最后一个时间步作为当前步的预测
        next_input = output[:, -1:, :]
        
        # 将当前步的预测添加到下一步的输入中（如果需要多步依赖）
        # 为简单起见，这里我们只用上一步的预测来预测下一步
        decoder_input = torch.cat([decoder_input, next_input], dim=1)[:, 1:, :]
        
        outputs.append(next_input)
    
    return torch.cat(outputs, dim=1)

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
            # 在Seq2Seq中，输入x的长度就是lookback
            out, h = model(x.float(), h)
            # 我们需要预测horizon步，但模型输出的长度可能与lookback相同
            # 取模型输出的最后horizon个点进行比较
            out = out[:, -horizon:, :]
        elif isinstance(model, TimeSeriesTransformer):
            # Teacher Forcing: 训练时，解码器的输入是真实标签（向右移一位）
            tgt_in = torch.cat([x[:, -1:, :], y[:, :-1, :]], dim=1)
            tgt_mask = generate_square_subsequent_mask(tgt_in.size(1), tgt_in.size(1)).to(device)
            out = model(x.float(), tgt_in.float(), tgt_mask=tgt_mask)
        else: # MLP, CNN (Many-to-One)
            out = model(x.float())

        # 根据模型类型调整损失计算
        if is_s2s:
            loss = criterion(out, y.float())
        else: # Many-to-One
            # y的形状是 (batch, features)，需要扩展维度以匹配out
            loss = criterion(out, y.float().unsqueeze(1).squeeze(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device, 
             scaler: StandardScaler, horizon: int) -> float:
    model.eval()
    total_loss_real = 0
    with torch.no_grad():
        for i, (x, y, y_clean) in enumerate(test_loader):
            x, y, y_clean = x.to(device), y.to(device), y_clean.to(device)
            
            is_s2s = isinstance(model, (GRUNet_Seq2Seq, LSTMNet_Seq2Seq, TimeSeriesTransformer))

            if isinstance(model, (GRUNet_Seq2Seq, LSTMNet_Seq2Seq)):
                # 评估时，我们需要自回归地进行预测
                h = model.init_hidden(x.size(0), device)
                _, h = model(x.float(), h) # 用真实输入序列初始化隐藏状态
                
                # 第一个预测的输入是真实序列的最后一个点
                decoder_input = x[:, -1:, :]
                predictions = []
                for _ in range(horizon):
                    out, h = model(decoder_input.float(), h)
                    predictions.append(out)
                    decoder_input = out # 下一步的输入是上一步的输出
                out = torch.cat(predictions, dim=1)

            elif isinstance(model, TimeSeriesTransformer):
                 out = train_transformer_autoregressive(model, x.float(), horizon, device)

            else: # MLP, CNN
                out = model(x.float())

            # 反归一化
            # CNN/MLP输出 (batch, features), S2S输出 (batch, horizon, features)
            if is_s2s:
                out_shape = out.shape
                out_denorm = scaler.inverse_transform(out.cpu().numpy().reshape(-1, out.shape[-1])).reshape(out_shape)
                y_clean_denorm = y_clean.cpu().numpy() # shape (batch, horizon, features)
                
                # 比较整个预测序列和真实序列
                loss_real = np.mean((out_denorm - y_clean_denorm)**2)

            else: # Many-to-One
                out_denorm = scaler.inverse_transform(out.cpu().numpy())
                y_clean_denorm = y_clean.cpu().numpy() # shape (batch, features)
                loss_real = np.mean((out_denorm - y_clean_denorm)**2)

            total_loss_real += loss_real * x.size(0)
            
    return total_loss_real / len(test_loader.dataset)

def predict(model: nn.Module, test_loader: DataLoader, device: torch.device, 
            scaler: StandardScaler, horizon: int) -> Tuple[np.ndarray, float]:
    model.eval()
    all_predictions = []
    start_time = time.time()
    
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device)
            
            is_s2s = isinstance(model, (GRUNet_Seq2Seq, LSTMNet_Seq2Seq, TimeSeriesTransformer))

            if isinstance(model, (GRUNet_Seq2Seq, LSTMNet_Seq2Seq)):
                h = model.init_hidden(x.size(0), device)
                _, h = model(x.float(), h)
                decoder_input = x[:, -1:, :]
                predictions = []
                for _ in range(horizon):
                    out, h = model(decoder_input.float(), h)
                    predictions.append(out)
                    decoder_input = out
                out = torch.cat(predictions, dim=1)

            elif isinstance(model, TimeSeriesTransformer):
                out = train_transformer_autoregressive(model, x.float(), horizon, device)
            else:
                out = model(x.float())

            if is_s2s:
                out_shape = out.shape
                out_denorm = scaler.inverse_transform(out.cpu().numpy().reshape(-1, out.shape[-1])).reshape(out_shape)
                # 我们关心的是第horizon步的预测，即序列的最后一个点
                all_predictions.append(out_denorm[:, -1, :])
            else:
                out_denorm = scaler.inverse_transform(out.cpu().numpy())
                all_predictions.append(out_denorm)

    end_time = time.time()
    total_time = end_time - start_time

    return np.concatenate(all_predictions, axis=0), total_time


def epoch_loop(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, 
               epochs: int, lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau, 
               criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, 
               scaler: StandardScaler, model_name: str, results_dir: str, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    training_loss, validation_loss = [], []
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 30

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
            model_save_path = os.path.join(results_dir, f'{model_name}_{horizon}_best_model_magnitude_heli.pth')
            scaler_save_path = os.path.join(results_dir, f'{model_name}_{horizon}_scaler_magnitude_heli.gz')
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
                     naive_preds: np.ndarray, num_timesteps: int = 150, subcarrier_idx: int = 0):
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
    
    # Naive预测是使用t-1的值预测t，所以绘图时，naive_preds[i]对应true_vals[i]
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
    horizons = [1]
    lookback = 5
    ts_step = 1
    epochs = 500
    batch_size = 512
    learning_rates = {'GRU': 0.000025, 'LSTM': 0.0001, 'CNN': 0.0001, 'MLP': 0.0001, 'Transformer': 0.00005}

    # 设置所有随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)    
    
    print("--- 1. 配置实验参数 ---")
    project_root = '.'
    data_dir = os.path.join(project_root, 'data')
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    combined_data_file = os.path.join(data_dir, 'channel_estimates_20250827_174423_heli_flight_for_training_testing.mat')
    results_file = os.path.join(results_dir, f'performance_results_magnitude_lookback{lookback}_heli.csv')
    
    if not os.path.exists(combined_data_file):
        raise FileNotFoundError(f"数据文件不存在: {combined_data_file}")

    print(f"数据文件路径: {combined_data_file}")
    print(f"结果文件路径: {results_file}")

    # --- 4.2. 设备设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备已设置为: {device}")

    # --- 4.3. 数据加载与分割 ---
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

    # --- 4.4. 实验循环 ---
    print("\n--- 3. 开始实验循环 ---")
    performance_df = pd.DataFrame()
    if os.path.exists(results_file):
        performance_df = pd.read_csv(results_file, index_col=0)

    #models_to_run = ['Naive Outdated', 'GRU', 'LSTM', 'CNN', 'MLP', 'Transformer']
    models_to_run = ['Naive Outdated', 'Transformer']

    for horizon in horizons:
        print(f"\n{'='*25} 正在处理 Horizon = {horizon} {'='*25}")

        # ==================== 关键修复 1: 稳健的数据分割逻辑 ====================
        if len(continuous_blocks) == 1:
            print("检测到单个数据块。将在块内按80/20比例分割训练集和测试集。")
            single_block = continuous_blocks[0]
            split_point = int(len(single_block) * 0.8)
            
            # 确保分割后的块不为空
            if split_point > 0 and len(single_block) - split_point > 0:
                train_blocks = [single_block[:split_point]]
                test_blocks = [single_block[split_point:]]
            else:
                # 如果块太小无法分割，则全部放入测试集并后续跳过训练
                train_blocks = []
                test_blocks = [single_block]
        else:
            print(f"检测到 {len(continuous_blocks)} 个数据块。将按块列表分割训练集和测试集。")
            split_idx = int(len(continuous_blocks) * 0.8)
            if split_idx == 0 and len(continuous_blocks) > 1:
                split_idx = 1 # 确保至少有一个块用于训练
            train_blocks, test_blocks = continuous_blocks[:split_idx], continuous_blocks[split_idx:]
        
        print(f"数据分割结果: {len(train_blocks)} 个训练块, {len(test_blocks)} 个测试块。")
        # ========================================================================

        print("正在为 '多对一' 和 '序列到序列' 任务准备数据...")
        
        # 为训练集生成样本
        train_x_m2o_list, train_y_m2o_list = [], []
        train_x_s2s_list, train_y_s2s_list = [], []
        for block in train_blocks:
            x_m2o, y_m2o = generate_samples_from_block_m2o(block, lookback, horizon, ts_step)
            if x_m2o.size > 0:
                train_x_m2o_list.append(x_m2o)
                train_y_m2o_list.append(y_m2o)
                
            x_s2s, y_s2s = generate_samples_from_block_s2s(block, lookback, horizon, ts_step)
            if x_s2s.size > 0:
                train_x_s2s_list.append(x_s2s)
                train_y_s2s_list.append(y_s2s)
        
        # ==================== 关键修复 2: 样本生成安全检查 ====================
        # 在合并前检查是否有可用的训练数据，防止因数据过短无法生成样本而崩溃
        if not train_x_m2o_list or not train_x_s2s_list:
            print(f"警告: 未能为 horizon={horizon} 和 lookback={lookback} 生成足够的训练样本。")
            print("这可能是因为分割后的训练数据块对于当前参数来说太短了。将跳过此 horizon 的训练。")
            continue # 跳到下一个 horizon
        # ========================================================================

        train_x_m2o, train_y_m2o = np.concatenate(train_x_m2o_list), np.concatenate(train_y_m2o_list)
        train_x_s2s, train_y_s2s = np.concatenate(train_x_s2s_list), np.concatenate(train_y_s2s_list)

        # 为测试集生成样本
        test_x_m2o_list, test_y_m2o_list = [], []
        test_x_s2s_list, test_y_s2s_list = [], []
        for block in test_blocks:
            x_m2o, y_m2o = generate_samples_from_block_m2o(block, lookback, horizon, ts_step)
            if x_m2o.size > 0:
                test_x_m2o_list.append(x_m2o)
                test_y_m2o_list.append(y_m2o)
                
            x_s2s, y_s2s = generate_samples_from_block_s2s(block, lookback, horizon, ts_step)
            if x_s2s.size > 0:
                test_x_s2s_list.append(x_s2s)
                test_y_s2s_list.append(y_s2s)
        
        if not test_x_m2o_list or not test_x_s2s_list:
             print(f"警告: 未能为 horizon={horizon} 和 lookback={lookback} 生成测试样本。跳过此 horizon。")
             continue

        test_x_m2o, test_y_m2o = np.concatenate(test_x_m2o_list), np.concatenate(test_y_m2o_list)
        test_x_s2s, test_y_s2s = np.concatenate(test_x_s2s_list), np.concatenate(test_y_s2s_list)
        
        # 数据归一化
        scaler_m2o = StandardScaler().fit(train_x_m2o.reshape(-1, NUM_ACTIVE_SUBCARRIERS))
        scaler_s2s = StandardScaler().fit(train_x_s2s.reshape(-1, NUM_ACTIVE_SUBCARRIERS))
        print("数据准备和归一化完成。")

        # Naive 预测基准
        # Naive 预测使用 x 的最后一个时间步来预测 y
        naive_predictions = test_x_m2o[:, -1, :]
        # Naive 预测的目标是 m2o 的目标
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
                train_x_norm = scaler_s2s.transform(train_x_s2s.reshape(-1, NUM_ACTIVE_SUBCARRIERS)).reshape(train_x_s2s.shape)
                train_y_norm = scaler_s2s.transform(train_y_s2s.reshape(-1, NUM_ACTIVE_SUBCARRIERS)).reshape(train_y_s2s.shape)
                test_x_norm = scaler_s2s.transform(test_x_s2s.reshape(-1, NUM_ACTIVE_SUBCARRIERS)).reshape(test_x_s2s.shape)
                test_y_norm = scaler_s2s.transform(test_y_s2s.reshape(-1, NUM_ACTIVE_SUBCARRIERS)).reshape(test_y_s2s.shape)
                train_dataset = TensorDataset(torch.from_numpy(train_x_norm), torch.from_numpy(train_y_norm), torch.from_numpy(train_y_s2s))
                test_dataset = TensorDataset(torch.from_numpy(test_x_norm), torch.from_numpy(test_y_norm), torch.from_numpy(test_y_s2s))
                scaler = scaler_s2s
            else: # CNN, MLP
                train_x_norm = scaler_m2o.transform(train_x_m2o.reshape(-1, NUM_ACTIVE_SUBCARRIERS)).reshape(train_x_m2o.shape)
                train_y_norm = scaler_m2o.transform(train_y_m2o.reshape(-1, 1, NUM_ACTIVE_SUBCARRIERS)).reshape(train_y_m2o.shape)
                test_x_norm = scaler_m2o.transform(test_x_m2o.reshape(-1, NUM_ACTIVE_SUBCARRIERS)).reshape(test_x_m2o.shape)
                test_y_norm = scaler_m2o.transform(test_y_m2o.reshape(-1, 1, NUM_ACTIVE_SUBCARRIERS)).reshape(test_y_m2o.shape)
                train_dataset = TensorDataset(torch.from_numpy(train_x_norm), torch.from_numpy(train_y_norm), torch.from_numpy(train_y_m2o))
                test_dataset = TensorDataset(torch.from_numpy(test_x_norm), torch.from_numpy(test_y_norm), torch.from_numpy(test_y_m2o))
                scaler = scaler_m2o

            # 模型初始化
            if model_name == 'Transformer':
                model = TimeSeriesTransformer(input_size=NUM_ACTIVE_SUBCARRIERS, dim_val=1024, n_encoder_layers=2, n_decoder_layers=2, n_heads=8,
                                              dropout_encoder=0.1, dropout_decoder=0.1, dropout_pos_enc=0.1,
                                              dim_feedforward_encoder=1024, dim_feedforward_decoder=1024,
                                              max_seq_len=lookback+horizon, dec_seq_len=horizon, out_seq_len=horizon)
            elif model_name == 'GRU':
                model = GRUNet_Seq2Seq(input_dim=NUM_ACTIVE_SUBCARRIERS, hidden_dim=512, output_dim=NUM_ACTIVE_SUBCARRIERS, n_layers=2)
            elif model_name == 'LSTM':
                model = LSTMNet_Seq2Seq(input_dim=NUM_ACTIVE_SUBCARRIERS, hidden_dim=512, output_dim=NUM_ACTIVE_SUBCARRIERS, n_layers=2)
            elif model_name == 'CNN': 
                model = CNNNet(lookback, input_channels=NUM_ACTIVE_SUBCARRIERS)
            elif model_name == 'MLP': 
                model = MLPNet(lookback, input_channels=NUM_ACTIVE_SUBCARRIERS)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[model_name], weight_decay=1e-5)
            criterion = nn.MSELoss()
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)

            training_loss, validation_loss = epoch_loop(
                model, train_loader, test_loader, epochs, lr_scheduler, 
                criterion, optimizer, device, scaler, model_name, results_dir, horizon
            )

            if len(validation_loss) > 0:
                final_loss = min(validation_loss)
                performance_df.loc[horizon, model_name] = final_loss
                print(f" -> {model_name} 完成训练。最佳验证 MSE: {final_loss:.6f}")
                
            print(f" -> 正在加载最佳模型并进行最终预测...")
            model.load_state_dict(torch.load(os.path.join(results_dir, f'{model_name}_{horizon}_best_model_magnitude_heli.pth')))
            preds, total_pred_time = predict(model, test_loader, device, scaler, horizon)

            num_samples = len(test_loader.dataset)
            avg_time = total_pred_time / num_samples
            print(f" -> 在 {num_samples} 个测试样本上总耗时: {total_pred_time:.4f} 秒")
            print(f" -> 平均单步预测时间: {avg_time * 1e6:.2f} µs")

            # 根据模型类型选择正确的真实值进行绘图
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