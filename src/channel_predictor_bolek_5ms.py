# -*- coding: utf-8 -*-
"""
DNN Channel Prediction (Magnitude Version): A comprehensive script for training
and evaluating various deep learning models for predicting wireless channel magnitude.

v5: 根据用户需求，加入了TS_STEP参数，以确保预测时的数据采样方式与训练时完全一致。
"""

import os
import time
import math
import scipy.io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import random
from typing import Tuple

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
        decoder_output = self.positional_encoding_layer(decoder_output)
        decoder_output = self.decoder(tgt=decoder_output, memory=src, tgt_mask=tgt_mask, memory_mask=src_mask)
        decoder_output = self.linear_mapping(decoder_output)
        return decoder_output[:, -self.out_seq_len:, :]

def generate_square_subsequent_mask(dim1, dim2):
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

# ==============================================================================
# 2. 预测与可视化函数 (Prediction & Visualization Functions)
# ==============================================================================

def predict_transformer_autoregressive(model, x_norm_tensor, horizon, device):
    model.eval()
    with torch.no_grad():
        src = model.encoder_input_layer(x_norm_tensor)
        src = model.positional_encoding_layer(src)
        memory = model.encoder(src=src)
        decoder_input = x_norm_tensor[:, -1:, :].float()
        for _ in range(horizon):
            tgt_len = decoder_input.size(1)
            tgt_mask = generate_square_subsequent_mask(tgt_len, tgt_len).to(device)
            tgt_encoded = model.decoder_input_layer(decoder_input)
            tgt_with_pe = model.positional_encoding_layer(tgt_encoded)
            output = model.decoder(tgt=tgt_with_pe, memory=memory, tgt_mask=tgt_mask)
            output = model.linear_mapping(output)
            next_step_pred = output[:, -1:, :]
            decoder_input = torch.cat([decoder_input, next_step_pred], dim=1)
    return decoder_input[:, 1:, :]

def plot_predictions(true_vals, transformer_preds, naive_preds, num_timesteps=150, subcarrier_idx=0):
    plt.figure(figsize=(15, 7))
    plt.plot(true_vals[:num_timesteps, subcarrier_idx], label='True Magnitude', color='blue', linewidth=2)
    plt.plot(transformer_preds[:num_timesteps, subcarrier_idx], label='Transformer predicted magnitude', color='red', linestyle='--')
    plt.plot(naive_preds[:num_timesteps, subcarrier_idx], label='Naive predicted magnitude', color='green', linestyle=':')
    plt.title(f'Subcarrier #{subcarrier_idx} magnitude prediction comparison (first {num_timesteps} timesteps)')
    plt.xlabel('Time Step')
    plt.ylabel('CSI Magnitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 3. 主执行流程 (Main Execution Flow)
# ==============================================================================

if __name__ == '__main__':
    # --- 配置参数 (必须与训练时完全一致) ---
    LOOKBACK = 10
    HORIZON = 20
    
    # =======================================================================
    #               ↓↓↓ 新增 TS_STEP 参数 ↓↓↓
    # =======================================================================
    # 这个值必须与您训练模型时使用的ts_step完全一致
    TS_STEP = 2 
    # =======================================================================

    NUM_SUBCARRIERS = 576
    TOTAL_FEATURES = NUM_SUBCARRIERS
    DIM_VAL = 512
    N_ENCODER_LAYERS = 2
    N_DECODER_LAYERS = 2
    N_HEADS = 16
    DROPOUT = 0.2
    DIM_FEEDFORWARD = 1024
    
    # --- 加载模型和Scaler ---
    print("--- 正在加载模型和Scaler ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model_path = f'results/Transformer_lookback{LOOKBACK}_horizon{HORIZON}_best_model_magnitude_bolek_para2_5ms.pth' 
    scaler_path = f'results/Transformer_lookback{LOOKBACK}_horizon{HORIZON}_scaler_magnitude_bolek_para2_5ms.gz'

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"未找到与 Horizon={HORIZON}, Lookback={LOOKBACK} 匹配的模型或Scaler文件。请检查路径:\n{model_path}\n{scaler_path}")

    model = TimeSeriesTransformer(
        input_size=TOTAL_FEATURES, dim_val=DIM_VAL, n_encoder_layers=N_ENCODER_LAYERS, 
        n_decoder_layers=N_DECODER_LAYERS, n_heads=N_HEADS, dropout_encoder=DROPOUT, 
        dropout_decoder=DROPOUT, dropout_pos_enc=0.1, dim_feedforward_encoder=DIM_FEEDFORWARD, 
        dim_feedforward_decoder=DIM_FEEDFORWARD, 
        max_seq_len=LOOKBACK + HORIZON,
        dec_seq_len=HORIZON,
        out_seq_len=HORIZON
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(f"模型已从 {model_path} 加载。")
    
    scaler = joblib.load(scaler_path)
    print(f"Scaler已从 {scaler_path} 加载。")
    
    # --- 加载并解析测试.mat文件 ---
    print("\n--- 正在加载并解析测试.mat文件 ---")
    test_mat_file_path = 'data/my_H_real_data_bolek_for_5ms_40m_10mps_for_demo.mat'
    
    if not os.path.exists(test_mat_file_path):
        raise FileNotFoundError(f"测试数据文件不存在: {test_mat_file_path}")
        
    mat = scipy.io.loadmat(test_mat_file_path)
    test_data_1d = mat['H'].flatten().astype(np.float32)
    
    complex_data = test_data_1d.reshape(-1, NUM_SUBCARRIERS, 2)
    real_part = complex_data[:, :, 0]
    imag_part = complex_data[:, :, 1]
    test_data_2d = np.sqrt(real_part**2 + imag_part**2)
    print(f"成功从 {test_mat_file_path} 加载并转换为 {test_data_2d.shape[0]} 个时间步的幅度数据。")
    
    print("\n--- 正在过滤保护区域的0值，并将数据分割成有效块 ---")
    is_zero_row = ~test_data_2d.any(axis=1)
    block_change_indices = np.where(np.diff(is_zero_row))[0] + 1
    data_blocks = np.split(test_data_2d, block_change_indices)
    continuous_blocks = [block for block in data_blocks if not np.all(block == 0)]
    print(f"已将测试数据分割成 {len(continuous_blocks)} 个有效的连续数据块。")
    
    # --- 对整个测试文件进行预测 ---
    print(f"\n--- 正在对所有有效数据块进行预测 (Lookback={LOOKBACK}, Horizon={HORIZON}, Step={TS_STEP}) ---")
    
    all_transformer_preds = []
    all_naive_preds = []
    all_true_vals = []
    
    for block_idx, block in enumerate(continuous_blocks):
        print(f"  正在处理数据块 #{block_idx + 1}/{len(continuous_blocks)} (长度: {len(block)})...")
        
        # =======================================================================
        #               ↓↓↓ 重构数据采样循环以支持 TS_STEP ↓↓↓
        # =======================================================================
        # 计算生成一个样本所需要的总长度
        required_len = (LOOKBACK - 1) * TS_STEP + HORIZON
        
        if len(block) < required_len:
            print(f"    数据块过短 (长度 < {required_len})，跳过。")
            continue
        
        # 计算当前块可以生成多少个预测样本
        num_block_predictions = len(block) - required_len

        for i in range(num_block_predictions):
            # 确定输入序列的结束点
            input_end_idx = i + (LOOKBACK - 1) * TS_STEP
            # 根据ts_step生成输入序列的索引
            input_indices = np.arange(i, input_end_idx + 1, TS_STEP)
            history_data = block[input_indices]
            
            # 真实值是输入序列最后一个点之后 horizon 步的值
            target_idx = input_end_idx + HORIZON
            true_value = block[target_idx]
            all_true_vals.append(true_value)
            
            # Naive 预测是输入序列的最后一个值
            naive_pred = history_data[-1, :]
            all_naive_preds.append(naive_pred)
            
            # 预测流程
            history_data_norm = scaler.transform(history_data)
            src = torch.from_numpy(history_data_norm).float().unsqueeze(0).to(device)
            prediction_norm_tensor = predict_transformer_autoregressive(model, src, HORIZON, device)
            
            # 从预测序列中取出第 horizon 步
            final_step_norm = prediction_norm_tensor[:, HORIZON - 1, :].cpu().numpy()
            prediction_real = scaler.inverse_transform(final_step_norm)
            all_transformer_preds.append(prediction_real.squeeze())
        # =======================================================================
        #               ↑↑↑ 循环重构结束 ↑↑↑
        # =======================================================================

    print("\n所有数据块处理完毕，正在合并结果...")
    transformer_preds = np.array(all_transformer_preds)
    naive_preds = np.array(all_naive_preds)
    true_vals = np.array(all_true_vals)

    print("\n--- 性能评估 ---")
    if len(true_vals) > 0:
        transformer_mse = np.mean((transformer_preds - true_vals)**2)
        naive_mse = np.mean((naive_preds - true_vals)**2)

        print(f"Transformer MSE: {transformer_mse:.6f}")
        print(f"Naive MSE: {naive_mse:.6f}")

        if transformer_mse < naive_mse:
            improvement = (naive_mse - transformer_mse) / naive_mse * 100
            print(f"Transformer 比 Naive 方法改进了 {improvement:.2f}%")
        else:
            deterioration = (transformer_mse - naive_mse) / naive_mse * 100
            print(f"Transformer 比 Naive 方法差了 {deterioration:.2f}%")
    else:
        print("没有生成任何有效的预测，无法进行评估。")

    print("所有预测已完成。")
    
    print("\n--- 正在生成预测对比图 ---")
    if len(true_vals) > 0:
        plot_predictions(true_vals, transformer_preds, naive_preds)
    else:
        print("没有可用于绘图的数据。")