# channel_predictor_magnitude.py
# [V3 - Final Audited Version]
# - Added logic to split the test data into continuous blocks, removing zero-padded
#   guard regions, ensuring the prediction process matches the training process.
# - The prediction loop now iterates over each valid block.
# - This provides a more accurate and realistic performance evaluation.

import torch
import torch.nn as nn
import numpy as np
import joblib
import math
import scipy.io
import matplotlib.pyplot as plt
import os

# --- 步骤 1: 复制模型定义 (与训练脚本完全一致) ---
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

def generate_square_subsequent_mask(dim1, dim2):
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

# --- 步骤 2: 移植并重构预测与可视化函数 ---

def predict_transformer_autoregressive(model, x_norm_tensor, horizon, device):
    """与 DS2.py 训练脚本完全一致的自回归预测函数。"""
    model.eval()
    with torch.no_grad():
        src = model.encoder_input_layer(x_norm_tensor)
        src = model.positional_encoding_layer(src)
        memory = model.encoder(src=src)
        decoder_input = x_norm_tensor[:, -1:, :]
        outputs = []
        for i in range(horizon):
            tgt_mask = generate_square_subsequent_mask(decoder_input.size(1), decoder_input.size(1)).to(device)
            tgt = model.decoder_input_layer(decoder_input)
            tgt = model.positional_encoding_layer(tgt)
            output = model.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            output = model.linear_mapping(output)
            next_input = output[:, -1:, :]
            decoder_input = torch.cat([decoder_input, next_input], dim=1)
            outputs.append(next_input)
    return torch.cat(outputs, dim=1)

def plot_predictions(true_vals, transformer_preds, naive_preds, num_timesteps=150, subcarrier_idx=0):
    """绘制真实值与预测值的对比图 (只针对幅度)。"""
    plt.figure(figsize=(15, 7))
    plt.plot(true_vals[:num_timesteps, subcarrier_idx], label='真实幅度 (True Magnitude)', color='blue', linewidth=2)
    plt.plot(transformer_preds[:num_timesteps, subcarrier_idx], label='Transformer 预测幅度', color='red', linestyle='--')
    plt.plot(naive_preds[:num_timesteps, subcarrier_idx], label='Naive 预测幅度', color='green', linestyle=':')
    plt.title(f'子载波 #{subcarrier_idx} 幅度预测对比 (前 {num_timesteps} 个时间步)')
    plt.xlabel('时间步 (Time Step)')
    plt.ylabel('信道系数幅度')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- 步骤 3: 主执行流程 ---

if __name__ == '__main__':
    # --- 配置参数 (必须与训练时完全一致) ---
    LOOKBACK = 5
    HORIZON = 1
    NUM_SUBCARRIERS = 576
    TOTAL_FEATURES = NUM_SUBCARRIERS
    
    # Transformer模型参数 (必须与训练时完全一致)
    DIM_VAL = 1024
    N_ENCODER_LAYERS = 2
    N_DECODER_LAYERS = 2
    N_HEADS = 8
    
    # --- 加载模型和Scaler ---
    print("--- 正在加载模型和Scaler ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model_path = f'results/Transformer_{HORIZON}_best_model_magnitude_heli.pth' 
    scaler_path = f'results/Transformer_{HORIZON}_scaler_magnitude_heli.gz'

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"未找到与 Horizon={HORIZON} 匹配的模型或Scaler文件。请检查路径:\n{model_path}\n{scaler_path}")

    model = TimeSeriesTransformer(
        input_size=TOTAL_FEATURES, dim_val=DIM_VAL, n_encoder_layers=N_ENCODER_LAYERS, 
        n_decoder_layers=N_DECODER_LAYERS, n_heads=N_HEADS, dropout_encoder=0.1, 
        dropout_decoder=0.1, dropout_pos_enc=0.1, dim_feedforward_encoder=1024, 
        dim_feedforward_decoder=1024, max_seq_len=LOOKBACK, dec_seq_len=LOOKBACK, out_seq_len=HORIZON
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(f"模型已从 {model_path} 加载。")
    
    scaler = joblib.load(scaler_path)
    print(f"Scaler已从 {scaler_path} 加载。")
    
    # --- 加载并解析测试.mat文件 ---
    print("\n--- 正在加载并解析测试.mat文件 ---")
    test_mat_file_path = 'data/heli_flight_H_real_data.mat'
    
    if not os.path.exists(test_mat_file_path):
        raise FileNotFoundError(f"测试数据文件不存在: {test_mat_file_path}")
        
    mat = scipy.io.loadmat(test_mat_file_path)
    test_data_1d = mat['H'].flatten().astype(np.float32)
    
    complex_data = test_data_1d.reshape(-1, NUM_SUBCARRIERS, 2)
    real_part = complex_data[:, :, 0]
    imag_part = complex_data[:, :, 1]
    test_data_2d = np.sqrt(real_part**2 + imag_part**2)
    print(f"成功从 {test_mat_file_path} 加载并转换为 {test_data_2d.shape[0]} 个时间步的幅度数据。")
    
    # <--- 变更点 1: 加入与训练脚本完全一致的数据块分割逻辑 ---
    print("\n--- 正在过滤保护区域的0值，并将数据分割成有效块 ---")
    is_zero_row = ~test_data_2d.any(axis=1)
    block_change_indices = np.where(np.diff(is_zero_row))[0] + 1
    data_blocks = np.split(test_data_2d, block_change_indices)
    continuous_blocks = [block for block in data_blocks if not np.all(block == 0)]
    print(f"已将测试数据分割成 {len(continuous_blocks)} 个有效的连续数据块。")
    
    # --- 对整个测试文件进行预测 ---
    print(f"\n--- 正在对所有有效数据块进行预测 (Lookback={LOOKBACK}, Horizon={HORIZON}) ---")
    
    # <--- 变更点 2: 创建空的列表，用于存储所有数据块的预测结果 ---
    all_transformer_preds = []
    all_naive_preds = []
    all_true_vals = []
    
    # <--- 变更点 3: 循环遍历每一个有效的数据块 ---
    for block_idx, block in enumerate(continuous_blocks):
        print(f"  正在处理数据块 #{block_idx + 1}/{len(continuous_blocks)} (长度: {len(block)})...")

        # 检查当前数据块的长度是否足够进行至少一次预测
        if len(block) < LOOKBACK + HORIZON:
            print(f"    数据块过短 (长度 < {LOOKBACK + HORIZON})，跳过。")
            continue
        
        # 在当前数据块内进行滑动窗口预测
        num_block_predictions = len(block) - LOOKBACK - HORIZON + 1
        for i in range(num_block_predictions):
            history_data = block[i : i + LOOKBACK]
            true_value = block[i + LOOKBACK + HORIZON - 1]
            all_true_vals.append(true_value)
            
            naive_pred = history_data[-1, :]
            all_naive_preds.append(naive_pred)
            
            # 预测流程
            history_data_norm = scaler.transform(history_data)
            src = torch.from_numpy(history_data_norm).float().unsqueeze(0).to(device)
            prediction_norm_tensor = predict_transformer_autoregressive(model, src, HORIZON, device)
            final_step_norm = prediction_norm_tensor[:, -1, :].cpu().numpy()
            prediction_real = scaler.inverse_transform(final_step_norm)
            all_transformer_preds.append(prediction_real.squeeze())

    # <--- 变更点 4: 将所有数据块的结果合并成最终的Numpy数组 ---
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