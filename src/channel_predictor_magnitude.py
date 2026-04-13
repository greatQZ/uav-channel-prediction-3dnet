# channel_predictor_new_magnitude.py

import torch
import torch.nn as nn
import numpy as np
import joblib
import math
import scipy.io
import matplotlib.pyplot as plt
import os

# --- 步骤 1: 复制模型定义 (无变化) ---
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

# --- 步骤 2: 创建预测与可视化函数 (预测函数逻辑无变化) ---
def predict_transformer(model, scaler, history_data, horizon, device):
    """改进的Transformer预测函数"""
    model.eval()
    
    # 标准化输入数据
    history_shape = history_data.shape
    history_data_norm = scaler.transform(history_data.reshape(-1, history_shape[-1])).reshape(history_shape)
    src = torch.from_numpy(history_data_norm).float().unsqueeze(0).to(device)
    
    # 创建decoder输入：使用历史序列的最后一个值重复
    decoder_input = src[:, -1:, :].repeat(1, src.size(1), 1)
    
    # 逐步预测（自回归）
    predictions = []
    for _ in range(horizon):
        with torch.no_grad():
            tgt_mask = generate_square_subsequent_mask(decoder_input.size(1), decoder_input.size(1)).to(device)
            prediction = model(src, decoder_input, tgt_mask=tgt_mask)
            
            # 获取最后一个时间步的预测
            next_step = prediction[:, -1:, :]
            predictions.append(next_step.cpu().numpy())
            
            # 更新decoder输入：移除第一个时间步，添加新的预测
            decoder_input = torch.cat([decoder_input[:, 1:, :], next_step], dim=1)
    
    # 合并所有预测
    prediction_norm = np.concatenate(predictions, axis=1)
    
    # 反标准化
    prediction_real = scaler.inverse_transform(
        prediction_norm.reshape(-1, prediction_norm.shape[-1])
    ).reshape(prediction_norm.shape)
    
    return prediction_real.squeeze()

# ### 修改点 ### - 简化绘图函数，只绘制幅度
def plot_predictions(true_vals, transformer_preds, naive_preds, num_timesteps=150, subcarrier_idx=0):
    """绘制真实值与预测值的对比图 (只针对幅度)。"""
    plt.figure(figsize=(15, 7))
    
    # 绘制幅度
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
    # ### 修改点 1 ### - 更新特征总数，现在是幅度，所以没有虚部
    NUM_SUBCARRIERS = 576
    TOTAL_FEATURES = NUM_SUBCARRIERS 
    
    # Transformer模型参数 (必须与训练时完全一致)
    DIM_VAL = 128
    N_ENCODER_LAYERS = 2
    N_DECODER_LAYERS = 2
    N_HEADS = 8
    
    # --- 加载模型和Scaler ---
    print("--- 正在加载模型和Scaler ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模型初始化时使用新的TOTAL_FEATURES
    model = TimeSeriesTransformer(
        input_size=TOTAL_FEATURES, dim_val=DIM_VAL, n_encoder_layers=N_ENCODER_LAYERS, 
        n_decoder_layers=N_DECODER_LAYERS, n_heads=N_HEADS, dropout_encoder=0.1, 
        dropout_decoder=0.1, dropout_pos_enc=0.1, dim_feedforward_encoder=512, 
        dim_feedforward_decoder=512, max_seq_len=LOOKBACK, dec_seq_len=LOOKBACK, out_seq_len=HORIZON
    )
    
    model_path = 'results/Transformer_best_model.pth' 
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(f"模型已从 {model_path} 加载。")
    
    scaler_path = 'results/Transformer_scaler.gz'
    scaler = joblib.load(scaler_path)
    print(f"Scaler已从 {scaler_path} 加载。")
    
    # --- 【新功能】加载并解析测试.mat文件 ---
    print("\n--- 正在加载并解析测试.mat文件 ---")
    test_mat_file_path = 'data/my_H_clean_combined_padded_optimized_test_autopilot_40m_for_test.mat'
    
    if not os.path.exists(test_mat_file_path):
        raise FileNotFoundError(f"测试数据文件不存在: {test_mat_file_path}")
        
    mat = scipy.io.loadmat(test_mat_file_path)
    test_data_1d = mat['H'].flatten().astype(np.float32)
    
    # ### 修改点 2 ### - 将复数数据转换为幅度数据
    # 原始数据是 [real1, imag1, real2, imag2, ...], 维度为 (timesteps, 576*2)
    # 1. 转换成 (timesteps, 576, 2) 以便分离实部和虚部
    complex_data = test_data_1d.reshape(-1, NUM_SUBCARRIERS, 2)
    # 2. 计算幅度 sqrt(real^2 + imag^2)
    real_part = complex_data[:, :, 0]
    imag_part = complex_data[:, :, 1]
    test_data_2d = np.sqrt(real_part**2 + imag_part**2)
    
    print(f"成功从 {test_mat_file_path} 加载并转换为 {test_data_2d.shape[0]} 个时间步的幅度数据。")
    
    # 验证标准化过程 (逻辑不变)
    print("\n验证标准化过程:")
    sample_data = test_data_2d[:LOOKBACK]
    print(f"原始数据范围: [{sample_data.min():.6f}, {sample_data.max():.6f}]")
    sample_norm = scaler.transform(sample_data)
    print(f"标准化后范围: [{sample_norm.min():.6f}, {sample_norm.max():.6f}]")
    sample_restored = scaler.inverse_transform(sample_norm)
    print(f"反标准化后范围: [{sample_restored.min():.6f}, {sample_restored.max():.6f}]")
    mse = np.mean((sample_data - sample_restored)**2)
    print(f"标准化-反标准化MSE: {mse:.10f}")
    
    # --- 【新功能】对整个测试文件进行预测 ---
    print("\n--- 正在对整个测试文件进行预测 ---")
    transformer_preds = []
    naive_preds = []
    true_vals = []
    
    num_test_predictions = len(test_data_2d) - LOOKBACK - HORIZON
    for i in range(num_test_predictions):
        history_data = test_data_2d[i : i + LOOKBACK]
        true_value = test_data_2d[i + LOOKBACK + HORIZON - 1]
        true_vals.append(true_value)
        
        naive_pred = history_data[-1, :]
        naive_preds.append(naive_pred)
        
        transformer_pred = predict_transformer(model, scaler, history_data, HORIZON, device)
        transformer_preds.append(transformer_pred)
        
        if (i + 1) % 500 == 0:
            print(f"  已完成 {i+1}/{num_test_predictions} 次预测...")

    transformer_preds = np.array(transformer_preds)
    naive_preds = np.array(naive_preds)
    true_vals = np.array(true_vals)

    print("\n--- 性能评估 ---")
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

    print("所有预测已完成。")
    
    print("\n--- 正在生成预测对比图 ---")
    plot_predictions(true_vals, transformer_preds, naive_preds)