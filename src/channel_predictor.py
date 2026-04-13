import torch
import torch.nn as nn
import numpy as np
import joblib
import math
import scipy.io
import matplotlib.pyplot as plt
import os

# --- 步骤 1: 复制模型定义 ---
# 您必须将您使用的模型（这里是Transformer）的完整类定义复制到这个新脚本中，
# 因为PyTorch需要知道模型的结构才能加载权重。

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

# --- 步骤 2: 创建预测与可视化函数 ---

#def predict_single_step(model, scaler, history_data, device):
    """对单一样本进行预测。"""
    model.eval()
    history_shape = history_data.shape
    history_data_norm = scaler.transform(history_data.reshape(-1, history_shape[-1])).reshape(history_shape)
    
    # 1. src (encoder input) 保持不变，是完整的历史
    src = torch.from_numpy(history_data_norm).float().unsqueeze(0).to(device)
    
    # 2. tgt (decoder input) 应该是解码的起始信号，即历史的最后一个点
    #    形状需要是 (batch, seq_len, features)，这里 seq_len 是 1
    tgt = src[:, -1:, :] 
    
    with torch.no_grad():
        # 对于单步预测，tgt_mask 不是必需的，但保留亦无妨
        prediction_norm = model(src, tgt)
        
    # 模型的 forward 函数返回的是 (batch, out_seq_len, features)
    # 因为 HORIZON=1, 所以 out_seq_len=1
    prediction_norm_single = prediction_norm.cpu().numpy()
    
    # 反归一化
    prediction_real = scaler.inverse_transform(prediction_norm_single.reshape(1, -1))
    
    return prediction_real.squeeze()
def predict_single_step(model, scaler, history_data, device):
    """对单一样本进行预测。"""
    model.eval()
    history_shape = history_data.shape
    history_data_norm = scaler.transform(history_data.reshape(-1, history_shape[-1])).reshape(history_shape)
    src = torch.from_numpy(history_data_norm).float().unsqueeze(0).to(device)  # 形状: [1, lookback, features]
    
    # 修正decoder输入：使用历史序列的最后一个值重复，构成长度为lookback的序列
    last_value = src[:, -1:, :]  # 取最后一个时间步，形状: [1, 1, features]
    tgt = last_value.repeat(1, src.size(1), 1)  # 重复lookback次，形状: [1, lookback, features]
    
    with torch.no_grad():
        tgt_mask = generate_square_subsequent_mask(tgt.size(1), tgt.size(1)).to(device)
        prediction_norm = model(src, tgt, tgt_mask=tgt_mask)  # 预测输出形状: [1, out_seq_len, features]
    prediction_norm_single = prediction_norm[:, -1, :].cpu().numpy()  # 取最后一个时间步（对应horizon=1）
    prediction_real = scaler.inverse_transform(prediction_norm_single)
    return prediction_real.squeeze()

def plot_predictions(true_vals, transformer_preds, naive_preds, num_timesteps=150, subcarrier_idx=0):
    """【新功能】绘制真实值与预测值的对比图，包括实部和虚部。"""
    fig, axs = plt.subplots(2, 1, figsize=(15, 12))
    
    # 绘制实部
    axs[0].plot(true_vals[:num_timesteps, subcarrier_idx], label='Real Part', color='blue', linewidth=2)
    axs[0].plot(transformer_preds[:num_timesteps, subcarrier_idx], label='Transformer', color='red', linestyle='--')
    axs[0].plot(naive_preds[:num_timesteps, subcarrier_idx], label='Naive outdated', color='green', linestyle=':')
    axs[0].set_title(f'subcarrier #{subcarrier_idx} Real part prediction comparison (first {num_timesteps} timesteps)')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Real part of channel coefficient')
    axs[0].legend()
    axs[0].grid(True)
    
    # 绘制虚部 (索引是 subcarrier_idx + 1，因为数据是交错排列的)
    imag_idx = subcarrier_idx + 1
    axs[1].plot(true_vals[:num_timesteps, imag_idx], label='Imaginary Part', color='blue', linewidth=2)
    axs[1].plot(transformer_preds[:num_timesteps, imag_idx], label='Transformer ', color='red', linestyle='--')
    axs[1].plot(naive_preds[:num_timesteps, imag_idx], label='Naive outdated', color='green', linestyle=':')
    axs[1].set_title(f'subcarrier #{subcarrier_idx} Imaginary part prediction comparison (first {num_timesteps} timesteps)')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Imaginary part of channel coefficient')
    axs[1].legend()
    axs[1].grid(True)
    
    fig.tight_layout()
    plt.show()

# --- 步骤 3: 主执行流程 ---

if __name__ == '__main__':
    # --- 配置参数 (必须与训练时完全一致) ---
    LOOKBACK = 5
    HORIZON = 1
    TOTAL_FEATURES = 576 * 2
    
    # Transformer模型参数 (必须与训练时完全一致)
    DIM_VAL = 128
    N_ENCODER_LAYERS = 2
    N_DECODER_LAYERS = 2
    N_HEADS = 8
    
    # --- 加载模型和Scaler ---
    print("--- 正在加载模型和Scaler ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    # ★★★ 请将这里替换为您的测试.mat文件路径 ★★★
    test_mat_file_path = 'data/my_H_clean_combined_padded_optimized_test_autopilot_40m_for_test.mat'
    
    if not os.path.exists(test_mat_file_path):
        raise FileNotFoundError(f"测试数据文件不存在: {test_mat_file_path}")
        
    mat = scipy.io.loadmat(test_mat_file_path)
    test_data_1d = mat['H'].flatten().astype(np.float32)
    test_data_2d = test_data_1d.reshape(-1, TOTAL_FEATURES)
    print(f"成功从 {test_mat_file_path} 加载并解析了 {test_data_2d.shape[0]} 个时间步的数据。")

    # --- 【新功能】对整个测试文件进行预测 ---
    print("\n--- 正在对整个测试文件进行预测 ---")
    transformer_preds = []
    naive_preds = []
    true_vals = []
    
    # 遍历测试数据，创建输入序列并进行预测
    num_test_predictions = len(test_data_2d) - LOOKBACK - HORIZON
    for i in range(num_test_predictions):
        # 提取历史数据作为输入
        history_data = test_data_2d[i : i + LOOKBACK]
        
        # 获取真实的目标值
        true_value = test_data_2d[i + LOOKBACK + HORIZON - 1]
        true_vals.append(true_value)
        
        # Naive预测：用历史序列的最后一个值作为预测
        naive_pred = history_data[-1, :]
        naive_preds.append(naive_pred)
        
        # Transformer预测
        transformer_pred = predict_single_step(model, scaler, history_data, device)
        transformer_preds.append(transformer_pred)
        
        # 打印进度
        if (i + 1) % 500 == 0:
            print(f"  已完成 {i+1}/{num_test_predictions} 次预测...")

    # 将结果列表转换为Numpy数组
    transformer_preds = np.array(transformer_preds)
    naive_preds = np.array(naive_preds)
    true_vals = np.array(true_vals)
    
    print("所有预测已完成。")
    
    # --- [修正点] 为绘图准备正确对齐的数据 ---
    print("\n--- 正在为绘图准备真正对齐的数据 ---")

    # 预测开始的时间点索引。第一个预测是针对 test_data_2d[5] 的，所以索引是 5。
    prediction_start_idx = LOOKBACK 

    # 我们想要绘制的总时间步数量
    num_timesteps_to_plot = 150

    # 1. 准备用于绘图的真实值：这是原始数据的一部分，包含历史和预测范围
    # 确保我们只截取需要的部分，以防数据不足
    plot_end_idx = prediction_start_idx + len(transformer_preds)
    plot_true_vals = test_data_2d[:plot_end_idx]

    # 2. 为预测值创建带正确时间索引的数组
    # 创建一个和 plot_true_vals 等长的、填满 NaN 的空数组
    plot_transformer_preds = np.full_like(plot_true_vals, np.nan, dtype=np.float32)
    plot_naive_preds = np.full_like(plot_true_vals, np.nan, dtype=np.float32)

    # 3. 在正确的位置（从 prediction_start_idx 开始）填入我们计算出的预测值
    plot_transformer_preds[prediction_start_idx:plot_end_idx] = transformer_preds
    plot_naive_preds[prediction_start_idx:plot_end_idx] = naive_preds

    # --- [修正点] 使用对齐后的新数组来调用绘图函数 ---
    print("\n--- 正在生成最终的、对齐的预测对比图 ---")
    
    # 我们传递新创建的、包含 NaN 的对齐数组
    # 并指定我们想看的窗口大小
    plot_predictions(
        plot_true_vals, 
        plot_transformer_preds, 
        plot_naive_preds, 
        num_timesteps=num_timesteps_to_plot
    )# --- 【新功能】可视化预测结果 ---
    