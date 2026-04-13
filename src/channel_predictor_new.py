import torch
import torch.nn as nn
import numpy as np
import joblib
import math
import scipy.io
import matplotlib.pyplot as plt
import os

# --- 步骤 1: 复制模型定义 ---
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

def predict_with_diagnostics(model, scaler, history_data, horizon, device):
    """带有诊断信息的预测函数"""
    print(f"历史数据形状: {history_data.shape}")
    
    # 标准化输入数据
    history_shape = history_data.shape
    history_data_norm = scaler.transform(history_data.reshape(-1, history_shape[-1])).reshape(history_shape)
    print(f"标准化后历史数据形状: {history_data_norm.shape}")
    
    src = torch.from_numpy(history_data_norm).float().unsqueeze(0).to(device)
    print(f"模型输入形状: {src.shape}")
    
    # 准备decoder输入
    decoder_input = src[:, -1:, :].clone()
    if src.size(1) > 1:
        decoder_input = decoder_input.repeat(1, src.size(1), 1)
    print(f"Decoder输入形状: {decoder_input.shape}")
    
    with torch.no_grad():
        # 生成掩码
        tgt_mask = generate_square_subsequent_mask(decoder_input.size(1), decoder_input.size(1)).to(device)
        print(f"掩码形状: {tgt_mask.shape}")
        
        # 进行预测
        prediction = model(src, decoder_input, tgt_mask=tgt_mask)
        print(f"原始预测形状: {prediction.shape}")
        
        # 获取预测结果
        prediction_norm = prediction.cpu().numpy()
        print(f"标准化预测值范围: [{prediction_norm.min():.6f}, {prediction_norm.max():.6f}]")
        
        # 反标准化
        prediction_real = scaler.inverse_transform(
            prediction_norm.reshape(-1, prediction_norm.shape[-1])
        ).reshape(prediction_norm.shape)
        print(f"反标准化预测值范围: [{prediction_real.min():.6f}, {prediction_real.max():.6f}]")
    
    return prediction_real.squeeze()

def plot_predictions(true_vals, transformer_preds, naive_preds, num_timesteps=150, subcarrier_idx=0):
    """绘制真实值与预测值的对比图，包括实部和虚部。"""
    fig, axs = plt.subplots(2, 1, figsize=(15, 12))
    
    # 绘制实部
    axs[0].plot(true_vals[:num_timesteps, subcarrier_idx], label='真实值 (Real Part)', color='blue', linewidth=2)
    axs[0].plot(transformer_preds[:num_timesteps, subcarrier_idx], label='Transformer 预测值', color='red', linestyle='--')
    axs[0].plot(naive_preds[:num_timesteps, subcarrier_idx], label='Naive 预测值', color='green', linestyle=':')
    axs[0].set_title(f'子载波 #{subcarrier_idx} 实部预测对比 (前 {num_timesteps} 个时间步)')
    axs[0].set_xlabel('时间步 (Time Step)')
    axs[0].set_ylabel('信道系数实部')
    axs[0].legend()
    axs[0].grid(True)
    
    # 绘制虚部 (索引是 subcarrier_idx + 1，因为数据是交错排列的)
    imag_idx = subcarrier_idx + 1
    axs[1].plot(true_vals[:num_timesteps, imag_idx], label='真实值 (Imaginary Part)', color='blue', linewidth=2)
    axs[1].plot(transformer_preds[:num_timesteps, imag_idx], label='Transformer 预测值', color='red', linestyle='--')
    axs[1].plot(naive_preds[:num_timesteps, imag_idx], label='Naive 预测值', color='green', linestyle=':')
    axs[1].set_title(f'子载波 #{subcarrier_idx} 虚部预测对比 (前 {num_timesteps} 个时间步)')
    axs[1].set_xlabel('时间步 (Time Step)')
    axs[1].set_ylabel('信道系数虚部')
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
    
    # 验证标准化过程
    print("\n验证标准化过程:")
    sample_data = test_data_2d[:LOOKBACK]
    print(f"原始数据范围: [{sample_data.min():.6f}, {sample_data.max():.6f}]")

    sample_norm = scaler.transform(sample_data.reshape(-1, sample_data.shape[-1])).reshape(sample_data.shape)
    print(f"标准化后范围: [{sample_norm.min():.6f}, {sample_norm.max():.6f}]")

    sample_restored = scaler.inverse_transform(sample_norm.reshape(-1, sample_norm.shape[-1])).reshape(sample_norm.shape)
    print(f"反标准化后范围: [{sample_restored.min():.6f}, {sample_restored.max():.6f}]")

    # 检查标准化-反标准化是否可逆
    mse = np.mean((sample_data - sample_restored)**2)
    print(f"标准化-反标准化MSE: {mse:.10f}")
    
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
        transformer_pred = predict_transformer(model, scaler, history_data, HORIZON, device)
        transformer_preds.append(transformer_pred)
        
        # 打印进度
        if (i + 1) % 500 == 0:
            print(f"  已完成 {i+1}/{num_test_predictions} 次预测...")
            
            # 可选：添加一些诊断信息
            if i == 0:  # 只在第一次迭代时显示详细诊断
                print("第一次预测的详细诊断:")
                predict_with_diagnostics(model, scaler, history_data, HORIZON, device)

    # 将结果列表转换为Numpy数组
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
    
    # --- 【新功能】可视化预测结果 ---
    print("\n--- 正在生成预测对比图 ---")
    plot_predictions(true_vals, transformer_preds, naive_preds)