# paper_plot_generator.py
# 专为IEEE论文生成静态图表和统计数据
# 基于: channel_predictor_demo_bolek_transformer_newUI_usage.py

import torch
import torch.nn as nn
import numpy as np
import joblib
import math
import scipy.io
import matplotlib.pyplot as plt
import os

# 设置学术绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# =============================================================================
# 1. 模型定义 (保持与您训练时一致)
# =============================================================================
class PositionalEncoder(nn.Module):
    def __init__(self, dropout: float=0.1, max_seq_len: int=5000, d_model: int=512, batch_first: bool=True):
        super().__init__()
        self.d_model = d_model; self.dropout = nn.Dropout(p=dropout); self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(self.x_dim)]; return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, dim_val, n_encoder_layers, n_decoder_layers, n_heads, dropout_encoder, 
                 dropout_decoder, dropout_pos_enc, dim_feedforward_encoder, dim_feedforward_decoder, 
                 max_seq_len, dec_seq_len, out_seq_len):
        super().__init__()
        self.dec_seq_len = dec_seq_len; self.out_seq_len = out_seq_len
        self.encoder_input_layer = nn.Linear(in_features=input_size, out_features=dim_val)
        self.decoder_input_layer = nn.Linear(in_features=input_size, out_features=dim_val)
        self.positional_encoding_layer = PositionalEncoder(d_model=dim_val, dropout=dropout_pos_enc, max_seq_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_val, nhead=n_heads, dim_feedforward=dim_feedforward_encoder, dropout=dropout_encoder, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_encoder_layers, norm=None)
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_val, nhead=n_heads, dim_feedforward=dim_feedforward_decoder, dropout=dropout_decoder, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_decoder_layers, norm=None)
        self.linear_mapping = nn.Linear(in_features=dim_val, out_features=input_size)
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        src = self.encoder_input_layer(src); src = self.positional_encoding_layer(src); src = self.encoder(src=src)
        decoder_output = self.decoder_input_layer(tgt)
        decoder_output = self.positional_encoding_layer(decoder_output)
        decoder_output = self.decoder(tgt=decoder_output, memory=src, tgt_mask=tgt_mask, memory_mask=src_mask)
        decoder_output = self.linear_mapping(decoder_output)
        return decoder_output[:, -self.out_seq_len:, :]

def generate_square_subsequent_mask(dim1: int, dim2: int) -> torch.Tensor:
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

def predict_transformer_autoregressive(model, x_norm_tensor, horizon, device):
    model.eval()
    with torch.no_grad():
        src = model.encoder_input_layer(x_norm_tensor)
        src = model.positional_encoding_layer(src)
        memory = model.encoder(src=src)
        decoder_input = x_norm_tensor[:, -1:, :]
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

# =============================================================================
# 2. 配置与参数
# =============================================================================
ALLOCATED_SYMBOLS = 12; ALLOCATED_RBS = 50; DMRS_PER_RB = 12; OH_PER_RB = 0
LOOKBACK = 10; HORIZON = 5 # 注意：根据您的文件名，horizon可能是5
NUM_SUBCARRIERS = 576; TIME_STEP_INTERVAL_S = 0.010

# 3GPP Tables (保持不变)
MCS_TABLE_38_214_5_1_3_1_2 = {0: {'Qm': 2, 'R_x1024': 120}, 1: {'Qm': 2, 'R_x1024': 193}, 2: {'Qm': 2, 'R_x1024': 308}, 3: {'Qm': 2, 'R_x1024': 449}, 4: {'Qm': 2, 'R_x1024': 602}, 5: {'Qm': 4, 'R_x1024': 378}, 6: {'Qm': 4, 'R_x1024': 434}, 7: {'Qm': 4, 'R_x1024': 490}, 8: {'Qm': 4, 'R_x1024': 553}, 9: {'Qm': 4, 'R_x1024': 616}, 10: {'Qm': 4, 'R_x1024': 658}, 11: {'Qm': 6, 'R_x1024': 466}, 12: {'Qm': 6, 'R_x1024': 517}, 13: {'Qm': 6, 'R_x1024': 567}, 14: {'Qm': 6, 'R_x1024': 616}, 15: {'Qm': 6, 'R_x1024': 666}, 16: {'Qm': 6, 'R_x1024': 719}, 17: {'Qm': 6, 'R_x1024': 772}, 18: {'Qm': 6, 'R_x1024': 822}, 19: {'Qm': 6, 'R_x1024': 873}, 20: {'Qm': 8, 'R_x1024': 682.5}, 21: {'Qm': 8, 'R_x1024': 711}, 22: {'Qm': 8, 'R_x1024': 754}, 23: {'Qm': 8, 'R_x1024': 797}, 24: {'Qm': 8, 'R_x1024': 841}, 25: {'Qm': 8, 'R_x1024': 885}, 26: {'Qm': 8, 'R_x1024': 916.5}, 27: {'Qm': 8, 'R_x1024': 948}, 28: {'Qm': 2, 'R_x1024': 0}, 29: {'Qm': 4, 'R_x1024': 0}, 30: {'Qm': 6, 'R_x1024': 0}, 31: {'Qm': 8, 'R_x1024': 0}}
SNR_TO_CQI_TABLE = {15: 22.7, 14: 21.1, 13: 19.1, 12: 17.4, 11: 15.4, 10: 13.5, 9: 11.5, 8: 9.6, 7: 7.8, 6: 5.9, 5: 4.1, 4: 2.3, 3: 0.2, 2: -2.1, 1: -4.3}
SORTED_SNR_TO_CQI = sorted(SNR_TO_CQI_TABLE.items(), key=lambda item: item[1], reverse=True)
CQI_TO_MCS_TABLE = {0: None, 1: 0, 2: 2, 3: 4, 4: 6, 5: 8, 6: 10, 7: 12, 8: 14, 9: 16, 10: 18, 11: 20, 12: 22, 13: 24, 14: 26, 15: 27}

def calculate_tbs(Qm, R_x1024, num_rb, num_symb, dmrs_per_rb, oh_per_rb, Nl=1):
    if Qm is None or num_rb == 0 or R_x1024 == 0: return 0
    nbp_re = 12 * num_symb - dmrs_per_rb - oh_per_rb
    nb_re = min(156, nbp_re) * num_rb
    Ninfo = nb_re * (R_x1024 / 1024.0) * Qm * Nl
    if Ninfo <= 3824:
        n = max(3, math.floor(math.log2(Ninfo)) - 6) if Ninfo > 0 else 3
        Np_info = max(24, (2**n) * round(Ninfo / (2**n)))
        Tbstable_nr_approx = [24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 408, 432, 456, 480, 504, 528, 552, 576, 608, 640, 672, 704, 736, 768, 808, 848, 888, 928, 984, 1032, 1064, 1128, 1160, 1192, 1224, 1256, 1288, 1320, 1352, 1416, 1480, 1544, 1608, 1672, 1736, 1800, 1864, 1928, 2024, 2088, 2152, 2216, 2280, 2408, 2472, 2536, 2600, 2664, 2728, 2792, 2856, 2976, 3104, 3240, 3368, 3496, 3624, 3752, 3824]
        for tbs_val in Tbstable_nr_approx:
            if tbs_val >= Np_info: return tbs_val
        return Tbstable_nr_approx[-1]
    else:
        n = math.log2(Ninfo - 24) - 5 if (Ninfo - 24) > 0 else 1
        Np_info = max(3840, (2**n) * round((Ninfo - 24) / (2**n)))
        if (R_x1024 / 1024.0) <= 0.25:
             C = math.ceil((Np_info + 24) / 3816); tbs = 8 * C * math.ceil((Np_info + 24) / (8 * C)) - 24
        else:
            if Np_info > 8424:
                C = math.ceil((Np_info + 24) / 8424); tbs = 8 * C * math.ceil((Np_info + 24) / (8 * C)) - 24
            else:
                tbs = 8 * math.ceil((Np_info + 24) / 8) - 24
        return tbs

# =============================================================================
# 3. 核心仿真引擎 (Performance Replay) - 完整修复版
# =============================================================================
class PerformanceReplayEngine:
    def __init__(self, h_real, sp_np_real, model, scaler, device):
        self.h_real = h_real
        self.sp_np_real = sp_np_real
        self.model = model
        self.scaler = scaler
        self.device = device
        # 初始化结果字典
        self.results = {
            'reactive': {'throughputs': [], 'errors': []}, 
            'transformer': {'throughputs': [], 'errors': []}, 
            'real_snr': [],
            'optimal_throughputs': []
        }

    def _sp_np_to_snr_db(self, sp, np_val):
        if np_val <= 0 or sp <= 0: return -100.0
        return 10 * np.log10(sp / np_val)

    def _predicted_mag_to_signal_power(self, pred_mag_vector):
        return np.mean(pred_mag_vector**2)

    def _select_mcs_params(self, snr_db):
        cqi = 0
        for c, min_snr in SORTED_SNR_TO_CQI:
            if snr_db >= min_snr: 
                cqi = c
                break
        mcs_index = CQI_TO_MCS_TABLE.get(cqi)
        if mcs_index is None: return None, None, None, cqi
        mcs_params = MCS_TABLE_38_214_5_1_3_1_2.get(mcs_index)
        if mcs_params is None or mcs_params['R_x1024'] == 0: return None, None, None, cqi
        return mcs_index, mcs_params['Qm'], mcs_params['R_x1024'], cqi

    def _is_tx_successful(self, chosen_cqi, actual_snr_db):
        if chosen_cqi == 0: return True
        # 获取该 CQI 对应的最低 SNR 要求
        required_snr = SNR_TO_CQI_TABLE.get(chosen_cqi, 99)
        return actual_snr_db >= required_snr

    def run_simulation(self):
        print(">>> 正在运行链路级仿真 (可能需要几分钟)...")
        
        start_t = 3410
        # 如果您想跑全量数据，将 limit_steps 设为 None
        # 如果只想测试一下，可以设为 1000
        limit_steps = 1000 
        
        # --- 核心修复：正确计算循环结束位置 ---
        # 数据总长度
        data_len = self.h_real.shape[0]
        
        # 能够进行预测的最后一个时间步 t
        # 因为我们需要访问 t + HORIZON，所以 t + HORIZON 必须 < data_len
        # 即 t < data_len - HORIZON
        # range 函数是左闭右开，所以 end_t = data_len - HORIZON
        max_end_t = data_len - HORIZON
        
        if limit_steps:
            end_t = min(max_end_t, start_t + limit_steps)
        else:
            end_t = max_end_t
            
        print(f"仿真范围: t=[{start_t}, {end_t}), 数据总长: {data_len}")

        # 用于进度显示
        total_loops = end_t - start_t
        processed_count = 0

        for t in range(start_t, end_t):
            # 打印进度
            if processed_count % 100 == 0:
                print(f"Processing step {processed_count}/{total_loops}...", end='\r')
            processed_count += 1

            actual_t = t + HORIZON
            
            # 获取 Ground Truth 的信噪比
            sp_real_current, np_real_current = self.sp_np_real[actual_t]
            # 注意：这里减去 5dB 是为了模拟实际系统中需要的余量 (Backoff)
            snr_real_current = self._sp_np_to_snr_db(sp_real_current, np_real_current)-2
            self.results['real_snr'].append(snr_real_current)
            
            # ==========================================
            # 1. Oracle (Optimal / Genie-Aided)
            # ==========================================
            _, qm_opt, r_opt, _ = self._select_mcs_params(snr_real_current)
            optimal_tbs = calculate_tbs(qm_opt, r_opt, ALLOCATED_RBS, ALLOCATED_SYMBOLS, DMRS_PER_RB, OH_PER_RB)
            self.results['optimal_throughputs'].append(optimal_tbs / (1e6 * TIME_STEP_INTERVAL_S))
            
            # ==========================================
            # 2. Reactive (Baseline / Outdated CSI)
            # ==========================================
            # 只能看到当前时刻 t 的 CSI (即 t-HORIZON 的陈旧信息)
            sp_reactive, np_reactive = self.sp_np_real[t]
            snr_reactive_input = self._sp_np_to_snr_db(sp_reactive, np_reactive)-2
            _, qm_r, r_r, cqi_r = self._select_mcs_params(snr_reactive_input)
            
            # 判断是否传输成功 (使用 actual_t 的真实 SNR 判定)
            if self._is_tx_successful(cqi_r, snr_real_current):
                tbs = calculate_tbs(qm_r, r_r, ALLOCATED_RBS, ALLOCATED_SYMBOLS, DMRS_PER_RB, OH_PER_RB)
                self.results['reactive']['throughputs'].append(tbs / (1e6 * TIME_STEP_INTERVAL_S))
                self.results['reactive']['errors'].append(0)
            else:
                self.results['reactive']['throughputs'].append(0) # 传输失败，吞吐量为0
                self.results['reactive']['errors'].append(1)
            
            # ==========================================
            # 3. Transformer (Proposed / Predictive)
            # ==========================================
            history_start = t - LOOKBACK
            history_end = t
            
            # 准备输入数据
            history_complex = self.h_real[history_start:history_end]
            history_mag = np.sqrt(history_complex[:,:,0]**2 + history_complex[:,:,1]**2)
            history_norm = self.scaler.transform(history_mag)
            src = torch.from_numpy(history_norm).float().unsqueeze(0).to(self.device)
            
            # 执行预测
            prediction_norm_tensor = predict_transformer_autoregressive(self.model, src, HORIZON, self.device)
            
            # 提取第 HORIZON 步的预测值 (即 t + HORIZON 时刻)
            pred_step_norm = prediction_norm_tensor[:, HORIZON - 1, :].cpu().numpy()
            pred_step_mag = self.scaler.inverse_transform(pred_step_norm).squeeze()
            
            # 计算预测的信号功率和信噪比
            pred_signal_power = self._predicted_mag_to_signal_power(pred_step_mag)
            predicted_noise_power = self.sp_np_real[t, 1] # 假设噪声功率变化不大，使用当前的噪声
            snr_transformer_input = self._sp_np_to_snr_db(pred_signal_power, predicted_noise_power)-2
            
            # 选择 MCS
            _, qm_t, r_t, cqi_t = self._select_mcs_params(snr_transformer_input)
            
            # 判断是否传输成功
            if self._is_tx_successful(cqi_t, snr_real_current):
                tbs = calculate_tbs(qm_t, r_t, ALLOCATED_RBS, ALLOCATED_SYMBOLS, DMRS_PER_RB, OH_PER_RB)
                self.results['transformer']['throughputs'].append(tbs / (1e6 * TIME_STEP_INTERVAL_S))
                self.results['transformer']['errors'].append(0)
            else:
                self.results['transformer']['throughputs'].append(0)
                self.results['transformer']['errors'].append(1)
                
        print("\nSimulation Complete.")
        return self.results

def generate_paper_results(results, output_dir='paper_results'):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # 提取数据
    tp_r = np.array(results['reactive']['throughputs'])
    tp_t = np.array(results['transformer']['throughputs'])
    tp_opt = np.array(results['optimal_throughputs'])
    snr_history = np.array(results['real_snr']) 
    
    # --- A. 计算统计指标 ---
    avg_r = np.mean(tp_r); avg_t = np.mean(tp_t); avg_opt = np.mean(tp_opt)
    safe_avg_r = avg_r if avg_r > 0 else 1e-6
    gain = (avg_t - avg_r) / safe_avg_r * 100
    
    sum_tp_opt = np.sum(tp_opt) if np.sum(tp_opt) > 0 else 1
    util_r = (np.sum(tp_r) / sum_tp_opt) * 100
    util_t = (np.sum(tp_t) / sum_tp_opt) * 100
    bler_r = np.mean(results['reactive']['errors']) * 100
    bler_t = np.mean(results['transformer']['errors']) * 100

    print("\n" + "="*40)
    print("      PAPER STATISTICAL RESULTS      ")
    print("="*40)
    print(f"Throughput Gain                 : +{gain:.2f}%")
    print(f"Block Error Rate (Transformer)  : {bler_t:.2f}% (vs {bler_r:.2f}%)")
    print("="*40 + "\n")

    # --- B. 绘图1: 时序快照 (Snapshot) - 双Y轴版 (左吞吐量，右SNR) ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    window = 200 
    start = 50 
    # 自动调整 start 以防越界
    if start + window > len(tp_r): start = 0
    x = range(window)
    
    # 左轴: 吞吐量
    ax1.set_xlabel('Time Steps (10ms interval)')
    ax1.set_ylabel('Throughput (Mbps)', color='black')
    l1, = ax1.plot(x, tp_opt[start:start+window], '--', color='gray', label='Oracle (Upper Bound)', alpha=0.5, linewidth=1)
    l2, = ax1.plot(x, tp_r[start:start+window], 'o-', color='orange', label='Reactive Baseline', markersize=3, alpha=0.8)
    l3, = ax1.plot(x, tp_t[start:start+window], '*-', color='blue', label='Proposed Transformer', markersize=3, linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(-0.2, 5.5) 

    # 右轴: SNR
    ax2 = ax1.twinx()
    ax2.set_ylabel('SNR (dB)', color='red')
    snr_segment = snr_history[start:start+window]
    l4, = ax2.plot(x, snr_segment, '-', color='red', label='Channel SNR', alpha=0.25, linewidth=2, zorder=0)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(np.min(snr_segment) - 5, np.max(snr_segment) + 10)

    # 合并图例
    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/throughput_timeseries.png', dpi=300)
    print(f"Figure saved: {output_dir}/throughput_timeseries.png")

    # --- C. 绘图2: CDF (纯净版 - 无放大) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    def get_cdf_data(data):
        sorted_data = np.sort(data)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        return sorted_data, yvals
    
    x_opt, y_opt = get_cdf_data(tp_opt)
    x_r, y_r = get_cdf_data(tp_r)
    x_t, y_t = get_cdf_data(tp_t)
    
    # 绘制主曲线
    ax.plot(x_opt, y_opt, '--', color='gray', label='Oracle (Upper Bound)', alpha=0.6, linewidth=1.5)
    ax.plot(x_r, y_r, label='Reactive Baseline', color='orange', linewidth=2)
    ax.plot(x_t, y_t, label='Proposed Transformer', color='blue', linewidth=2)
    
    ax.set_xlabel('Throughput (Mbps)')
    ax.set_ylabel('CDF')
    # ax.set_title('Cumulative Distribution of Throughput') # 论文插图通常不需要标题
    
    # 图例放右下角
    ax.legend(loc='upper left', frameon=True, fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/throughput_cdf.pdf')
    plt.savefig(f'{output_dir}/throughput_cdf.png', dpi=300)
    print(f"Figure saved: {output_dir}/throughput_cdf.png (纯净版)")
# =============================================================================
# 5. 主程序入口
# =============================================================================
if __name__ == "__main__":
    # 请确保路径与您的文件名一致
    MODEL_PATH = f'results/Transformer_lookback{LOOKBACK}_horizon{HORIZON}_best_model_magnitude_bolek_para2.pth'
    SCALER_PATH = f'results/Transformer_lookback{LOOKBACK}_horizon{HORIZON}_scaler_magnitude_bolek_para2.gz'
    DATA_PATH_H = 'data/my_H_real_data_bolek_for_final_demo.mat'
    DATA_PATH_SP_NP = 'data/my_SP_NP_real_data_bolek_for_final_demo.npy'

    print("--- Loading Resources ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        scaler = joblib.load(SCALER_PATH)
        model = TimeSeriesTransformer(
            input_size=NUM_SUBCARRIERS, dim_val=512, n_encoder_layers=2, n_decoder_layers=2, n_heads=16, 
            dropout_encoder=0.2, dropout_decoder=0.2, dropout_pos_enc=0.1, 
            dim_feedforward_encoder=1024, dim_feedforward_decoder=1024, 
            max_seq_len=LOOKBACK + HORIZON, dec_seq_len=HORIZON, out_seq_len=HORIZON
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
    except Exception as e:
        print(f"Error loading model/scaler: {e}"); exit()

    print("--- Loading Data ---")
    try:
        mat = scipy.io.loadmat(DATA_PATH_H)
        data_1d = mat['H'].flatten().astype(np.float32)
        h_with_padding = data_1d.reshape(-1, NUM_SUBCARRIERS, 2)
        sp_np_real_data = np.load(DATA_PATH_SP_NP)
        is_valid_step = np.any(h_with_padding != 0, axis=(1, 2))
        h_real_clean = h_with_padding[is_valid_step]
        if len(h_real_clean) != len(sp_np_real_data):
            min_len = min(len(h_real_clean), len(sp_np_real_data))
            h_real_clean = h_real_clean[:min_len]; sp_np_real_data = sp_np_real_data[:min_len]
    except Exception as e:
        print(f"Error loading data: {e}"); exit()

    # 运行引擎
    engine = PerformanceReplayEngine(h_real_clean, sp_np_real_data, model, scaler, device)
    sim_results = engine.run_simulation()
    
    # 生成论文图表
    generate_paper_results(sim_results)