import torch
import torch.nn as nn
import numpy as np
import joblib
import math
import scipy.io
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. 基础配置 (保持不变)
# =============================================================================
LOOKBACK = 10
HORIZON = 5 
NUM_SUBCARRIERS = 576
TIME_STEP_INTERVAL_S = 0.010
ALLOCATED_SYMBOLS = 12; ALLOCATED_RBS = 50; DMRS_PER_RB = 12; OH_PER_RB = 0

# 3GPP Tables
MCS_TABLE_38_214_5_1_3_1_2 = {0: {'Qm': 2, 'R_x1024': 120}, 1: {'Qm': 2, 'R_x1024': 193}, 2: {'Qm': 2, 'R_x1024': 308}, 3: {'Qm': 2, 'R_x1024': 449}, 4: {'Qm': 2, 'R_x1024': 602}, 5: {'Qm': 4, 'R_x1024': 378}, 6: {'Qm': 4, 'R_x1024': 434}, 7: {'Qm': 4, 'R_x1024': 490}, 8: {'Qm': 4, 'R_x1024': 553}, 9: {'Qm': 4, 'R_x1024': 616}, 10: {'Qm': 4, 'R_x1024': 658}, 11: {'Qm': 6, 'R_x1024': 466}, 12: {'Qm': 6, 'R_x1024': 517}, 13: {'Qm': 6, 'R_x1024': 567}, 14: {'Qm': 6, 'R_x1024': 616}, 15: {'Qm': 6, 'R_x1024': 666}, 16: {'Qm': 6, 'R_x1024': 719}, 17: {'Qm': 6, 'R_x1024': 772}, 18: {'Qm': 6, 'R_x1024': 822}, 19: {'Qm': 6, 'R_x1024': 873}, 20: {'Qm': 8, 'R_x1024': 682.5}, 21: {'Qm': 8, 'R_x1024': 711}, 22: {'Qm': 8, 'R_x1024': 754}, 23: {'Qm': 8, 'R_x1024': 797}, 24: {'Qm': 8, 'R_x1024': 841}, 25: {'Qm': 8, 'R_x1024': 885}, 26: {'Qm': 8, 'R_x1024': 916.5}, 27: {'Qm': 8, 'R_x1024': 948}, 28: {'Qm': 2, 'R_x1024': 0}, 29: {'Qm': 4, 'R_x1024': 0}, 30: {'Qm': 6, 'R_x1024': 0}, 31: {'Qm': 8, 'R_x1024': 0}}
SNR_TO_CQI_TABLE = {15: 22.7, 14: 21.1, 13: 19.1, 12: 17.4, 11: 15.4, 10: 13.5, 9: 11.5, 8: 9.6, 7: 7.8, 6: 5.9, 5: 4.1, 4: 2.3, 3: 0.2, 2: -2.1, 1: -4.3}
SORTED_SNR_TO_CQI = sorted(SNR_TO_CQI_TABLE.items(), key=lambda item: item[1], reverse=True)
CQI_TO_MCS_TABLE = {0: None, 1: 0, 2: 2, 3: 4, 4: 6, 5: 8, 6: 10, 7: 12, 8: 14, 9: 16, 10: 18, 11: 20, 12: 22, 13: 24, 14: 26, 15: 27}

# =============================================================================
# 2. 模型定义 (严格对应 uploaded file)
# =============================================================================

# --- Transformer (保持不变) ---
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

# --- GRU (严格复制自 uploaded file) ---
class GRUNet_Seq2Seq(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, drop_prob: float = 0.2):
        super(GRUNet_Seq2Seq, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, h = self.gru(x, h)
        out = self.fc(out)
        return out, h

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

# --- 预测辅助函数 ---
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

def predict_gru_autoregressive(model, x_norm_tensor, horizon, device):
    # 完全复刻 uploaded file 中的 run_inference 逻辑
    model.eval()
    with torch.no_grad():
        # 1. 初始化隐藏状态
        h = model.init_hidden(x_norm_tensor.size(0), device)
        
        # 2. 处理历史序列 (Context)
        _, h = model(x_norm_tensor, h)
        
        # 3. 自回归预测 (Horizon)
        decoder_input = x_norm_tensor[:, -1:, :] # 历史的最后一步作为起始
        predictions = []
        for _ in range(horizon):
            out_step, h = model(decoder_input, h)
            predictions.append(out_step)
            decoder_input = out_step # 当前输出作为下一步输入
            
        return torch.cat(predictions, dim=1) 

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
# 3. 核心仿真引擎 (Performance Replay)
# =============================================================================
class PerformanceReplayEngine:
    def __init__(self, h_real, sp_np_real, trans_model, gru_model, scaler, device):
        self.h_real = h_real
        self.sp_np_real = sp_np_real
        self.trans_model = trans_model
        self.gru_model = gru_model
        self.scaler = scaler
        self.device = device
        self.results = {
            'reactive': {'throughputs': [], 'errors': []}, 
            'transformer': {'throughputs': [], 'errors': []}, 
            'gru': {'throughputs': [], 'errors': []}, 
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
            if snr_db >= min_snr: cqi = c; break
        mcs_index = CQI_TO_MCS_TABLE.get(cqi)
        if mcs_index is None: return None, None, None, cqi
        mcs_params = MCS_TABLE_38_214_5_1_3_1_2.get(mcs_index)
        if mcs_params is None or mcs_params['R_x1024'] == 0: return None, None, None, cqi
        return mcs_index, mcs_params['Qm'], mcs_params['R_x1024'], cqi

    def _is_tx_successful(self, chosen_cqi, actual_snr_db):
        if chosen_cqi == 0: return True
        return actual_snr_db >= SNR_TO_CQI_TABLE.get(chosen_cqi, 99)

    def run_simulation(self):
        print(">>> 正在运行含 GRU 对比的链路级仿真...")
        
        # 使用最佳展示区间
        start_t = 2330
        limit_steps = 1000
        
        data_len = self.h_real.shape[0]
        max_end_t = data_len - HORIZON
        end_t = min(max_end_t, start_t + limit_steps)
        print(f"仿真范围: t=[{start_t}, {end_t})")

        for t in range(start_t, end_t):
            if (t-start_t) % 100 == 0: print(f"Processing step {t-start_t}/{end_t-start_t}...", end='\r')
            
            actual_t = t + HORIZON
            sp_real_current, np_real_current = self.sp_np_real[actual_t]
            snr_real_current = self._sp_np_to_snr_db(sp_real_current, np_real_current) - 2 
            self.results['real_snr'].append(snr_real_current)
            
            # --- 1. Oracle ---
            _, qm_opt, r_opt, _ = self._select_mcs_params(snr_real_current)
            optimal_tbs = calculate_tbs(qm_opt, r_opt, ALLOCATED_RBS, ALLOCATED_SYMBOLS, DMRS_PER_RB, OH_PER_RB)
            self.results['optimal_throughputs'].append(optimal_tbs / (1e6 * TIME_STEP_INTERVAL_S))
            
            # --- 2. Reactive ---
            sp_reactive, np_reactive = self.sp_np_real[t]
            snr_reactive_input = self._sp_np_to_snr_db(sp_reactive, np_reactive) - 2
            _, qm_r, r_r, cqi_r = self._select_mcs_params(snr_reactive_input)
            if self._is_tx_successful(cqi_r, snr_real_current):
                tbs = calculate_tbs(qm_r, r_r, ALLOCATED_RBS, ALLOCATED_SYMBOLS, DMRS_PER_RB, OH_PER_RB)
                self.results['reactive']['throughputs'].append(tbs / (1e6 * TIME_STEP_INTERVAL_S))
                self.results['reactive']['errors'].append(0)
            else:
                self.results['reactive']['throughputs'].append(0)
                self.results['reactive']['errors'].append(1)
            
            # --- 准备预测输入 ---
            history_start = t - LOOKBACK; history_end = t
            history_complex = self.h_real[history_start:history_end]
            history_mag = np.sqrt(history_complex[:,:,0]**2 + history_complex[:,:,1]**2)
            history_norm = self.scaler.transform(history_mag)
            src = torch.from_numpy(history_norm).float().unsqueeze(0).to(self.device)

            # --- 3. Transformer ---
            if self.trans_model:
                pred_tensor = predict_transformer_autoregressive(self.trans_model, src, HORIZON, self.device)
                pred_mag = self.scaler.inverse_transform(pred_tensor[:, HORIZON-1, :].cpu().numpy()).squeeze()
                pred_pwr = self._predicted_mag_to_signal_power(pred_mag)
                snr_in = self._sp_np_to_snr_db(pred_pwr, self.sp_np_real[t, 1]) - 2
                _, qm, r, cqi = self._select_mcs_params(snr_in)
                if self._is_tx_successful(cqi, snr_real_current):
                    tbs = calculate_tbs(qm, r, ALLOCATED_RBS, ALLOCATED_SYMBOLS, DMRS_PER_RB, OH_PER_RB)
                    self.results['transformer']['throughputs'].append(tbs / (1e6 * TIME_STEP_INTERVAL_S))
                    self.results['transformer']['errors'].append(0)
                else:
                    self.results['transformer']['throughputs'].append(0)
                    self.results['transformer']['errors'].append(1)

            # --- 4. GRU (Baseline) ---
            if self.gru_model:
                pred_tensor_gru = predict_gru_autoregressive(self.gru_model, src, HORIZON, self.device)
                pred_mag_gru = self.scaler.inverse_transform(pred_tensor_gru[:, HORIZON-1, :].cpu().numpy()).squeeze()
                pred_pwr_gru = self._predicted_mag_to_signal_power(pred_mag_gru)
                snr_in_gru = self._sp_np_to_snr_db(pred_pwr_gru, self.sp_np_real[t, 1]) - 2
                _, qm_g, r_g, cqi_g = self._select_mcs_params(snr_in_gru)
                if self._is_tx_successful(cqi_g, snr_real_current):
                    tbs = calculate_tbs(qm_g, r_g, ALLOCATED_RBS, ALLOCATED_SYMBOLS, DMRS_PER_RB, OH_PER_RB)
                    self.results['gru']['throughputs'].append(tbs / (1e6 * TIME_STEP_INTERVAL_S))
                    self.results['gru']['errors'].append(0)
                else:
                    self.results['gru']['throughputs'].append(0)
                    self.results['gru']['errors'].append(1)

        print("\n仿真完成。")
        return self.results

# =============================================================================
# 4. 绘图模块 (带 GRU，纯净CDF + 双Y轴SNR)
# =============================================================================
def generate_paper_results(results, output_dir='paper_results'):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    tp_r = np.array(results['reactive']['throughputs'])
    tp_t = np.array(results['transformer']['throughputs'])
    tp_g = np.array(results['gru']['throughputs']) if results['gru']['throughputs'] else None
    tp_opt = np.array(results['optimal_throughputs'])
    snr_history = np.array(results['real_snr']) 
    
    # 统计
    avg_r = np.mean(tp_r); bler_r = np.mean(results['reactive']['errors']) * 100
    avg_t = np.mean(tp_t); bler_t = np.mean(results['transformer']['errors']) * 100
    avg_g = 0; bler_g = 0
    if tp_g is not None:
        avg_g = np.mean(tp_g); bler_g = np.mean(results['gru']['errors']) * 100

    print(f"\nStats | Avg TP (Mbps) | BLER (%)")
    print(f"React | {avg_r:.2f} | {bler_r:.2f}")
    if tp_g is not None: print(f"GRU   | {avg_g:.2f} | {bler_g:.2f}")
    print(f"Trans | {avg_t:.2f} | {bler_t:.2f}")
    print(f"Optim | {np.mean(tp_opt):.2f} | 0.00")

    # --- 绘图1: 时序图 (双Y轴) ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    window = 200; start = 400
    if start + window > len(tp_r): start = 0
    x = range(window)
    
    ax1.set_xlabel('Time Steps (10ms)'); ax1.set_ylabel('Throughput (Mbps)', color='black')
    l1, = ax1.plot(x, tp_opt[start:start+window], '--', color='gray', label='Oracle', alpha=0.5, lw=1)
    l2, = ax1.plot(x, tp_r[start:start+window], 'o-', color='orange', label='Reactive', markersize=3, alpha=0.8)
    l_g = None
    if tp_g is not None:
        l_g, = ax1.plot(x, tp_g[start:start+window], 's-', color='green', label='GRU Baseline', markersize=3, alpha=0.7)
    l3, = ax1.plot(x, tp_t[start:start+window], '*-', color='blue', label='Transformer', markersize=3, lw=1.5)
    
    ax1.set_ylim(-0.2, 5.5)
    ax2 = ax1.twinx(); ax2.set_ylabel('SNR (dB)', color='red')
    snr_seg = snr_history[start:start+window]
    l4, = ax2.plot(x, snr_seg, '-', color='red', label='SNR', alpha=0.25, lw=2, zorder=0)
    ax2.set_ylim(np.min(snr_seg)-5, np.max(snr_seg)+10)
    
    lines = [l1, l2, l3, l4]
    if l_g: lines.insert(2, l_g)
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/throughput_timeseries_gru.png', dpi=300)

    # --- 绘图2: CDF (纯净版) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    def get_cdf(d):
        s = np.sort(d); y = np.arange(len(s))/float(len(s)-1)
        return s, y
    
    xr, yr = get_cdf(tp_r); xt, yt = get_cdf(tp_t); xo, yo = get_cdf(tp_opt)
    ax.plot(xo, yo, '--', color='gray', label='Oracle', alpha=0.6)
    ax.plot(xr, yr, label='Reactive Baseline', color='blue', lw=2)
    if tp_g is not None:
        xg, yg = get_cdf(tp_g)
        ax.plot(xg, yg, label='GRU Baseline', color='green', lw=2, linestyle='-.')
    ax.plot(xt, yt, label='Proposed Transformer', color='orange', lw=2)
    
    ax.set_xlabel('Throughput (Mbps)'); ax.set_ylabel('CDF')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/throughput_cdf_gru.png', dpi=300)
    print("Figures saved.")

# =============================================================================
# 5. 主程序 (加载模型)
# =============================================================================
if __name__ == "__main__":
    # --- 配置路径 ---
    TRANS_PATH = f'results/Transformer_lookback{LOOKBACK}_horizon{HORIZON}_best_model_magnitude_bolek_para2.pth'
    SCALER_PATH = f'results/Transformer_lookback{LOOKBACK}_horizon{HORIZON}_scaler_magnitude_bolek_para2.gz'
    # !!! 请在这里填入您的 GRU 模型路径 !!!
    GRU_PATH = 'results/GRU_lookback10_horizon5_best_model_magnitude_bolek_para2.pth' 
    
    DATA_H = 'data/my_H_real_data_bolek_for_final_demo.mat'
    DATA_SP = 'data/my_SP_NP_real_data_bolek_for_final_demo.npy'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load(SCALER_PATH)

    # 加载 Transformer
    trans_model = TimeSeriesTransformer(NUM_SUBCARRIERS, 512, 2, 2, 16, 0.2, 0.2, 0.1, 1024, 1024, LOOKBACK+HORIZON, HORIZON, HORIZON)
    trans_model.load_state_dict(torch.load(TRANS_PATH, map_location=device))
    trans_model.to(device)

    # 加载 GRU (根据 uploaded file 设置参数: 512, 2 layers)
    gru_model = None
    if os.path.exists(GRU_PATH):
        try:
            gru_model = GRUNet_Seq2Seq(input_dim=NUM_SUBCARRIERS, hidden_dim=512, output_dim=NUM_SUBCARRIERS, n_layers=2, drop_prob=0.2)
            gru_model.load_state_dict(torch.load(GRU_PATH, map_location=device))
            gru_model.to(device)
            print("✅ GRU 模型加载成功")
        except Exception as e:
            print(f"❌ GRU 模型加载失败: {e}")
            gru_model = None
    else:
        print(f"⚠️ 未找到 GRU 模型文件: {GRU_PATH}，将跳过 GRU 对比。")

    # 加载数据
    mat = scipy.io.loadmat(DATA_H); h_data = mat['H'].flatten().astype(np.float32).reshape(-1, NUM_SUBCARRIERS, 2)
    sp_np = np.load(DATA_SP)
    is_valid = np.any(h_data != 0, axis=(1, 2)); h_clean = h_data[is_valid]
    min_len = min(len(h_clean), len(sp_np)); h_clean = h_clean[:min_len]; sp_np = sp_np[:min_len]

    # 运行
    engine = PerformanceReplayEngine(h_clean, sp_np, trans_model, gru_model, scaler, device)
    res = engine.run_simulation()
    generate_paper_results(res)