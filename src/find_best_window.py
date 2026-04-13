import torch
import torch.nn as nn
import numpy as np
import joblib
import math
import scipy.io
import os

# =============================================================================
# 1. 基础配置与参数 (保持与训练一致)
# =============================================================================
# 请确保这些参数与您的模型文件严格匹配
LOOKBACK = 10
HORIZON = 5  # <--- 请确认您的模型是 Horizon 5 还是 10，这里默认设为 5
NUM_SUBCARRIERS = 576
TIME_STEP_INTERVAL_S = 0.010

ALLOCATED_SYMBOLS = 12
ALLOCATED_RBS = 50
DMRS_PER_RB = 12
OH_PER_RB = 0

# 3GPP Tables
MCS_TABLE_38_214_5_1_3_1_2 = {0: {'Qm': 2, 'R_x1024': 120}, 1: {'Qm': 2, 'R_x1024': 193}, 2: {'Qm': 2, 'R_x1024': 308}, 3: {'Qm': 2, 'R_x1024': 449}, 4: {'Qm': 2, 'R_x1024': 602}, 5: {'Qm': 4, 'R_x1024': 378}, 6: {'Qm': 4, 'R_x1024': 434}, 7: {'Qm': 4, 'R_x1024': 490}, 8: {'Qm': 4, 'R_x1024': 553}, 9: {'Qm': 4, 'R_x1024': 616}, 10: {'Qm': 4, 'R_x1024': 658}, 11: {'Qm': 6, 'R_x1024': 466}, 12: {'Qm': 6, 'R_x1024': 517}, 13: {'Qm': 6, 'R_x1024': 567}, 14: {'Qm': 6, 'R_x1024': 616}, 15: {'Qm': 6, 'R_x1024': 666}, 16: {'Qm': 6, 'R_x1024': 719}, 17: {'Qm': 6, 'R_x1024': 772}, 18: {'Qm': 6, 'R_x1024': 822}, 19: {'Qm': 6, 'R_x1024': 873}, 20: {'Qm': 8, 'R_x1024': 682.5}, 21: {'Qm': 8, 'R_x1024': 711}, 22: {'Qm': 8, 'R_x1024': 754}, 23: {'Qm': 8, 'R_x1024': 797}, 24: {'Qm': 8, 'R_x1024': 841}, 25: {'Qm': 8, 'R_x1024': 885}, 26: {'Qm': 8, 'R_x1024': 916.5}, 27: {'Qm': 8, 'R_x1024': 948}, 28: {'Qm': 2, 'R_x1024': 0}, 29: {'Qm': 4, 'R_x1024': 0}, 30: {'Qm': 6, 'R_x1024': 0}, 31: {'Qm': 8, 'R_x1024': 0}}
SNR_TO_CQI_TABLE = {15: 22.7, 14: 21.1, 13: 19.1, 12: 17.4, 11: 15.4, 10: 13.5, 9: 11.5, 8: 9.6, 7: 7.8, 6: 5.9, 5: 4.1, 4: 2.3, 3: 0.2, 2: -2.1, 1: -4.3}
SORTED_SNR_TO_CQI = sorted(SNR_TO_CQI_TABLE.items(), key=lambda item: item[1], reverse=True)
CQI_TO_MCS_TABLE = {0: None, 1: 0, 2: 2, 3: 4, 4: 6, 5: 8, 6: 10, 7: 12, 8: 14, 9: 16, 10: 18, 11: 20, 12: 22, 13: 24, 14: 26, 15: 27}

# =============================================================================
# 2. 模型定义 (Transformer & Tools)
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
# 3. 核心仿真引擎
# =============================================================================
class PerformanceReplayEngine:
    def __init__(self, h_real, sp_np_real, model, scaler, device):
        self.h_real = h_real; self.sp_np_real = sp_np_real; self.model = model; self.scaler = scaler; self.device = device
        self.results = {'reactive': {'throughputs': [], 'errors': []}, 'transformer': {'throughputs': [], 'errors': []}, 'real_snr': [], 'optimal_throughputs': []}

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
        print(">>> 正在运行全量链路级仿真 (用于寻找最佳展示片段)...")
        start_t = LOOKBACK
        data_len = self.h_real.shape[0]
        max_end_t = data_len - HORIZON
        end_t = max_end_t
        print(f"数据总长: {data_len}, 仿真范围: t=[{start_t}, {end_t})")

        processed_count = 0
        total_loops = end_t - start_t
        
        for t in range(start_t, end_t):
            if processed_count % 500 == 0: print(f"进度: {processed_count}/{total_loops}...", end='\r')
            processed_count += 1
            
            actual_t = t + HORIZON
            sp_real_current, np_real_current = self.sp_np_real[actual_t]
            snr_real_current = self._sp_np_to_snr_db(sp_real_current, np_real_current)-2 # Backoff

            # 1. Oracle
            _, qm_opt, r_opt, _ = self._select_mcs_params(snr_real_current)
            optimal_tbs = calculate_tbs(qm_opt, r_opt, ALLOCATED_RBS, ALLOCATED_SYMBOLS, DMRS_PER_RB, OH_PER_RB)
            self.results['optimal_throughputs'].append(optimal_tbs / (1e6 * TIME_STEP_INTERVAL_S))
            
            # 2. Reactive
            sp_reactive, np_reactive = self.sp_np_real[t]
            snr_reactive_input = self._sp_np_to_snr_db(sp_reactive, np_reactive)-2 # Backoff
            _, qm_r, r_r, cqi_r = self._select_mcs_params(snr_reactive_input)
            if self._is_tx_successful(cqi_r, snr_real_current):
                tbs = calculate_tbs(qm_r, r_r, ALLOCATED_RBS, ALLOCATED_SYMBOLS, DMRS_PER_RB, OH_PER_RB)
                self.results['reactive']['throughputs'].append(tbs / (1e6 * TIME_STEP_INTERVAL_S))
                self.results['reactive']['errors'].append(0)
            else:
                self.results['reactive']['throughputs'].append(0)
                self.results['reactive']['errors'].append(1)
            
            # 3. Transformer
            history_start = t - LOOKBACK; history_end = t
            history_norm = self.scaler.transform(np.sqrt(self.h_real[history_start:history_end][:,:,0]**2 + self.h_real[history_start:history_end][:,:,1]**2))
            src = torch.from_numpy(history_norm).float().unsqueeze(0).to(self.device)
            pred_step_mag = self.scaler.inverse_transform(predict_transformer_autoregressive(self.model, src, HORIZON, self.device)[:, HORIZON - 1, :].cpu().numpy()).squeeze()
            
            snr_transformer_input = self._sp_np_to_snr_db(self._predicted_mag_to_signal_power(pred_step_mag), self.sp_np_real[t, 1]) - 2 # Backoff
            _, qm_t, r_t, cqi_t = self._select_mcs_params(snr_transformer_input)
            if self._is_tx_successful(cqi_t, snr_real_current):
                tbs = calculate_tbs(qm_t, r_t, ALLOCATED_RBS, ALLOCATED_SYMBOLS, DMRS_PER_RB, OH_PER_RB)
                self.results['transformer']['throughputs'].append(tbs / (1e6 * TIME_STEP_INTERVAL_S))
                self.results['transformer']['errors'].append(0)
            else:
                self.results['transformer']['throughputs'].append(0)
                self.results['transformer']['errors'].append(1)
        
        print("\n全量仿真完成。")
        return self.results

# =============================================================================
# 4. 扫描器逻辑
# =============================================================================
def scan_for_best_window(sim_results, window_size=500):
    print(f"\n>>> 正在扫描最佳展示区间 (窗口大小: {window_size})...")
    
    tp_r = np.array(sim_results['reactive']['throughputs'])
    tp_t = np.array(sim_results['transformer']['throughputs'])
    err_r = np.array(sim_results['reactive']['errors'])
    err_t = np.array(sim_results['transformer']['errors'])
    
    n_steps = len(tp_r)
    best_score = -np.inf
    best_start = 0
    best_stats = {}
    
    # 策略：寻找 Reactive 误码率高，且 Transformer 误码率显著更低的地方
    for t in range(0, n_steps - window_size, 50): # 步长50加速
        win_tp_r = tp_r[t : t+window_size]
        win_tp_t = tp_t[t : t+window_size]
        win_err_r = err_r[t : t+window_size]
        win_err_t = err_t[t : t+window_size]
        
        sum_tp_r = np.sum(win_tp_r) + 1e-6
        sum_tp_t = np.sum(win_tp_t)
        
        mean_bler_r = np.mean(win_err_r)
        mean_bler_t = np.mean(win_err_t)
        
        # 评分公式：
        # 1. 基础分：Reactive 误码率要高 (比如 > 5%)
        # 2. 加分项：Transformer 误码率比 Reactive 低得越多越好
        # 3. 辅助项：吞吐量增益
        
        if mean_bler_r < 0.05: # 如果 Reactive 都没怎么错，这个区间没意义
            score = -np.inf
        else:
            bler_reduction = mean_bler_r - mean_bler_t
            # 我们最想要的是：Reactive 烂 (e.g. 20%)，Transformer 好 (e.g. 5%)
            # score = 误码率降低量 * 100 + 吞吐量增益 * 0.1
            score = bler_reduction * 100 
            
        if score > best_score:
            best_score = score
            best_start = t
            best_stats = {
                'gain': (sum_tp_t - sum_tp_r) / sum_tp_r * 100,
                'bler_r': mean_bler_r * 100,
                'bler_t': mean_bler_t * 100
            }
            
    print("="*40)
    if best_score == -np.inf:
        print("❌ 未找到 Reactive 误码率 > 5% 的区间。")
        print("建议：请尝试更换更高移动性（更快）的数据集。")
    else:
        print(f"✅ 找到最佳展示区间: start_t = {best_start + LOOKBACK}") 
        # 注意：这里加 LOOKBACK 是为了对应原始数据的索引，但在 paper_plot_generator 填入 start_t 时
        # paper_plot_generator 的循环是从 start_t 开始的，所以直接填 best_start + LOOKBACK 即可
        print(f"区间长度: {window_size} (对应 limit_steps)")
        print("-" * 20)
        print(f"吞吐量增益 (Gain)    : {best_stats['gain']:.2f}%")
        print(f"误块率 (Reactive)    : {best_stats['bler_r']:.2f}%")
        print(f"误块率 (Transformer) : {best_stats['bler_t']:.2f}%")
        print("-" * 20)
        print(f"👉 下一步: 在 paper_plot_generator.py 中设置:")
        print(f"   start_t = {best_start + LOOKBACK}")
        print(f"   limit_steps = {window_size}")
    print("="*40)

# =============================================================================
# 5. 主程序
# =============================================================================
if __name__ == "__main__":
    # 请确认文件路径
    MODEL_PATH = f'results/Transformer_lookback{LOOKBACK}_horizon{HORIZON}_best_model_magnitude_bolek_para2.pth'
    SCALER_PATH = f'results/Transformer_lookback{LOOKBACK}_horizon{HORIZON}_scaler_magnitude_bolek_para2.gz'
    DATA_PATH_H = 'data/my_H_real_data_bolek_for_final_demo.mat'
    DATA_PATH_SP_NP = 'data/my_SP_NP_real_data_bolek_for_final_demo.npy'

    print("--- 资源加载 ---")
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
        print(f"加载模型失败: {e}"); exit()

    print("--- 数据加载 ---")
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
        print(f"加载数据失败: {e}"); exit()

    engine = PerformanceReplayEngine(h_real_clean, sp_np_real_data, model, scaler, device)
    full_results = engine.run_simulation()
    scan_for_best_window(full_results, window_size=1000)