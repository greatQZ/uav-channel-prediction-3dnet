# final_demo.py
# 最终版: 融合Transformer(H=5, L=10), 使用真实SP/NP, 支持长序列滑动窗口可视化
# v3: 根据 lookback=10, horizon=5 的成功实验结果进行适配和修正
# v6: [修正] 补回遗漏的 calculate_tbs 函数，并整合样式表、差异填充和错误标记
# v_gemini: [新增] 增加“理论最优吞吐量”和“信道利用率”的计算与可视化。

import torch
import torch.nn as nn
import numpy as np
import joblib
import math
import scipy.io
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation

# 应用样式表来美化图表
plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
# 阶段一：模型定义
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
# 阶段二：配置、规则和仿真参数
# =============================================================================
VIEW_WINDOW_SIZE = 500
ALLOCATED_SYMBOLS = 12
ALLOCATED_RBS = 50
DMRS_PER_RB = 12
OH_PER_RB = 0
LOOKBACK = 10
HORIZON = 10
NUM_SUBCARRIERS = 576
TIME_STEP_INTERVAL_S = 0.010

MCS_TABLE_38_214_5_1_3_1_2 = {0: {'Qm': 2, 'R_x1024': 120}, 1: {'Qm': 2, 'R_x1024': 193}, 2: {'Qm': 2, 'R_x1024': 308}, 3: {'Qm': 2, 'R_x1024': 449}, 4: {'Qm': 2, 'R_x1024': 602}, 5: {'Qm': 4, 'R_x1024': 378}, 6: {'Qm': 4, 'R_x1024': 434}, 7: {'Qm': 4, 'R_x1024': 490}, 8: {'Qm': 4, 'R_x1024': 553}, 9: {'Qm': 4, 'R_x1024': 616}, 10: {'Qm': 4, 'R_x1024': 658}, 11: {'Qm': 6, 'R_x1024': 466}, 12: {'Qm': 6, 'R_x1024': 517}, 13: {'Qm': 6, 'R_x1024': 567}, 14: {'Qm': 6, 'R_x1024': 616}, 15: {'Qm': 6, 'R_x1024': 666}, 16: {'Qm': 6, 'R_x1024': 719}, 17: {'Qm': 6, 'R_x1024': 772}, 18: {'Qm': 6, 'R_x1024': 822}, 19: {'Qm': 6, 'R_x1024': 873}, 20: {'Qm': 8, 'R_x1024': 682.5}, 21: {'Qm': 8, 'R_x1024': 711}, 22: {'Qm': 8, 'R_x1024': 754}, 23: {'Qm': 8, 'R_x1024': 797}, 24: {'Qm': 8, 'R_x1024': 841}, 25: {'Qm': 8, 'R_x1024': 885}, 26: {'Qm': 8, 'R_x1024': 916.5}, 27: {'Qm': 8, 'R_x1024': 948}, 28: {'Qm': 2, 'R_x1024': 0}, 29: {'Qm': 4, 'R_x1024': 0}, 30: {'Qm': 6, 'R_x1024': 0}, 31: {'Qm': 8, 'R_x1024': 0}}
SNR_TO_CQI_TABLE = {15: 22.7, 14: 21.1, 13: 19.1, 12: 17.4, 11: 15.4, 10: 13.5, 9: 11.5, 8: 9.6, 7: 7.8, 6: 5.9, 5: 4.1, 4: 2.3, 3: 0.2, 2: -2.1, 1: -4.3}
SORTED_SNR_TO_CQI = sorted(SNR_TO_CQI_TABLE.items(), key=lambda item: item[1], reverse=True)
CQI_TO_MCS_TABLE = {0: None, 1: 0, 2: 2, 3: 4, 4: 6, 5: 8, 6: 10, 7: 12, 8: 14, 9: 16, 10: 18, 11: 20, 12: 22, 13: 24, 14: 26, 15: 27}

# =============================================================================
# 阶段三：核心仿真引擎
# =============================================================================
# 【修正】补回遗漏的 calculate_tbs 函数定义
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

class PerformanceReplayEngine:
    def __init__(self, h_real, sp_np_real, model, scaler, device):
        self.h_real = h_real; self.sp_np_real = sp_np_real; self.model = model; self.scaler = scaler; self.device = device
        # <<< 修改：为理论最优吞吐量创建存储列表 >>>
        self.results = {'reactive': {'throughputs': [], 'errors': []}, 
                        'transformer': {'throughputs': [], 'errors': []}, 
                        'real_snr': [],
                        'optimal_throughputs': []}
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
        print("正在基于真实的SP/NP测量值运行仿真...")
        start_t = LOOKBACK
        for t in range(start_t, self.h_real.shape[0] - HORIZON):
            actual_t = t + HORIZON
            sp_real_current, np_real_current = self.sp_np_real[actual_t]
            snr_real_current = self._sp_np_to_snr_db(sp_real_current, np_real_current) - 5######
            self.results['real_snr'].append(snr_real_current)
            
            # <<< 新增：计算理论最优（Oracle）吞吐量 >>>
            _, qm_opt, r_opt, _ = self._select_mcs_params(snr_real_current)
            optimal_tbs = calculate_tbs(qm_opt, r_opt, ALLOCATED_RBS, ALLOCATED_SYMBOLS, DMRS_PER_RB, OH_PER_RB)
            self.results['optimal_throughputs'].append(optimal_tbs / (1e6 * TIME_STEP_INTERVAL_S))
            
            # Reactive system
            sp_reactive, np_reactive = self.sp_np_real[t]
            snr_reactive_input = self._sp_np_to_snr_db(sp_reactive, np_reactive) - 5######
            _, qm_r, r_r, cqi_r = self._select_mcs_params(snr_reactive_input)
            if self._is_tx_successful(cqi_r, snr_real_current):
                tbs = calculate_tbs(qm_r, r_r, ALLOCATED_RBS, ALLOCATED_SYMBOLS, DMRS_PER_RB, OH_PER_RB)
                self.results['reactive']['throughputs'].append(tbs / (1e6 * TIME_STEP_INTERVAL_S)); self.results['reactive']['errors'].append(0)
            else:
                self.results['reactive']['throughputs'].append(0); self.results['reactive']['errors'].append(1)
            
            # Transformer System
            history_start = t - LOOKBACK
            history_end = t
            history_complex = self.h_real[history_start:history_end]
            history_mag = np.sqrt(history_complex[:,:,0]**2 + history_complex[:,:,1]**2)
            history_norm = self.scaler.transform(history_mag)
            src = torch.from_numpy(history_norm).float().unsqueeze(0).to(self.device)
            prediction_norm_tensor = predict_transformer_autoregressive(self.model, src, HORIZON, self.device)
            pred_step_norm = prediction_norm_tensor[:, HORIZON - 1, :].cpu().numpy()
            pred_step_mag = self.scaler.inverse_transform(pred_step_norm).squeeze()
            pred_signal_power = self._predicted_mag_to_signal_power(pred_step_mag)
            predicted_noise_power = self.sp_np_real[t, 1]
            snr_transformer_input = self._sp_np_to_snr_db(pred_signal_power, predicted_noise_power)-5 #######
            _, qm_t, r_t, cqi_t = self._select_mcs_params(snr_transformer_input)
            if self._is_tx_successful(cqi_t, snr_real_current):
                tbs = calculate_tbs(qm_t, r_t, ALLOCATED_RBS, ALLOCATED_SYMBOLS, DMRS_PER_RB, OH_PER_RB)
                self.results['transformer']['throughputs'].append(tbs / (1e6 * TIME_STEP_INTERVAL_S)); self.results['transformer']['errors'].append(0)
            else:
                self.results['transformer']['throughputs'].append(0); self.results['transformer']['errors'].append(1)
        return self.results

# =============================================================================
# 阶段四：动态可视化
# =============================================================================
def run_visualization(sim_results):
    print("正在启动可视化...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('Performance Replay Engine: Traditional Reactive System vs. Transformer-based Predictive System', fontsize=20)

    # --- 子图1: 瞬时吞吐量 ---
    ax1.set_title('Real-time Throughput'); ax1.set_ylabel('Data Rate (Mbps)')
    
    # <<< 修改：调整zorder并添加line_optimal >>>
    line_reactive, = ax1.plot([], [], 'o-', label=f'Reactive System (t-{HORIZON} delay)', color='orange', lw=2, zorder=2)
    line_transformer, = ax1.plot([], [], '*-', label=f'Transformer System (H={HORIZON} Predict)', color='blue', lw=2, zorder=3)
    
    # <<< 新增：理论最优吞吐量的线对象 >>>
    line_optimal, = ax1.plot([], [], '--', label='Optimal Throughput (Oracle)', color='gray', lw=1.5, zorder=1)

    max_tp = max(sim_results['optimal_throughputs']) if sim_results['optimal_throughputs'] else 1
    ax1.set_ylim(0, max_tp * 1.15 if max_tp > 0 else 10)
    
    # <<< 核心修改：更新文本模板以加入“利用率” >>>
    text_template = 'System: {}\nTotal Data: {:.1f} Mb\nTotal Errors: {}\nUtilization: {:.1f}%'
    
    text_reactive = ax1.text(0.98, 0.25, '', transform=ax1.transAxes, fontsize=14, verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='orange', alpha=0.4), zorder=10)
    text_transformer = ax1.text(0.98, 0.05, '', transform=ax1.transAxes, fontsize=14, verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='blue', alpha=0.4), zorder=10)
    
    # 为错误标记创建图线对象
    error_marker_r, = ax1.plot([], [], 'x', color='red', markersize=10, label='Reactive Error', alpha=0.8, mew=2.5, zorder=5) # mew是标记边框宽度
    error_marker_t, = ax1.plot([], [], 'x', color='darkviolet', markersize=10, label='Transformer Error', alpha=0.8, mew=2.5, zorder=5)
    ax1.legend()
    # <<< 修改：将图例移到绘图区域的右侧 >>>
    #ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    # --- 子图2: 信道质量 (SNR) ---
    ax2.set_title('Ground Truth Channel Quality'); ax2.set_ylabel('SNR (dB)'); ax2.set_xlabel('Time Step')
    real_snr_history = sim_results['real_snr']
    line_real_snr, = ax2.plot([], [], label='Ground Truth SNR', color='red', alpha=0.5)
    time_marker = ax2.axvline(x=0, color='black', linestyle='--', lw=2)
    ax2.set_xlim(0, VIEW_WINDOW_SIZE); ax2.set_ylim(min(real_snr_history) - 5, max(real_snr_history) + 5); ax2.legend()

    def update(frame):
        # <<< 核心修改：计算累积数据和利用率 >>>
        total_data_r = np.sum(sim_results['reactive']['throughputs'][:frame+1]) * TIME_STEP_INTERVAL_S
        total_err_r = np.sum(sim_results['reactive']['errors'][:frame+1])
        
        total_data_t = np.sum(sim_results['transformer']['throughputs'][:frame+1]) * TIME_STEP_INTERVAL_S
        total_err_t = np.sum(sim_results['transformer']['errors'][:frame+1])
        
        total_optimal_data = np.sum(sim_results['optimal_throughputs'][:frame+1]) * TIME_STEP_INTERVAL_S
        
        utilization_r = (total_data_r / total_optimal_data * 100) if total_optimal_data > 0 else 0
        utilization_t = (total_data_t / total_optimal_data * 100) if total_optimal_data > 0 else 0

        text_reactive.set_text(text_template.format(f"Reactive (t-{HORIZON})", total_data_r, total_err_r, utilization_r))
        text_transformer.set_text(text_template.format("Transformer", total_data_t, total_err_t, utilization_t))
        
        start_index = max(0, frame + 1 - VIEW_WINDOW_SIZE)
        x_data_window = range(start_index, frame + 1)
        
        # 更新吞吐量曲线
        reactive_throughputs_window = sim_results['reactive']['throughputs'][start_index:frame+1]
        transformer_throughputs_window = sim_results['transformer']['throughputs'][start_index:frame+1]
        line_reactive.set_data(x_data_window, reactive_throughputs_window)
        line_transformer.set_data(x_data_window, transformer_throughputs_window)
        
        # <<< 新增：更新最优吞吐量曲线 >>>
        line_optimal.set_data(x_data_window, sim_results['optimal_throughputs'][start_index:frame+1])

        # 更新填充区域
        # 为了防止区域重叠，先清除上一步的填充
        for collection in ax1.collections:
            collection.remove()
        ax1.fill_between(x_data_window, reactive_throughputs_window, transformer_throughputs_window, 
                         where=np.array(transformer_throughputs_window) > np.array(reactive_throughputs_window), 
                         facecolor='green', alpha=0.3, interpolate=True, zorder=0)
        ax1.fill_between(x_data_window, reactive_throughputs_window, transformer_throughputs_window, 
                         where=np.array(reactive_throughputs_window) >= np.array(transformer_throughputs_window), 
                         facecolor='red', alpha=0.3, interpolate=True, zorder=0)

        # 更新错误标记
        error_indices_r = np.where(np.array(sim_results['reactive']['errors'][start_index:frame+1]) == 1)[0]
        error_indices_t = np.where(np.array(sim_results['transformer']['errors'][start_index:frame+1]) == 1)[0]
        x_errors_r = [start_index + i for i in error_indices_r]
        x_errors_t = [start_index + i for i in error_indices_t]
        y_errors = [0] * len(x_errors_r) # 错误的 y 坐标为 0
        error_marker_r.set_data(x_errors_r, y_errors)
        error_marker_t.set_data(x_errors_t, [0] * len(x_errors_t))

        # 更新SNR曲线
        line_real_snr.set_data(x_data_window, real_snr_history[start_index:frame+1])
        time_marker.set_xdata([frame, frame])
        
        # 滑动X轴
        if frame >= VIEW_WINDOW_SIZE:
            ax1.set_xlim(frame - VIEW_WINDOW_SIZE + 1, frame + 10)
            ax2.set_xlim(frame - VIEW_WINDOW_SIZE + 1, frame + 10)
            
        # <<< 修改：在返回列表中加入 line_optimal >>>
        return line_reactive, line_transformer, line_optimal, line_real_snr, time_marker, text_reactive, text_transformer, error_marker_r, error_marker_t

    # 将blit设置为False以兼容fill_between功能
    ani = animation.FuncAnimation(fig, update, frames=len(real_snr_history), blit=False, interval=20, repeat=False)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()
    # <<< 修改：调整布局，为右侧的图例留出15%的空间 >>>
    #plt.tight_layout(rect=[0, 0, 0.85, 0.96]) # rect=[left, bottom, right, top]
    #plt.show()

# =============================================================================
# 阶段五：主程序入口
# =============================================================================
if __name__ == "__main__":
    MODEL_PATH = f'results/Transformer_lookback{LOOKBACK}_horizon{HORIZON}_best_model_magnitude_bolek_para2.pth'
    SCALER_PATH = f'results/Transformer_lookback{LOOKBACK}_horizon{HORIZON}_scaler_magnitude_bolek_para2.gz'
    DATA_PATH_H = 'data/my_H_real_data_bolek_for_final_demo.mat'
    DATA_PATH_SP_NP = 'data/my_SP_NP_real_data_bolek_for_final_demo.npy'

    print("--- 正在加载模型和Scaler ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"使用设备: {device}")
    try:
        scaler = joblib.load(SCALER_PATH); print(f"Scaler已从 {SCALER_PATH} 加载。")
        model = TimeSeriesTransformer(
            input_size=NUM_SUBCARRIERS, 
            dim_val=512,
            n_encoder_layers=2, 
            n_decoder_layers=2, 
            n_heads=16, 
            dropout_encoder=0.2,
            dropout_decoder=0.2, 
            dropout_pos_enc=0.1, 
            dim_feedforward_encoder=1024,
            dim_feedforward_decoder=1024, 
            max_seq_len=LOOKBACK + HORIZON,
            dec_seq_len=HORIZON,
            out_seq_len=HORIZON
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device)); model.to(device); print(f"模型已从 {MODEL_PATH} 加载。")
    except FileNotFoundError as e:
        print(f"错误: 未找到模型或Scaler文件: {e}"); exit()
    except RuntimeError as e:
        print(f"错误: 加载模型失败，可能是模型结构不匹配: {e}")
        print(f"请确认当前脚本中的 LOOKBACK ({LOOKBACK}) 和 HORIZON ({HORIZON}) 与您训练时所用的完全一致。")
        exit()

    print(f"\n--- 正在加载并清洗测量数据 ---")
    try:
        mat = scipy.io.loadmat(DATA_PATH_H)
        data_1d = mat['H'].flatten().astype(np.float32)
        h_with_padding = data_1d.reshape(-1, NUM_SUBCARRIERS, 2)
        print(f"成功从 {DATA_PATH_H} 加载了 {h_with_padding.shape[0]} 个时间步的H数据（含padding）。")
        sp_np_real_data = np.load(DATA_PATH_SP_NP)
        print(f"成功从 {DATA_PATH_SP_NP} 加载了 {len(sp_np_real_data)} 个时间步的SP/NP数据。")
        is_valid_step = np.any(h_with_padding != 0, axis=(1, 2))
        h_real_clean = h_with_padding[is_valid_step]
        print(f"已从H数据中剔除padding，剩余 {h_real_clean.shape[0]} 个有效时间步。")
        if len(h_real_clean) != len(sp_np_real_data):
            print(f"警告：清洗后的H数据长度({len(h_real_clean)})与SP/NP数据长度({len(sp_np_real_data)})不匹配！")
            min_len = min(len(h_real_clean), len(sp_np_real_data))
            h_real_clean = h_real_clean[:min_len]; sp_np_real_data = sp_np_real_data[:min_len]
            print(f"已将数据统一截断为较短的长度: {min_len}")
    except FileNotFoundError as e:
        print(f"错误: 未找到数据文件: {e}。程序将退出。"); exit()
    
    engine = PerformanceReplayEngine(h_real_clean, sp_np_real_data, model, scaler, device)
    simulation_results = engine.run_simulation()
    run_visualization(simulation_results)