import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# =============================================================================
# 阶段一：配置、规则和仿真参数 (不变)
# =============================================================================
NUM_TIMESTEPS = 200
NUM_SUBCARRIERS = 576
NOISE_POWER_DBM = -95
ALLOCATED_SYMBOLS = 12
ALLOCATED_RBS = 50
DMRS_PER_RB = 12
OH_PER_RB = 0

MCS_TABLE_38_214_5_1_3_1_2 = {
    0: {'Qm': 2, 'R_x1024': 120}, 1: {'Qm': 2, 'R_x1024': 193}, 2: {'Qm': 2, 'R_x1024': 308},
    3: {'Qm': 2, 'R_x1024': 449}, 4: {'Qm': 2, 'R_x1024': 602}, 5: {'Qm': 4, 'R_x1024': 378},
    6: {'Qm': 4, 'R_x1024': 434}, 7: {'Qm': 4, 'R_x1024': 490}, 8: {'Qm': 4, 'R_x1024': 553},
    9: {'Qm': 4, 'R_x1024': 616}, 10: {'Qm': 4, 'R_x1024': 658}, 11: {'Qm': 6, 'R_x1024': 466},
    12: {'Qm': 6, 'R_x1024': 517}, 13: {'Qm': 6, 'R_x1024': 567}, 14: {'Qm': 6, 'R_x1024': 616},
    15: {'Qm': 6, 'R_x1024': 666}, 16: {'Qm': 6, 'R_x1024': 719}, 17: {'Qm': 6, 'R_x1024': 772},
    18: {'Qm': 6, 'R_x1024': 822}, 19: {'Qm': 6, 'R_x1024': 873}, 20: {'Qm': 8, 'R_x1024': 682.5},
    21: {'Qm': 8, 'R_x1024': 711}, 22: {'Qm': 8, 'R_x1024': 754}, 23: {'Qm': 8, 'R_x1024': 797},
    24: {'Qm': 8, 'R_x1024': 841}, 25: {'Qm': 8, 'R_x1024': 885}, 26: {'Qm': 8, 'R_x1024': 916.5},
    27: {'Qm': 8, 'R_x1024': 948}, 28: {'Qm': 2, 'R_x1024': 0}, 29: {'Qm': 4, 'R_x1024': 0},
    30: {'Qm': 6, 'R_x1024': 0}, 31: {'Qm': 8, 'R_x1024': 0}
}
SNR_TO_CQI_TABLE = {
    15: 22.7, 14: 21.1, 13: 19.1, 12: 17.4, 11: 15.4, 10: 13.5, 9: 11.5,
    8:  9.6,  7:  7.8,  6:  5.9,  5:  4.1,  4:  2.3,  3:  0.2, 2: -2.1, 1: -4.3
}
SORTED_SNR_TO_CQI = sorted(SNR_TO_CQI_TABLE.items(), key=lambda item: item[1], reverse=True)
CQI_TO_MCS_TABLE = {
    0: None, 1: 0, 2: 2, 3: 4, 4: 6, 5: 8, 6: 10, 7: 12, 8: 14,
    9: 16, 10: 18, 11: 20, 12: 22, 13: 24, 14: 26, 15: 27
}

# =============================================================================
# 阶段二：仿真数据生成 (不变)
# =============================================================================
def generate_channel_data(n_steps, n_subcarriers):
    print("正在生成仿真的信道数据...")
    channel_gain = 0.003
    time = np.linspace(0, 10, n_steps)
    channel_quality_trend = 0.5 * (np.sin(time * 1.2) + 1) + \
                            -0.4 * (1 / (1 + np.exp(-(time - 5) * 3))) * (1 - 1 / (1 + np.exp(-(time - 6.5) * 3)))
    h_real = np.zeros((n_steps, n_subcarriers), dtype=complex)
    current_h = (np.random.randn(n_subcarriers) + 1j * np.random.randn(n_subcarriers)) / np.sqrt(2)
    for t in range(n_steps):
        innovation = (np.random.randn(n_subcarriers) + 1j * np.random.randn(n_subcarriers)) / np.sqrt(2)
        current_h = 0.90 * current_h + 0.10 * innovation
        h_real[t, :] = current_h * channel_quality_trend[t] * channel_gain
    # 在这个版本中我们不再需要h_pred
    return h_real

# =============================================================================
# 阶段三：核心仿真引擎 (已根据您的要求修改)
# =============================================================================
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
    def __init__(self, h_real, noise_power_dbm):
        self.h_real = h_real
        self.noise_power_linear = 10**((noise_power_dbm - 30) / 10)
        self.results = {'reactive': {'throughputs': [], 'errors': []}, 'proactive': {'throughputs': [], 'errors': []}, 'real_snr': []}
    def _channel_to_snr_db(self, h_vector):
        signal_power_linear = np.mean(np.abs(h_vector)**2)
        if signal_power_linear < 1e-15: return -100.0
        return 10 * np.log10(signal_power_linear / self.noise_power_linear)
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
        print("正在运行仿真...")
        # <<< --- 关键改动 1: 循环从第5步开始，以确保t-5有数据 --- >>>
        for t in range(5, self.h_real.shape[0]):
            snr_real_current = self._channel_to_snr_db(self.h_real[t])
            self.results['real_snr'].append(snr_real_current)

            # --- 反应式系统 (Reactive), 延迟5步 ---
            # <<< --- 关键改动 2: 决策依据改为 t-5 时刻的SNR --- >>>
            snr_reactive_input = self._channel_to_snr_db(self.h_real[t-5])
            _, qm_r, r_r, cqi_r = self._select_mcs_params(snr_reactive_input)
            
            if self._is_tx_successful(cqi_r, snr_real_current):
                tbs_bits = calculate_tbs(qm_r, r_r, ALLOCATED_RBS, ALLOCATED_SYMBOLS, DMRS_PER_RB, OH_PER_RB)
                self.results['reactive']['throughputs'].append(tbs_bits / 1000.0)
                self.results['reactive']['errors'].append(0)
            else:
                self.results['reactive']['throughputs'].append(0); self.results['reactive']['errors'].append(1)
            
            # --- 主动式系统 (Proactive), 延迟1步 ---
            # <<< --- 关键改动 3: 决策依据改为 t-1 时刻的SNR --- >>>
            snr_proactive_input = self._channel_to_snr_db(self.h_real[t-1])
            _, qm_p, r_p, cqi_p = self._select_mcs_params(snr_proactive_input)

            if self._is_tx_successful(cqi_p, snr_real_current):
                tbs_bits = calculate_tbs(qm_p, r_p, ALLOCATED_RBS, ALLOCATED_SYMBOLS, DMRS_PER_RB, OH_PER_RB)
                self.results['proactive']['throughputs'].append(tbs_bits / 1000.0)
                self.results['proactive']['errors'].append(0)
            else:
                self.results['proactive']['throughputs'].append(0); self.results['proactive']['errors'].append(1)
        return self.results

# =============================================================================
# 阶段四：动态可视化 (不变)
# =============================================================================
def run_visualization(sim_results):
    print("正在启动可视化...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('Performance Replay Engine: Reactive vs. Proactive', fontsize=20)
    ax1.set_title('Real-time Throughput'); ax1.set_ylabel('Data Rate (Mbps)'); ax1.grid(True, linestyle='--', alpha=0.6)
    line_reactive, = ax1.plot([], [], 'o-', label='Reactive System (t-5 delay)', color='orange', lw=2)
    line_proactive, = ax1.plot([], [], 'o-', label='Proactive System (t-1 delay)', color='blue', lw=2)
    ax1.legend()
    max_throughput = max(max(sim_results['reactive']['throughputs']), max(sim_results['proactive']['throughputs'])) if sim_results['proactive']['throughputs'] else 1
    ax1.set_ylim(-5, max_throughput * 1.15 if max_throughput > 0 else 10)
    text_template = 'Total Data: {:.1f} Mb\nTotal Errors: {}'
    text_reactive = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
    text_proactive = ax1.text(0.02, 0.78, '', transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
    ax2.set_title('Ground Truth Channel Quality'); ax2.set_ylabel('SNR (dB)'); ax2.set_xlabel('Time Step'); ax2.grid(True, linestyle='--', alpha=0.6)
    real_snr_history = sim_results['real_snr']
    line_real_snr, = ax2.plot([], [], label='Ground Truth SNR', color='green', alpha=0.7)
    time_marker = ax2.axvline(x=0, color='red', linestyle='--', lw=2)
    ax2.set_xlim(0, len(real_snr_history)); ax2.set_ylim(min(real_snr_history) - 5, max(real_snr_history) + 5); ax2.legend()
    
    def update(frame):
        # 注意：x_data的范围需要与结果数据的长度匹配
        x_data = range(frame + 1)
        line_reactive.set_data(x_data, sim_results['reactive']['throughputs'][:frame+1])
        line_proactive.set_data(x_data, sim_results['proactive']['throughputs'][:frame+1])
        line_real_snr.set_data(x_data, real_snr_history[:frame+1])
        time_marker.set_xdata([frame, frame])
        
        total_data_r_mb = np.sum(sim_results['reactive']['throughputs'][:frame+1]) * 0.001
        total_err_r = np.sum(sim_results['reactive']['errors'][:frame+1])
        text_reactive.set_text(text_template.format(total_data_r_mb, total_err_r))
        
        total_data_p_mb = np.sum(sim_results['proactive']['throughputs'][:frame+1]) * 0.001
        total_err_p = np.sum(sim_results['proactive']['errors'][:frame+1])
        text_proactive.set_text(text_template.format(total_data_p_mb, total_err_p))
        return line_reactive, line_proactive, line_real_snr, time_marker, text_reactive, text_proactive
        
    ani = animation.FuncAnimation(fig, update, frames=len(real_snr_history), blit=True, interval=50, repeat=False)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

# =============================================================================
# 阶段五：主程序入口 (已修改)
# =============================================================================
if __name__ == "__main__":
    # 1. 生成数据 (我们不再需要 h_pred)
    H_real = generate_channel_data(NUM_TIMESTEPS, NUM_SUBCARRIERS)
    
    # 2. 创建并运行仿真引擎
    engine = PerformanceReplayEngine(H_real, NOISE_POWER_DBM)
    simulation_results = engine.run_simulation()
    
    # 3. 启动可视化
    run_visualization(simulation_results)