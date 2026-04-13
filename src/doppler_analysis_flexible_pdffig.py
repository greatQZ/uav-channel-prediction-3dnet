# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio 
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
import traceback

# ==============================================================================
# --- 1. 通用配置 ---
# ==============================================================================
MAT_VAR_NAME_H = 'H_active_matrix'
NPERSEG = 1024 

# --- 字体配置 (严格匹配上一个代码) ---
FONT_TITLE = 24
FONT_LABEL = 20
FONT_TICK  = 16
FONT_LEGEND = 14

# ==============================================================================
# --- 2. 核心分析任务定义 ---
# ==============================================================================
ANALYSIS_TASKS = {
    '40m_compare': {
        'type': 'doppler', 
        'description': 'Doppler: 5m/s vs 15m/s at 40m altitude',
        'fs': 100, 
        'base_path': 'data', 
        'files_to_compare': {
            '5 m/s': 'ch_est_40m_5mps_autopilot_2d.mat',
            '15 m/s': 'ch_est_40m_15mps_autopilot_2d.mat'
        },
        'plot_title': 'Doppler Spectrum at 40m (Fs=100Hz)'
    },
    'time_domain_50m_5mps': {
        'type': 'time_domain',
        'description': 'Time Domain plot for 50m, 5m/s',
        'fs': 200, 
        'base_path': 'data', 
        'file_to_plot': 'ch_est_50m_5mps_5ms_autopilot_2d.mat'
    },
    # 新增：如果你想看 40m 5m/s 的相位，请切换到这个任务
    'time_domain_40m_5mps': {
        'type': 'time_domain',
        'description': 'Time Domain plot for 40m_5mps',
        'fs': 100, 
        'base_path': 'data', 
        'file_to_plot': 'ch_est_40m_5mps_autopilot_2d.mat'
    }
}

# --- !! 更改这一行 !! ---
# 如果你想看相位图，请务必选一个 type 为 'time_domain' 的任务
CHOSEN_TASK_NAME = 'time_domain_40m_5mps' 

# ==============================================================================
# --- 辅助函数 ---
# ==============================================================================

def load_mat_file(filepath, var_name):
    try:
        mat_data = sio.loadmat(filepath)
        if var_name not in mat_data:
            return None
        H_matrix = mat_data[var_name]
        if not np.iscomplexobj(H_matrix):
             return None
        return H_matrix
    except:
        return None

def apply_publication_style():
    """统一应用大字体样式"""
    plt.title(plt.gca().get_title(), fontsize=FONT_TITLE, pad=20, fontweight='bold')
    plt.xlabel(plt.gca().get_xlabel(), fontsize=FONT_LABEL, fontweight='bold')
    plt.ylabel(plt.gca().get_ylabel(), fontsize=FONT_LABEL, fontweight='bold')
    plt.tick_params(axis='both', labelsize=FONT_TICK)
    if plt.gca().get_legend():
        plt.setp(plt.gca().get_legend().get_texts(), fontsize=FONT_LEGEND)

def run_doppler_comparison(task_config):
    print(f"\n--- Running Doppler Task: {task_config['description']} ---")
    FS = task_config['fs']
    plt.figure(figsize=(16, 10))
    
    max_power_all = -np.inf
    median_powers = []

    for speed_label, filename in task_config['files_to_compare'].items():
        filepath = os.path.join(task_config['base_path'], filename)
        H = load_mat_file(filepath, MAT_VAR_NAME_H)
        if H is None: continue

        # 均值 PSD 计算
        f, Pxx = signal.welch(H, fs=FS, nperseg=min(NPERSEG, len(H)), axis=0, return_onesided=False)
        Pxx_avg = np.mean(Pxx, axis=1)
        
        f_s = np.fft.fftshift(f)
        P_s = 10 * np.log10(np.fft.fftshift(Pxx_avg) + 1e-20)
        
        plt.plot(f_s, P_s, label=f'Max Speed: {speed_label}', lw=2)
        max_power_all = max(max_power_all, np.max(P_s))
        median_powers.append(np.median(P_s))

    plt.title(task_config['plot_title'])
    plt.xlabel('Doppler Frequency (Hz)')
    plt.ylabel('Average PSD (dB/Hz)')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.xlim(-FS*0.4, FS*0.4)
    if median_powers:
        plt.ylim(np.mean(median_powers)-10, max_power_all+5)
    
    apply_publication_style()
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    save_path = os.path.join('results', f"{CHOSEN_TASK_NAME}_Doppler.pdf")
    plt.savefig(save_path, format='pdf', dpi=300)
    print(f"Saved: {save_path}")
    plt.show()

def run_time_domain_plot(task_config):
    print(f"\n--- Running Time-Domain Task: {task_config['description']} ---")
    filepath = os.path.join(task_config['base_path'], task_config['file_to_plot'])
    H = load_mat_file(filepath, MAT_VAR_NAME_H)
    if H is None: return

    h_single = H[:, 0] # 取第一个子载波
    os.makedirs('results', exist_ok=True)
    limit = min(2000, len(h_single))

    # 1. 幅度图
    plt.figure(figsize=(16, 9))
    plt.plot(20 * np.log10(np.abs(h_single[:limit]) + 1e-9))
    plt.title(f"Amplitude - {task_config['description']}")
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude (dB)')
    plt.grid(True, linestyle='--', alpha=0.7)
    apply_publication_style()
    plt.savefig(os.path.join('results', f"{CHOSEN_TASK_NAME}_Amplitude.pdf"), format='pdf')
    
    # 2. 相位图
    plt.figure(figsize=(16, 9))
    plt.plot(np.unwrap(np.angle(h_single[:limit])), color='orange')
    plt.title(f"Phase - {task_config['description']}")
    plt.xlabel('Sample Index')
    plt.ylabel('Unwrapped Phase (rad)')
    plt.grid(True, linestyle='--', alpha=0.7)
    apply_publication_style()
    plt.savefig(os.path.join('results', f"{CHOSEN_TASK_NAME}_Phase.pdf"), format='pdf')
    
    print(f"Saved: Amplitude and Phase PDFs in 'results' folder.")
    plt.show() # 同时显示两张图（需关闭窗口）

if __name__ == "__main__":
    task = ANALYSIS_TASKS.get(CHOSEN_TASK_NAME)
    if task:
        if task['type'] == 'doppler':
            run_doppler_comparison(task)
        else:
            run_time_domain_plot(task)