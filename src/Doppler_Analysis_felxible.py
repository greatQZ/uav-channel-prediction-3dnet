import numpy as np
import scipy.io as sio  # 用于加载 .mat 文件
import scipy.signal as signal # 用于 Welch 方法
import matplotlib.pyplot as plt
import os # 用于文件路径操作
import traceback # 导入以进行更详细的错误追踪

# ==============================================================================
# --- 1. 通用配置 ---
# ==============================================================================
MAT_VAR_NAME_H = 'H_active_matrix' # V_FINAL 脚本保存的变量名
NPERSEG = 1024 # Welch 方法的 FFT 段长度

# ==============================================================================
# --- 2. 核心 "分析任务" 定义 ---
#
#    在这里定义您所有的实验。
#    您可以添加任意多个任务。
#
# ==============================================================================
ANALYSIS_TASKS = {
    
    # --- 任务 1: 对比 40m 高度的 5m/s 和 15m/s (100Hz 采样) ---
    '40m_compare': {
        'type': 'doppler', # 任务类型: 'doppler' 或 'time_domain'
        'description': 'Doppler: 5m/s vs 15m/s at 40m altitude',
        'fs': 100, # <-- 关键: 10ms 采样 = 100Hz
        'base_path': 'data', # 包含 .mat 文件的目录
        'files_to_compare': {
            # --- !! 请务必检查这些文件名是否正确 !! ---
            '5 m/s': 'ch_est_40m_5mps_autopilot_2d.mat',
            '15 m/s': 'ch_est_40m_15mps_autopilot_2d.mat'
        },
        'plot_title': 'Doppler Spectrum at 40m (Fs=100Hz)'
    },
# --- !! 在这里添加新任务 !! ---
    '60m_compare': {
        'type': 'doppler',
        'description': 'Doppler: 10m/s vs 15m/s at 60m altitude',
        'fs': 100, # <-- 60m 也是 10ms 采样 (100Hz)
        'base_path': 'data',
        'files_to_compare': {
            # --- !! 请务必检查这些文件名是否正确 !! ---
            '10 m/s': 'ch_est_60m_10mps_autopilot_2d.mat',
            '15 m/s': 'ch_est_60m_15mps_autopilot_2d.mat'
        },
        'plot_title': 'Doppler Spectrum at 60m (Fs=100Hz)'
    },
    # --- !! 添加结束 !! ---    
    # --- 任务 2: 对比 50m 高度的 5m/s 和 10m/s (200Hz 采样) ---
    '50m_compare': {
        'type': 'doppler',
        'description': 'Doppler: 5m/s vs 10m/s at 50m altitude',
        'fs': 200, # <-- 关键: 5ms 采样 = 200Hz
        'base_path': 'data',
        'files_to_compare': {
            '5 m/s': 'ch_est_50m_5mps_5ms_autopilot_2d.mat',
            '10 m/s': 'ch_est_50m_10mps_5ms_autopilot_2d.mat'
        },
        'plot_title': 'Doppler Spectrum at 50m (Fs=200Hz)'
    },
    
    # --- 任务 3: 绘制 50m, 5m/s 的时域图 (200Hz 采样) ---
    'time_domain_50m_5mps': {
        'type': 'time_domain',
        'description': 'Time Domain plot for 50m, 5m/s',
        'fs': 200, # 这里的 fs 仅用于信息展示，不参与计算
        'base_path': 'data',
        'file_to_plot': 'ch_est_50m_5mps_5ms_autopilot_2d.mat'
    }
}

# ==============================================================================
# --- 3. 选择您想运行的任务 ---
# ==============================================================================

# --- !! 更改这一行来选择您的分析 !! ---
CHOSEN_TASK_NAME = '40m_compare'
#CHOSEN_TASK_NAME = '60m_compare'
# CHOSEN_TASK_NAME = '50m_compare'
# CHOSEN_TASK_NAME = 'time_domain_50m_5mps'

# ==============================================================================
# --- 4. 辅助函数 (无需修改) ---
# ==============================================================================

def load_mat_file(filepath, var_name):
    """安全地加载 .mat 文件并提取变量。"""
    try:
        mat_data = sio.loadmat(filepath)
    except FileNotFoundError:
        print(f"  ERROR: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"  ERROR: Could not load .mat file: {e}")
        traceback.print_exc()
        return None
        
    if var_name not in mat_data:
        print(f"  ERROR: Variable '{var_name}' not found in {filepath}.")
        return None
        
    H_matrix = mat_data[var_name]
    
    if not np.iscomplexobj(H_matrix):
         print(f"  ERROR: Data in '{var_name}' is not complex.")
         return None
    if H_matrix.ndim != 2:
        print(f"  ERROR: '{var_name}' is not a 2D matrix. Shape is {H_matrix.shape}")
        return None
    
    return H_matrix

def run_doppler_comparison(task_config):
    """执行多普勒谱对比任务。"""
    print(f"\n--- Running Doppler Task: {task_config['description']} ---")
    
    FS = task_config['fs']
    BASE_PATH = task_config['base_path']
    PLOT_TITLE = task_config['plot_title']
    
    plt.figure(figsize=(14, 8))
    all_results = {}
    max_power_all = -np.inf
    median_power_all = []
    num_sc_found = 0

    for speed_label, filename in task_config['files_to_compare'].items():
        filepath = os.path.join(BASE_PATH, filename)
        print(f"Processing: {filename} (Fs={FS}Hz)")
        
        H_active_matrix = load_mat_file(filepath, MAT_VAR_NAME_H)
        if H_active_matrix is None:
            continue

        num_timesteps, num_active_sc = H_active_matrix.shape
        num_sc_found = num_active_sc
        print(f"  Loaded complex matrix with shape: {H_active_matrix.shape}")

        all_Pxx = []
        current_nperseg = min(NPERSEG, num_timesteps)
        if num_timesteps < NPERSEG:
            print(f"  WARNING: Not enough data ({num_timesteps} frames). Using nperseg={num_timesteps}.")
        
        if current_nperseg == 0:
            print("  ERROR: No timesteps to process.")
            continue

        for sc_index in range(num_active_sc):
            h_t_single_sc = H_active_matrix[:, sc_index]
            f, Pxx = signal.welch(
                h_t_single_sc, fs=FS, nperseg=current_nperseg, noverlap=current_nperseg // 2,
                return_onesided=False, scaling='density', average='mean'
            )
            all_Pxx.append(Pxx)
            
        Pxx_avg = np.mean(all_Pxx, axis=0)
        
        f_shifted = np.fft.fftshift(f)
        Pxx_shifted = np.fft.fftshift(Pxx_avg)
        Pxx_db = 10 * np.log10(Pxx_shifted + 1e-20) 
        
        all_results[speed_label] = (f_shifted, Pxx_db)
        plt.plot(f_shifted, Pxx_db, label=f'Max Speed: {speed_label}')
        max_power_all = max(max_power_all, np.max(Pxx_db))
        median_power_all.append(np.median(Pxx_db))

    # --- 格式化绘图 ---
    if not all_results:
        print("\nNo data was successfully processed. Cannot generate plot.")
    else:
        print("\nPlotting results...")
        plt.title(f'{PLOT_TITLE} ({num_sc_found} Active SCs)', fontsize=16)
        plt.xlabel('Doppler Frequency (Hz)', fontsize=12)
        plt.ylabel('Average Power Spectral Density (dB/Hz)', fontsize=12)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        # 自动设置 Y 轴范围
        if median_power_all:
            noise_floor_level = np.mean(median_power_all) - 10 
            plt.ylim(noise_floor_level, max_power_all + 5) 
        
        # 自动设置 X 轴范围 (例如 Fs 的 +/- 40%)
        plt.xlim(-FS * 0.4, FS * 0.4)
        
        plt.tight_layout()
        plt.show()

def run_time_domain_plot(task_config):
    """执行时域特征绘图任务。"""
    print(f"\n--- Running Time-Domain Task: {task_config['description']} ---")
    
    BASE_PATH = task_config['base_path']
    filename = task_config['file_to_plot']
    filepath = os.path.join(BASE_PATH, filename)
    
    H_active_matrix = load_mat_file(filepath, MAT_VAR_NAME_H)
    if H_active_matrix is None:
        return

    print(f"  Loaded {filename}, shape: {H_active_matrix.shape}")
    
    # 提取第一个活跃子载波的时间序列
    h_t_single_sc = H_active_matrix[:, 0]
    
    # --- 图 (a): 幅度 ---
    amplitude_db = 20 * np.log10(np.abs(h_t_single_sc) + 1e-9)
    plt.figure(figsize=(12, 6))
    plt.plot(amplitude_db[:2000], label='Amplitude (dB)')
    plt.title(f'Time Domain Amplitude (First 2000 Samples) - {filename}')
    plt.xlabel('Sample Index (Time)')
    plt.ylabel('Amplitude (dB)')
    plt.grid(True)
    plt.legend()
    plt.show() # 显示幅度图
    # !! 提示：请将此图另存为 fig_time_amplitude.png !!

    # --- 图 (b): 相位 ---
    phase_rad = np.angle(h_t_single_sc)
    phase_unwrapped = np.unwrap(phase_rad) # 解环绕
    plt.figure(figsize=(12, 6))
    plt.plot(phase_unwrapped[:2000], label='Unwrapped Phase', color='orange')
    plt.title(f'Time Domain Phase (First 2000 Samples) - {filename}')
    plt.xlabel('Sample Index (Time)')
    plt.ylabel('Unwrapped Phase (radians)')
    plt.grid(True)
    plt.legend()
    plt.show() # 显示相位图
    # !! 提示：请将此图另存为 fig_time_phase.png !!

# ==============================================================================
# --- 5. 主执行逻辑 ---
# ==============================================================================

if __name__ == "__main__":
    if CHOSEN_TASK_NAME not in ANALYSIS_TASKS:
        print(f"Error: Task '{CHOSEN_TASK_NAME}' not defined in ANALYSIS_TASKS.")
    else:
        task = ANALYSIS_TASKS[CHOSEN_TASK_NAME]
        
        if task['type'] == 'doppler':
            run_doppler_comparison(task)
        elif task['type'] == 'time_domain':
            run_time_domain_plot(task)
        else:
            print(f"Error: Unknown task type '{task['type']}'")
            
    print("\nAnalysis complete.")