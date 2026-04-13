import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import traceback

# ==============================================================================
# --- 1. 配置 (Config) ---
# ==============================================================================

MAT_VAR_NAME_H = 'H_active_matrix'
MAT_VAR_NAME_IDX = 'active_indices'
FS = 100
BASE_PATH = 'data'
FILE_PATHS = {
    '5 m/s': os.path.join(BASE_PATH, 'ch_est_40m_5mps_autopilot_2d.mat'),
    '15 m/s': os.path.join(BASE_PATH, 'ch_est_40m_15mps_autopilot_2d.mat')
}
NPERSEG = 1024
TOTAL_SUBCARRIERS = 1024 

# ==============================================================================
# --- 2. 数据处理与绘图函数 ---
# ==============================================================================

def get_doppler_spectrum_V_FINAL(filepath, var_name_h, fs, nperseg):
    print(f"Processing: {os.path.basename(filepath)}...")
    try:
        mat_data = sio.loadmat(filepath)
    except Exception as e:
        print(f"  ERROR: Could not load .mat file: {e}")
        return None, None, 0

    if var_name_h not in mat_data:
        print(f"  ERROR: Variable '{var_name_h}' not found.")
        return None, None, 0

    H_active_matrix = mat_data[var_name_h]
    
    # --- 重建完整的 1024 子载波矩阵 (使用 NaN 填充背景) ---
    if MAT_VAR_NAME_IDX in mat_data:
        active_indices = mat_data[MAT_VAR_NAME_IDX].flatten()
        num_timesteps = H_active_matrix.shape[0]
        
        # !! 核心修改：初始化为 NaN (Not a Number) !!
        # 这样 plot_surface 会自动忽略这些点，不绘制任何东西
        H_full_matrix = np.full((num_timesteps, TOTAL_SUBCARRIERS), np.nan, dtype=np.complex128)
        
        valid_idx_mask = active_indices < TOTAL_SUBCARRIERS
        safe_indices = active_indices[valid_idx_mask]
        
        # 只有有信号的地方才填入数值
        H_full_matrix[:, safe_indices] = H_active_matrix[:, valid_idx_mask]
        
        matrix_to_plot = H_full_matrix
    else:
        print("  WARNING: 'active_indices' not found. Using packed matrix.")
        matrix_to_plot = H_active_matrix

    # --- 切片：只取前 1000 个时间步 ---
    total_timesteps = matrix_to_plot.shape[0]
    cutoff_index = min(total_timesteps, 1000)
    
    print(f"  Data Slicing: Using first {cutoff_index} time steps.")
    matrix_to_plot = matrix_to_plot[:cutoff_index, :]
    
    num_timesteps_sliced, num_sc_sliced = matrix_to_plot.shape

    # --- !! 仅针对 5m/s 数据生成详细图像 !! ---
    if '5mps' in filepath:
        print("  DEBUG: Generating Masked (NaN) 3D Surface Plot...")

        # -------------------------------------------------------
        # 3. 绘制 3D 时频幅度图 (空值透明化)
        # -------------------------------------------------------
        
        # 1. 计算绝对幅度
        Amp_abs = np.abs(matrix_to_plot)
        
        # 2. 归一化到 [0, 1] (仅基于有效值计算 max)
        # np.nanmax 会忽略 NaN 自动找到真实信号的最大值
        max_val = np.nanmax(Amp_abs)
        if max_val == 0: max_val = 1 
        
        Amp_norm = Amp_abs / max_val
        
        # 创建网格
        X_sc, Y_time = np.meshgrid(np.arange(num_sc_sliced), np.arange(num_timesteps_sliced))
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制表面
        # NaN 的区域将完全透明，只显示有数据的“彩带”
        surf = ax.plot_surface(Y_time, X_sc, Amp_norm, cmap='turbo', 
                               edgecolor='none', antialiased=False, rstride=1, cstride=1)
        
        #ax.set_title(f'Channel Amplitude (Active Subcarriers Only)\n{os.path.basename(filepath)}', fontsize=14)
        ax.set_title(f'Channel Amplitude (Active Subcarriers Only)', fontsize=14)
        ax.set_xlabel('Time (Sample Index)', fontsize=12)
        ax.set_ylabel('Frequency (Subcarrier Index 0-1023)', fontsize=12)
        ax.set_zlabel('Normalized Amplitude', fontsize=12)
        
        ax.set_zlim(0, 1.0)
        ax.view_init(elev=45, azim=-45)
        
        fig.colorbar(surf, shrink=0.5, aspect=10, label='Normalized Amplitude')
        
        # 保存图片
        plt.savefig('fig_3d_time_freq_masked.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  SUCCESS: Saved fig_3d_time_freq_masked.png")

    # --- 3. 计算多普勒谱 (保持不变) ---
    # 注意：必须用原始 H_active_matrix (不含NaN) 来计算PSD，否则welch会报错
    H_active_sliced = H_active_matrix[:cutoff_index, :]
    _, num_active_real = H_active_sliced.shape
    
    all_Pxx = []
    current_nperseg = min(nperseg, cutoff_index)
    
    if current_nperseg == 0: return None, None, 0

    for sc_index in range(num_active_real):
        h_t_single_sc = H_active_sliced[:, sc_index]
        f, Pxx = signal.welch(
            h_t_single_sc, fs=fs, nperseg=current_nperseg, noverlap=current_nperseg // 2,
            return_onesided=False, scaling='density', average='mean'
        )
        all_Pxx.append(Pxx)

    Pxx_avg = np.mean(all_Pxx, axis=0)
    return f, Pxx_avg, num_active_real

# ==============================================================================
# --- 3. 主执行逻辑 ---
# ==============================================================================

print(f"--- Starting Analysis (40m, Fs={FS}Hz) - Masked Plots ---")
plt.figure(figsize=(14, 8))

all_results = {}
max_power_all = -np.inf
median_power_all = []
num_sc_found = 0

for speed_label, filepath in FILE_PATHS.items():
    f, Pxx_avg, num_sc = get_doppler_spectrum_V_FINAL(filepath, MAT_VAR_NAME_H, FS, NPERSEG)

    if f is not None:
        if num_sc > 0: num_sc_found = num_sc
        f_shifted = np.fft.fftshift(f)
        Pxx_shifted = np.fft.fftshift(Pxx_avg)
        Pxx_db = 10 * np.log10(Pxx_shifted + 1e-20)
        all_results[speed_label] = (f_shifted, Pxx_db)
        plt.plot(f_shifted, Pxx_db, label=f'Max Speed: {speed_label}')
        max_power_all = max(max_power_all, np.max(Pxx_db))
        median_power_all.append(np.median(Pxx_db))

if not all_results:
    print("\nNo data processed.")
else:
    print("\nPlotting Doppler results...")
    plt.title(f'Doppler Power Spectrum (First 1000 Samples, Fs={FS} Hz)', fontsize=16)
    plt.xlabel('Doppler Frequency (Hz)', fontsize=12)
    plt.ylabel('Average PSD (dB/Hz)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    if median_power_all:
        noise_floor_level = np.mean(median_power_all) - 10
        plt.ylim(noise_floor_level, max_power_all + 5)

    plt.tight_layout()
    png_filename_doppler = 'fig_doppler_impact_masked.png'
    plt.savefig(png_filename_doppler, dpi=300, bbox_inches='tight')
    print(f"  SUCCESS: Saved {png_filename_doppler}")
    plt.close()

print("\nAnalysis complete.")