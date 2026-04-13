import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
import os

# ==============================================================================
# --- 配置 ---
# ==============================================================================
MAT_VAR_NAME_H = 'H_active_matrix'
FS = 200
BASE_PATH = 'data'
FILE_PATHS = {
    '5 m/s': os.path.join(BASE_PATH, 'ch_est_50m_5mps_5ms_autopilot_2d.mat'),
    '10 m/s': os.path.join(BASE_PATH, 'ch_est_50m_10mps_5ms_autopilot_2d.mat')
}
NPERSEG = 256 
NOVERLAP = 220 

# --- 关键设置：空间对齐 ---
# 我们以 5m/s 飞行的前 T_SLOW_SEC 秒为基准距离
T_SLOW_SEC = 60 
DISTANCE_LIMIT = 4 * T_SLOW_SEC # 300 米

def plot_spatial_spectrogram():
    plt.figure(figsize=(12, 10))
    
    plot_idx = 1
    # 硬编码速度值以便计算距离
    speeds = {'5 m/s': 5.0, '10 m/s': 10.0}
    
    for speed_label, filepath in FILE_PATHS.items():
        velocity = speeds[speed_label]
        print(f"Processing: {speed_label} (Velocity={velocity} m/s)...")
        
        try:
            mat_data = sio.loadmat(filepath)
            if MAT_VAR_NAME_H not in mat_data: continue
            H = mat_data[MAT_VAR_NAME_H]
            
            # 计算需要的样本数 = (距离 / 速度) * 采样率
            required_time = DISTANCE_LIMIT / velocity
            limit_samples = int(required_time * FS)
            
            # 截取数据
            if H.shape[0] < limit_samples:
                print(f"Warning: Not enough data for {DISTANCE_LIMIT}m. Using all available.")
                limit_samples = H.shape[0]
            
            H = H[:limit_samples, :]
            
            # STFT 处理 (取前100个子载波平均)
            num_sc = H.shape[1]
            sc_to_process = range(0, min(num_sc, 100))
            
            total_Sxx = None
            for i in sc_to_process:
                f, t, Zxx = signal.stft(H[:, i], fs=FS, window='hann', 
                                        nperseg=NPERSEG, noverlap=NOVERLAP, 
                                        return_onesided=False)
                Sxx = np.abs(Zxx)**2
                if total_Sxx is None: total_Sxx = Sxx
                else: total_Sxx += Sxx
            
            avg_Sxx = total_Sxx / len(sc_to_process)
            avg_Sxx_db = 10 * np.log10(avg_Sxx + 1e-12)
            
            avg_Sxx_shifted = np.fft.fftshift(avg_Sxx_db, axes=0)
            f_shifted = np.fft.fftshift(f)
            
            # --- 关键步骤：将时间轴转换为距离轴 ---
            # Distance = Time * Velocity
            d = t * velocity
            
            # --- 绘图 ---
            plt.subplot(2, 1, plot_idx)
            
            # 统一色阶以便对比
            vmin_val = np.percentile(avg_Sxx_shifted, 5)
            vmax_val = np.max(avg_Sxx_shifted)
            
            plt.pcolormesh(d, f_shifted, avg_Sxx_shifted, shading='gouraud', 
                           cmap='inferno', vmin=vmin_val, vmax=vmax_val)
            
            plt.title(f'Spatial Spectrogram - {speed_label} (First {DISTANCE_LIMIT}m)', fontsize=14, fontweight='bold')
            plt.ylabel('Doppler (Hz)', fontsize=12)
            
            # 只在最下面的图显示X轴标签
            if plot_idx == 2: 
                plt.xlabel('Distance Traveled (m)', fontsize=12)
            
            # 限制显示范围为 0 到 300米
            plt.xlim(0, DISTANCE_LIMIT)
            plt.ylim(-40, 40) 
            
            cbar = plt.colorbar()
            cbar.set_label('dB/Hz', rotation=270, labelpad=15)
            
            plot_idx += 1
            
        except Exception as e:
            print(f"Error: {e}")

    plt.tight_layout()
    filename = 'fig_spectrogram_spatial_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved spatial comparison plot to {filename}")

# 运行
plot_spatial_spectrogram()