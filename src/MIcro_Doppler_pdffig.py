# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
import os

# ==============================================================================
# --- 1. 核心配置 ---
# ==============================================================================
# 文件路径配置
BASE_PATH = 'data'
# 修改为 5m/s 的数据文件路径
FILE_PATH_5MPS = os.path.join(BASE_PATH, 'ch_est_40m_5mps_autopilot_2d.mat') 
MAT_VAR_NAME = 'H_active_matrix'

# 物理参数
FS = 100        # 采样率 100Hz (10ms interval)
VELOCITY = 5.0  # 修改为速度 5 m/s
FC = 3619.2e6   # 载波频率 3.6192 GHz
C = 3e8         # 光速

# 绘图参数 (论文发表级标准)
FONT_TITLE = 22
FONT_LABEL = 18
FONT_TICK  = 16
FONT_CBAR  = 16

def plot_spectrogram_5mps_clean():
    """
    生成 5 m/s 的干净时频图，不带任何理论虚线。
    用于展示即使在低速下，相位噪声依然是主要的谱扩展来源。
    """
    print(f">>> 正在加载数据: {FILE_PATH_5MPS}")
    
    try:
        # 1. 加载数据
        if not os.path.exists(FILE_PATH_5MPS):
            raise FileNotFoundError(f"找不到文件: {FILE_PATH_5MPS}")
            
        mat_data = sio.loadmat(FILE_PATH_5MPS)
        H = mat_data[MAT_VAR_NAME] # Shape: (Time, Subcarriers)
        
        # 2. 计算平均时频图 (STFT)
        num_sc = min(H.shape[1], 100) # 取前100个子载波
        sxx_accum = None
        
        nperseg = 256
        noverlap = 220
        
        for i in range(num_sc):
            f, t_stft, Zxx = signal.stft(H[:, i], fs=FS, window='hann', 
                                       nperseg=nperseg, noverlap=noverlap, 
                                       return_onesided=False)
            
            Sxx = np.abs(Zxx)**2
            
            if sxx_accum is None:
                sxx_accum = Sxx
            else:
                sxx_accum += Sxx
                
        avg_Sxx = sxx_accum / num_sc
        avg_Sxx = np.fft.fftshift(avg_Sxx, axes=0) 
        f_shifted = np.fft.fftshift(f)
        
        avg_Sxx_db = 10 * np.log10(avg_Sxx + 1e-12)
        
        # 3. 空间映射 (Time -> Distance)
        d = t_stft * VELOCITY
        
        # 4. 绘图
        plt.figure(figsize=(10, 6))
        
        vmin_val = np.percentile(avg_Sxx_db, 5) 
        vmax_val = np.max(avg_Sxx_db)
        
        pm = plt.pcolormesh(d, f_shifted, avg_Sxx_db, shading='gouraud', 
                           cmap='inferno', vmin=vmin_val, vmax=vmax_val)
        
        # --- 视觉修饰 ---
        # 标题更新为 5 m/s
        plt.title('Micro-Doppler Spectrogram (5 m/s)', fontsize=FONT_TITLE, fontweight='bold', pad=15)
        plt.xlabel('Distance Traveled (m)', fontsize=FONT_LABEL, fontweight='bold')
        plt.ylabel('Doppler Frequency (Hz)', fontsize=FONT_LABEL, fontweight='bold')
        
        plt.ylim(-50, 50)
        plt.xlim(0, 200)
        
        plt.tick_params(axis='both', labelsize=FONT_TICK)
        plt.grid(True, linestyle='--', alpha=0.3, color='white')
        
        cbar = plt.colorbar(pm)
        cbar.set_label('Power Spectral Density (dB/Hz)', fontsize=FONT_LABEL)
        cbar.ax.tick_params(labelsize=FONT_CBAR)
        
        plt.tight_layout()
        
        # 5. 保存结果
        output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)
        # 更新保存文件名
        save_path = os.path.join(output_dir, 'Spectrogram_5mps_Clean.pdf')
        
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n>>> 绘图完成！")
        print(f">>> 文件已保存至: {save_path}")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    plot_spectrogram_5mps_clean()