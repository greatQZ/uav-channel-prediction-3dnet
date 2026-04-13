# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
import os

# ==============================================================================
# --- 1. 核心配置 ---
# ==============================================================================
MAT_VAR_NAME_H = 'H_active_matrix'
FS = 100  # 采样率（Hz），对应 Mission 1/3 的 10ms 采样间隔
BASE_PATH = 'data'
FILE_PATHS = {
    '5 m/s': os.path.join(BASE_PATH, 'ch_est_40m_5mps_autopilot_2d.mat'),
    '15 m/s': os.path.join(BASE_PATH, 'ch_est_40m_15mps_autopilot_2d.mat')
}

# --- 论文级绘图参数 ---
FONT_TITLE = 22
FONT_LABEL = 18
FONT_TICK  = 16
FONT_LEGEND = 14

def plot_psd_comparison_final():
    """
    计算并对比 5 m/s 和 15 m/s 的 PSD，
    并标注正确的理论多普勒边界 (60.3Hz vs 181.0Hz)
    """
    plt.figure(figsize=(14, 9))
    
    # 物理常数
    fc = 3619.2e6  # 载波频率 3.6192 GHz
    c = 3e8        # 光速
    
    # 颜色配置
    colors = {'5 m/s': 'tab:blue', '15 m/s': 'tab:red'}
    
    # 存储最终显示的理论值，用于图例
    theo_values = {}

    for label, filepath in FILE_PATHS.items():
        print(f">>> 正在处理任务: {label}...")
        
        try:
            # 加载复数 CSI 数据
            mat_data = sio.loadmat(filepath)
            H = mat_data[MAT_VAR_NAME_H]
            
            # 1. 计算该速度下的理论最大多普勒频移 fD = (v/c) * fc
            # 提取速度数字
            v_max = float(label.split(' ')[0])
            f_D_theo = (v_max / c) * fc
            theo_values[label] = f_D_theo
            
            # 2. 计算平均功率谱密度 (PSD)
            # 选取前 100 个子载波进行 Welch 平均以获得平滑的硬件指纹
            num_sc = min(H.shape[1], 100)
            psd_list = []
            
            for i in range(num_sc):
                # 使用 Welch 方法计算，nperseg=256 提供较好的频率分辨率
                f, Pxx = signal.welch(H[:, i], fs=FS, window='hann', 
                                      nperseg=256, return_onesided=False)
                psd_list.append(Pxx)
            
            avg_psd = np.mean(psd_list, axis=0)
            
            # 3. 频率轴中心化与 dB 转换
            f_shifted = np.fft.fftshift(f)
            psd_db = 10 * np.log10(np.fft.fftshift(avg_psd) + 1e-12)
            
            # 4. 绘制实测 PSD 曲线
            plt.plot(f_shifted, psd_db, label=f'Measured PSD ({label})', 
                     color=colors[label], linewidth=3, zorder=3)
            
            # 5. 绘制理论多普勒边界 (垂直虚线)
            # 对应 5m/s -> 60.3Hz, 15m/s -> 181.0Hz
            plt.axvline(x=f_D_theo, color=colors[label], linestyle='--', 
                        alpha=0.7, linewidth=2.5, zorder=2,
                        label=f'Theo. Max $f_D$ at {label} ({f_D_theo:.1f} Hz)')
            plt.axvline(x=-f_D_theo, color=colors[label], linestyle='--', alpha=0.7, linewidth=2.5, zorder=2)

        except Exception as e:
            print(f"处理 {label} 时出错: {e}")

    # --- 图像修饰 ---
    plt.title('Validation of Hardware Phase Noise Dominance via PSD', fontsize=FONT_TITLE, fontweight='bold', pad=20)
    plt.xlabel('Doppler Frequency (Hz)', fontsize=FONT_LABEL)
    plt.ylabel('Power Spectral Density (dB/Hz)', fontsize=FONT_LABEL, fontweight='bold')
    
    # 关键：设置 X 轴范围
    # 即使 15m/s 的 fD 在 181Hz，我们也重点展示 0-100Hz 区域以观察谱宽的“不变性”
    plt.xlim(-200, 200) 
    plt.ylim(plt.gca().get_ylim()[0], plt.gca().get_ylim()[1] + 5) # 留出顶部空间给图例
    
    plt.xticks(np.arange(-200, 201, 50), fontsize=FONT_TICK)
    plt.yticks(fontsize=FONT_TICK)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 图例放在左上角避免遮挡曲线下降沿
    plt.legend(fontsize=FONT_LEGEND, loc='upper right', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    
    # --- 保存结果 ---
    os.makedirs('results', exist_ok=True)
    save_path = os.path.join('results', 'PSD_Comparison_Final_Corrected.pdf')
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n>>> 修正后的 PSD 对比图已成功保存至: {save_path}")
    print(f">>> 5m/s 理论极限: {theo_values['5 m/s']:.2f} Hz")
    print(f">>> 15m/s 理论极限: {theo_values['15 m/s']:.2f} Hz")

if __name__ == "__main__":
    plot_psd_comparison_final()