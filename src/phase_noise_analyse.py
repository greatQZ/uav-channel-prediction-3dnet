import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 加载数据
FILE_PATH = 'MIcro_Doppler.py' # 请确保指向你的 .mat 文件
mat_data = sio.loadmat('data/ch_est_40m_5mps_autopilot_2d.mat') # 示例路径
H = mat_data['H_active_matrix']

def refined_analysis(start_idx=2000, length=500):
    # 1. 选取子载波
    sc_ref = 0
    sc_near = 1    # 相邻子载波
    sc_far = 100   # 远端子载波
    
    seg = H[start_idx:start_idx+length, :]
    
    # 2. 检查幅度 (确认不是因为 SNR 太低)
    mag = np.abs(seg[:, sc_ref])
    print(f"分析段平均幅度: {np.mean(mag):.4f}")
    
    # 3. 提取相位抖动 (Jitter)
    def get_jitter(data):
        p = np.unwrap(np.angle(data))
        # 去除趋势
        trend = np.poly1d(np.polyfit(range(length), p, 1))(range(length))
        return p - trend

    j_ref = get_jitter(seg[:, sc_ref])
    j_near = get_jitter(seg[:, sc_near])
    j_far = get_jitter(seg[:, sc_far])
    
    # 4. 计算相关性
    corr_near, _ = pearsonr(j_ref, j_near)
    corr_far, _ = pearsonr(j_ref, j_far)
    
    print(f"相邻子载波 (0 & 1) 相关系数: {corr_near:.4f}")
    print(f"远端子载波 (0 & 100) 相关系数: {corr_far:.4f}")

    # 5. 绘图观察
    plt.figure(figsize=(12, 6))
    plt.plot(j_ref, label='SC 0 Jitter')
    plt.plot(j_near, label='SC 1 Jitter (Adjacent)', alpha=0.7)
    plt.title(f"Phase Jitter Synchronization Check\nNear Corr: {corr_near:.4f}, Far Corr: {corr_far:.4f}")
    plt.legend()
    plt.show()

refined_analysis()