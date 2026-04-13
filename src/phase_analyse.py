import scipy.io
import numpy as np

# Load Data
path = 'data/ch_est_60m_10mps_autopilot_2d.mat'
mat = scipy.io.loadmat(path)
H = mat['H_active_matrix'] # (Time, Sc)

# 取前 10000 个点，第 0 个子载波
h_t = H[:-1, 0]
h_next = H[1:, 0]

# 计算相关系数 rho
# rho = E[h_t * conj(h_next)] / sqrt(E[|h_t|^2] * E[|h_next|^2])
numerator = np.mean(h_t * np.conj(h_next))
denominator = np.sqrt(np.mean(np.abs(h_t)**2) * np.mean(np.abs(h_next)**2))
rho = np.abs(numerator / denominator)

print(f"========================================")
print(f"相邻时间步相关系数 (Correlation Coefficient): {rho:.4f}")
print(f"========================================")

if rho < 0.5:
    print("结论: 极低相关性。采样间隔太大，或移动速度太快。")
    print("      这是一个物理层面的'欠采样'问题，任何 AI 模型都无法预测。")
elif rho > 0.9:
    print("结论: 高相关性。数据是可预测的，模型代码可能有其他 Bug。")
else:
    print("结论: 中等相关性。预测很难，但应该能比 0 dB 好一点。")