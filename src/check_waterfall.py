import os
import re
import numpy as np
import matplotlib.pyplot as plt

def plot_channel_waterfall(file_path):
    print(f"📡 正在对 {os.path.basename(file_path)} 进行 X光瀑布图扫描...")
    
    header_pattern = re.compile(r"SRS Frame .* Real:")
    sc_pattern = re.compile(r"Sc (\d+): Re = (-?\d+), Im = (-?\d+)")
    
    power_matrix = []
    current_frame = np.zeros(1024)
    frame_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if header_pattern.search(line):
                if frame_count > 0:
                    power_matrix.append(current_frame)
                current_frame = np.zeros(1024)
                frame_count += 1
            else:
                m = sc_pattern.search(line)
                if m:
                    idx = int(m.group(1))
                    if idx < 1024:
                        re_val = float(m.group(2))
                        im_val = float(m.group(3))
                        current_frame[idx] = re_val**2 + im_val**2
                        
        # 保存最后收集的一帧
        if frame_count > 0:
            power_matrix.append(current_frame)
            
    # 转换为 Numpy 矩阵: 形状 (时间帧数, 子载波数)
    power_matrix = np.array(power_matrix)
    print(f"✅ 数据矩阵构建完成，尺寸: {power_matrix.shape}")
    
    # 转换为 dB
    power_matrix_db = 10 * np.log10(power_matrix + 1e-9)
    
    # 🔪 屏蔽极度高亮的 0Hz 直流峰值和保护带，防止它们破坏整张图的对比度
    min_db = np.min(power_matrix_db)
    power_matrix_db[:, 0:5] = min_db
    power_matrix_db[:, 270:718] = min_db
    power_matrix_db[:, 1019:1024] = min_db
    
    # 画图
    plt.figure(figsize=(12, 8))
    
    # 使用 50% 到 99.5% 的分位数来拉伸对比度，让微弱的真实信号无处遁形
    vmin = np.percentile(power_matrix_db, 50)
    vmax = np.percentile(power_matrix_db, 99.5)
    
    # 画出热力图 (使用 jet 伪彩色，红色代表高能量，蓝色代表冷底噪)
    plt.imshow(power_matrix_db, aspect='auto', origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
    
    plt.colorbar(label='Power (dB)')
    plt.title('Subcarrier Power Waterfall (X-Ray of Raw Physics Data)', fontsize=14, fontweight='bold')
    plt.xlabel('Subcarrier Index (0 to 1023)', fontsize=12)
    plt.ylabel('Time (Frame Index)', fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 请确保这里的路径指向你那份 150803 巡航数据的真实位置
    base_dir = os.getcwd()
    test_file = os.path.join(base_dir, 'data', 'channel_estimates_20250917_150803_bolek_50m_10mps_5ms_autopilot.txt')
    
    if os.path.exists(test_file):
        plot_channel_waterfall(test_file)
    else:
        print(f"❌ 找不到文件: {test_file}，请检查路径是否正确。")