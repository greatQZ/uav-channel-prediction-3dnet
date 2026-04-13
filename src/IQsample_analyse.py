import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import random # 导入 random 模块

def analyze_channel_correlation():
    """
    读取信道数据.mat文件，并生成信道轨迹图和自相关函数图，
    以分析信道的时间相关性。
    """
    # --- 1. 参数配置 (请根据您的项目结构进行调整) ---
    print("--- 1. 配置参数 ---")
    
    # 要分析的子载波索引 (0 到 575 之间)
    SUBCARRIER_TO_PLOT = 0
    
    # 用于分析的最小数据块长度
    MIN_BLOCK_LENGTH = 50
    
    # .mat 数据文件路径
    try:
        # 尝试使用 __file__ 来定位项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, 'data')
        combined_data_file = os.path.join(data_dir, 'my_H_clean_combined_padded_optimized_test_autopilot_40m_for_test.mat')
    except NameError:
        # 如果在交互式环境（如Jupyter）中运行，__file__ 可能未定义
        print("警告: 无法自动确定项目路径，将使用相对路径。请确保数据文件位置正确。")
        # 假设脚本在 src 目录，数据在与 src 同级的 data 目录
        combined_data_file = os.path.join('..', 'data', 'my_H_clean_combined_padded_optimized_test_autopilot_40m_for_test.mat')
        
    if not os.path.exists(combined_data_file):
        raise FileNotFoundError(f"数据文件不存在: {combined_data_file}")
        
    print(f"数据文件: {combined_data_file}")
    print(f"分析的子载波: #{SUBCARRIER_TO_PLOT}")

    # --- 2. 数据加载与处理 ---
    print("\n--- 2. 加载并处理数据 ---")
    NUM_ACTIVE_SUBCARRIERS = 576 
    total_features = NUM_ACTIVE_SUBCARRIERS * 2
    
    mat = scipy.io.loadmat(combined_data_file)
    
    if 'H' not in mat:
        raise ValueError(f"在文件 '{os.path.basename(combined_data_file)}' 中找不到名为 'H' 的变量。可用变量有: {list(mat.keys())}")

    # === FIX START: 处理总数据量无法被整除的问题 ===
    # 1. 加载原始数据并完全展平，确保其为一维数组。
    raw_data = mat['H']
    flat_data = raw_data.flatten().astype(np.float32)
    total_elements = flat_data.size
    print(f"成功加载数据，总元素个数为: {total_elements}")

    # 2. 检查总元素数量是否能被每个时间步的特征数整除。
    if total_elements % total_features != 0:
        # 如果不能，计算可以形成完整时间步的最大元素数量。
        num_timesteps_possible = total_elements // total_features
        elements_to_keep = num_timesteps_possible * total_features
        
        print(f"警告: 数据总大小 ({total_elements}) 不能被每个时间步的特征数 ({total_features}) 整除。")
        print(f"将截断数据，只保留前 {elements_to_keep} 个元素 ({num_timesteps_possible} 个完整的时间步)。")
        
        # 截断数组，丢弃末尾不完整的数据。
        truncated_data = flat_data[:elements_to_keep]
    else:
        # 如果可以整除，则直接使用。
        truncated_data = flat_data

    # 3. 对截断后、大小保证可被整除的数组进行重塑。
    if truncated_data.size == 0:
         raise ValueError("处理后的数据为空，无法继续。请检查原始数据文件。")
         
    H_2d_matrix = truncated_data.reshape(-1, total_features)
    # === FIX END ===

    print(f"数据重塑成功，最终矩阵形状为: {H_2d_matrix.shape}")
    
    is_zero_row = ~H_2d_matrix.any(axis=1)
    block_change_indices = np.where(np.diff(is_zero_row))[0] + 1
    data_blocks_2d = np.split(H_2d_matrix, block_change_indices)
    continuous_blocks = [block for block in data_blocks_2d if not np.all(block == 0)]
    
    if not continuous_blocks:
        raise ValueError("数据文件中未找到有效的连续数据块。")
        
    # === MODIFICATION: 筛选出足够长的数据块以供分析 ===
    print(f"共找到 {len(continuous_blocks)} 个有效数据块。正在筛选长度不小于 {MIN_BLOCK_LENGTH} 的数据块...")
    long_enough_blocks = [block for block in continuous_blocks if block.shape[0] >= MIN_BLOCK_LENGTH]

    if not long_enough_blocks:
        raise ValueError(f"未能找到任何长度至少为 {MIN_BLOCK_LENGTH} 的连续数据块。")

    # === MODIFICATION: 从筛选后的长数据块中随机选择一个 ===
    num_long_blocks = len(long_enough_blocks)
    selected_index = random.randint(0, num_long_blocks - 1)
    analysis_block = long_enough_blocks[selected_index]
    print(f"找到 {num_long_blocks} 个符合条件的数据块。从中随机选择一个进行分析。")
    # === END MODIFICATION ===

    num_timesteps = analysis_block.shape[0]
    print(f"所选数据块包含 {num_timesteps} 个时间步。")


    # 提取选定子载波的实部和虚部，并合成为复数序列
    real_part_idx = SUBCARRIER_TO_PLOT * 2
    imag_part_idx = real_part_idx + 1
    
    I_component = analysis_block[:, real_part_idx]
    Q_component = analysis_block[:, imag_part_idx]
    H_complex_series = I_component + 1j * Q_component
    H_magnitude = np.abs(H_complex_series)

    # --- 3. 生成方法一：信道轨迹图 ---
    print("\n--- 3. 正在生成信道轨迹图 ---")
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    
    ax1.plot(np.real(H_complex_series[:-1]), np.imag(H_complex_series[:-1]), 
             marker='o', markersize=2, linestyle='-', alpha=0.6, label='Trajectory')
    
    ax1.plot(np.real(H_complex_series[0]), np.imag(H_complex_series[0]), 
             'go', markersize=8, label='Start')
    ax1.plot(np.real(H_complex_series[-1]), np.imag(H_complex_series[-1]), 
             'ro', markersize=8, label='End')

    ax1.set_title(f'Channel Trajectory Plot for Subcarrier #{SUBCARRIER_TO_PLOT}')
    ax1.set_xlabel('In-Phase (Real Part)')
    ax1.set_ylabel('Quadrature (Imaginary Part)')
    ax1.grid(True)
    ax1.axhline(0, color='grey', lw=0.5)
    ax1.axvline(0, color='grey', lw=0.5)
    ax1.axis('equal')
    ax1.legend()
    
    # --- 4. 生成方法三：自相关函数(ACF)图 ---
    print("\n--- 4. 正在生成自相关函数(ACF)图 ---")
    
    # 动态调整 n_lags，确保它小于数据长度
    n_lags = min(100, num_timesteps - 2)
    if n_lags < 1:
        print("数据块过短，无法生成有意义的ACF图。")
    else:
        print(f"动态设置 ACF 的 lags 为: {n_lags}")
        fig2, axs = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
        fig2.suptitle(f'Temporal Correlation Analysis (ACF) for Subcarrier #{SUBCARRIER_TO_PLOT}', fontsize=16)
        
        # 绘制实部(I)的ACF
        plot_acf(I_component, ax=axs[0], lags=n_lags, title='Autocorrelation of In-Phase (I)')
        axs[0].set_xlabel('Time Lag')
        axs[0].set_ylabel('Autocorrelation')

        # 绘制虚部(Q)的ACF
        plot_acf(Q_component, ax=axs[1], lags=n_lags, title='Autocorrelation of Quadrature (Q)')
        axs[1].set_xlabel('Time Lag')
        axs[1].set_ylabel('Autocorrelation')

        # 绘制模(|H|)的ACF
        plot_acf(H_magnitude, ax=axs[2], lags=n_lags, title='Autocorrelation of Magnitude (|H|)')
        axs[2].set_xlabel('Time Lag')
        axs[2].set_ylabel('Autocorrelation')

    # --- 5. 显示图像 ---
    plt.show()

if __name__ == '__main__':
    try:
        analyze_channel_correlation()
    except (FileNotFoundError, ValueError) as e:
        print(f"\n错误: {e}")
    except Exception as e:
        print(f"\n发生未知错误: {e}")

