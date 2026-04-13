import os
import re
import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
#  这里包含了您画布中的核心处理函数，以便进行独立验证
# ==============================================================================

def process_log_file_optimally(file_path, padding_timesteps=100):
    """
    【加速版】采用单遍扫描算法，高效地解析、验证、分割并转换信道数据。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"信道文件未找到: {file_path}")

    print(f"正在以【单遍扫描模式】高效处理文件: {os.path.basename(file_path)}...")
    
    header_pattern = re.compile(r"SRS Frame (\d+),.* Real: ([\d\.]+)")
    sc_pattern = re.compile(r"Sc (\d+): Re = (-?\d+), Im = (-?\d+)")

    all_processed_blocks = []
    current_continuous_block_data = []
    
    last_valid_timestamp = None
    last_valid_frame_index = None
    current_timestamp = None
    num_subcarriers = 1024
    current_sc_data = {}
    num_active_subcarriers = 0

    FRAMES_PER_CYCLE = 1024
    EXPECTED_FRAME_DIFF = 1
    MAX_TIME_DIFF_S = 0.015 

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            header_match = header_pattern.search(line)
            if header_match:
                if current_timestamp is not None and current_sc_data:
                    if len(current_sc_data) == num_subcarriers:
                        is_discontinuous = False
                        if last_valid_timestamp is not None:
                            time_diff = (current_timestamp - last_valid_timestamp).total_seconds()
                            
                            if current_frame_index > last_valid_frame_index:
                                frame_diff = current_frame_index - last_valid_frame_index
                            else:
                                frame_diff = (current_frame_index + FRAMES_PER_CYCLE) - last_valid_frame_index
                            
                            if time_diff > MAX_TIME_DIFF_S or frame_diff != EXPECTED_FRAME_DIFF:
                                is_discontinuous = True

                        if is_discontinuous:
                            if current_continuous_block_data:
                                block_df = pd.DataFrame(current_continuous_block_data)
                                processed_array, num_active_sc = convert_to_interleaved_1d_array(block_df)
                                all_processed_blocks.append(processed_array)
                                if num_active_sc > 0: num_active_subcarriers = num_active_sc
                                
                                padding_length = padding_timesteps * num_active_subcarriers * 2
                                all_processed_blocks.append(np.zeros(padding_length))
                            current_continuous_block_data = []
                        
                        h_vector = np.zeros(num_subcarriers, dtype=np.complex128)
                        for sc_idx, value in current_sc_data.items(): h_vector[sc_idx] = value
                        current_continuous_block_data.append({'h_vector': h_vector})
                        last_valid_timestamp = current_timestamp
                        last_valid_frame_index = current_frame_index
                    else:
                        pass # 静默丢弃不完整的帧

                current_frame_index = int(header_match.group(1))
                ts_unix_nano = float(header_match.group(2))
                current_timestamp = pd.to_datetime(ts_unix_nano, unit='s')
                current_sc_data = {}
            
            elif current_timestamp:
                sc_match = sc_pattern.search(line)
                if sc_match:
                    sc_index, re_val, im_val = map(int, sc_match.groups())
                    if sc_index < num_subcarriers:
                        current_sc_data[sc_index] = re_val + 1j * im_val

    if current_continuous_block_data:
        block_df = pd.DataFrame(current_continuous_block_data)
        processed_array, num_active_sc = convert_to_interleaved_1d_array(block_df)
        all_processed_blocks.append(processed_array)
        if num_active_sc > 0: num_active_subcarriers = num_active_sc

    print(f"文件 {os.path.basename(file_path)} 处理完成。")
    return np.concatenate(all_processed_blocks) if all_processed_blocks else np.array([]), num_active_subcarriers

def convert_to_interleaved_1d_array(df):
    """
    将包含复数信道向量的DataFrame转换为项目所需的一维交错数组。
    """
    H_complex_matrix = np.stack(df['h_vector'].values)
    avg_power_per_sc = np.mean(np.abs(H_complex_matrix)**2, axis=0)
    active_sc_indices = np.where(avg_power_per_sc > 1e-9)[0]
    
    if len(active_sc_indices) == 0:
        return np.array([]), 0
        
    H_active_sc = H_complex_matrix[:, active_sc_indices]
    H_real = np.real(H_active_sc)
    H_imag = np.imag(H_active_sc)
    
    num_timesteps, num_active_subcarriers = H_active_sc.shape
    H_interleaved = np.zeros((num_timesteps, num_active_subcarriers * 2))
    
    H_interleaved[:, ::2] = H_real
    H_interleaved[:, 1::2] = H_imag
    
    return H_interleaved.flatten(), num_active_subcarriers

# ==============================================================================
#  验证工具主程序
# ==============================================================================

def find_first_valid_frame(log_path):
    """
    【新功能】扫描文件以找到第一个完整的帧的起始行和帧号。
    """
    header_pattern = re.compile(r"SRS Frame (\d+),")
    sc_pattern = re.compile(r"Sc (\d+):")
    
    current_frame_start_line = -1
    current_frame_index = -1
    sc_indices_in_frame = set()

    with open(log_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            header_match = header_pattern.search(line)
            if header_match:
                # 检查上一个收集的帧是否完整
                if len(sc_indices_in_frame) == 1024:
                    print(f" -> 找到第一个完整帧: {current_frame_index} (起始行: {current_frame_start_line})")
                    return current_frame_start_line, current_frame_index
                
                # 开始收集新的一帧
                current_frame_start_line = i
                current_frame_index = int(header_match.group(1))
                sc_indices_in_frame = set()

            sc_match = sc_pattern.search(line)
            if sc_match:
                sc_indices_in_frame.add(int(sc_match.group(1)))
    
    # 检查文件末尾的最后一帧
    if len(sc_indices_in_frame) == 1024:
        print(f" -> 找到第一个完整帧: {current_frame_index} (起始行: {current_frame_start_line})")
        return current_frame_start_line, current_frame_index
        
    return None, None

def verify_processing(log_path, start_frame_index, num_frames_to_check=10):
    """
    通过抽样检查来验证数据处理流程的准确性。
    """
    print("\n" + "="*70)
    print("### 开始数据处理验证流程 ###")
    print("="*70)

    # --- 步骤 1: 直接从原始文件中解析一小段数据作为“标准答案” ---
    print("\n[步骤 1] 正在直接解析原始数据以获取“标准答案”...")
    header_pattern = re.compile(r"SRS Frame (\d+),")
    sc_pattern = re.compile(r"Sc (\d+): Re = (-?\d+), Im = (-?\d+)")
    
    ground_truth_data = []
    frame_count = 0
    capture_started = False
    
    with open(log_path, 'r', encoding='utf-8') as file:
        for line in file:
            header_match = header_pattern.search(line)
            if header_match:
                current_frame = int(header_match.group(1))
                if current_frame == start_frame_index and not capture_started:
                    capture_started = True
                    print(f" -> 已在文件中定位到起始帧: {start_frame_index}")
                
                if capture_started:
                    if frame_count >= num_frames_to_check:
                        break
                    frame_count += 1
                    current_h_vector = np.zeros(1024, dtype=np.complex128)
                    ground_truth_data.append(current_h_vector)

            elif capture_started and ground_truth_data:
                sc_match = sc_pattern.search(line)
                if sc_match:
                    sc_index, re_val, im_val = map(int, sc_match.groups())
                    ground_truth_data[-1][sc_index] = re_val + 1j * im_val
    
    ground_truth_matrix = np.stack(ground_truth_data)
    print(f" -> “标准答案”提取完成，形状为: {ground_truth_matrix.shape}")

    # --- 步骤 2: 运行主处理脚本 ---
    print("\n[步骤 2] 正在运行您的主处理脚本...")
    processed_1d_array, num_active_sc = process_log_file_optimally(log_path)
    if processed_1d_array.size == 0:
        print("!!! 错误: 主处理脚本未能生成任何数据，无法进行验证。")
        return

    # --- 步骤 3: 从处理结果中“反向工程”出我们检查的数据段 ---
    print("\n[步骤 3] 正在从处理结果中恢复抽样数据...")
    reconstructed_2d_interleaved = processed_1d_array.reshape(-1, num_active_sc * 2)
    sample_to_check = reconstructed_2d_interleaved[:num_frames_to_check, :]
    reconstructed_complex = sample_to_check[:, ::2] + 1j * sample_to_check[:, 1::2]
    print(f" -> 数据恢复完成，恢复出的矩阵形状为: {reconstructed_complex.shape}")

    # --- 步骤 4: 可视化对比 ---
    print("\n[步骤 4] 正在生成可视化对比图...")
    
    avg_power_per_sc_gt = np.mean(np.abs(ground_truth_matrix)**2, axis=0)
    active_sc_indices_gt = np.where(avg_power_per_sc_gt > 1e-9)[0]
    ground_truth_active = ground_truth_matrix[:, active_sc_indices_gt]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)
    
    im1 = axes[0].imshow(np.abs(ground_truth_active).T, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Ground Truth (Parsed Directly from Log)', fontsize=16)
    axes[0].set_xlabel('Time Index', fontsize=12)
    axes[0].set_ylabel('Active Subcarrier Index', fontsize=12)
    
    im2 = axes[1].imshow(np.abs(reconstructed_complex).T, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Reconstructed Data (from Processing Script)', fontsize=16)
    axes[1].set_xlabel('Time Index', fontsize=12)
    
    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.8, label='Channel Amplitude')
    plt.suptitle('Data Processing Verification', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # --- 步骤 5: 数值对比 ---
    print("\n[步骤 5] 正在进行数值对比...")
    if ground_truth_active.shape == reconstructed_complex.shape and np.allclose(ground_truth_active, reconstructed_complex):
        print("✅ 验证成功！恢复后的数据与标准答案完全匹配。")
    else:
        diff = np.sum(np.abs(ground_truth_active - reconstructed_complex))
        print(f"❌ 验证失败：数据不匹配。总差异值为: {diff}")
        print(f"   - 标准答案形状: {ground_truth_active.shape}")
        print(f"   - 恢复数据形状: {reconstructed_complex.shape}")
    
    print("="*70)


if __name__ == '__main__':
    # --- 配置 ---
    file_to_verify = os.path.join(os.getcwd(), 'data', 'ch_est_pavel_40m_10mps_from_big_rec.txt')
    
    # --- 【核心修正】自动寻找第一个有效的Frame号进行验证 ---
    print(f"正在自动寻找文件 '{os.path.basename(file_to_verify)}' 中的第一个完整帧...")
    _, start_frame_for_verification = find_first_valid_frame(file_to_verify)
    
    if start_frame_for_verification is None:
        print("!!! 错误: 未能在文件中找到任何【完整】的Frame用于验证。")
    else:
        # --- 运行验证 ---
        verify_processing(file_to_verify, start_frame_for_verification)
