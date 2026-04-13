import os
import re
import numpy as np
import scipy.io
import pandas as pd

def process_log_file_optimally(file_path, padding_timesteps=100):
    """
    采用单遍扫描算法，高效地解析、验证、分割并转换信道数据。

    Args:
        file_path (str): 信道数据文件的路径。
        padding_timesteps (int): 在不连续的数据块之间插入的安全间隔长度（以时间点计）。

    Returns:
        np.ndarray: 包含所有处理好的、由安全间隔分隔开的数据的一维数组。
        int: 在数据中识别出的有效子载波数量。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"信道文件未找到: {file_path}")

    print(f"正在以【单遍扫描模式】高效处理文件: {os.path.basename(file_path)}...")
    
    header_pattern = re.compile(r"SRS Frame (\d+),.* Real: ([\d\.]+)")
    sc_pattern = re.compile(r"Sc (\d+): Re = (-?\d+), Im = (-?\d+)")

    all_processed_blocks = []
    current_continuous_block_data = [] # 存储一个连续块的所有有效帧的h_vector
    
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
                # 1. 首先处理并验证上一个完整的数据帧
                if current_timestamp is not None and current_sc_data:
                    if len(current_sc_data) == num_subcarriers: # 帧内验证
                        is_discontinuous = False
                        if last_valid_timestamp is not None: # 帧间验证
                            #time_diff = (current_timestamp - last_valid_timestamp).total_seconds()
                            
                            if current_frame_index > last_valid_frame_index:
                                frame_diff = current_frame_index - last_valid_frame_index
                            else:
                                frame_diff = (current_frame_index + FRAMES_PER_CYCLE) - last_valid_frame_index
                            
                            #if time_diff > MAX_TIME_DIFF_S and frame_diff != EXPECTED_FRAME_DIFF:
                            if frame_diff != EXPECTED_FRAME_DIFF:    
                                is_discontinuous = True

                        if is_discontinuous:
                            print(f" -> 检测到数据中断，正在处理上一个连续块...")
                            if current_continuous_block_data:
                                block_df = pd.DataFrame(current_continuous_block_data)
                                processed_array, num_active_sc = convert_to_interleaved_1d_array(block_df)
                                all_processed_blocks.append(processed_array)
                                if num_active_sc > 0: num_active_subcarriers = num_active_sc
                                print(f"DEBUG: Found {num_active_sc} active subcarriers for block.")#debug
                                padding_length = padding_timesteps * num_active_subcarriers * 2
                                all_processed_blocks.append(np.zeros(padding_length))
                                print(f" -> 已在上一个数据块后插入长度为 {padding_length} 的安全间隔。")
                            current_continuous_block_data = [] # 开始一个新的块
                        
                        h_vector = np.zeros(num_subcarriers, dtype=np.complex128)
                        for sc_idx, value in current_sc_data.items(): h_vector[sc_idx] = value
                        current_continuous_block_data.append({'h_vector': h_vector})
                        last_valid_timestamp = current_timestamp
                        last_valid_frame_index = current_frame_index
                    else:
                        print(f"!!! 警告: 时间戳 {current_timestamp} 的数据帧不完整，已丢弃。")

                # 2. 准备下一个新帧
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

    # --- 循环结束后，处理最后一个数据块 ---
    if current_continuous_block_data:
        print(" -> 正在处理文件末尾的最后一个连续数据块...")
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


if __name__ == '__main__':
    # --- 1. 用户配置区域 ---
    channel_log_files = [
        #os.path.join(os.getcwd(), 'data', 'ch_est_pavel_40m_10mps_from_big_rec_autopilot_phase1.txt'),
        #os.path.join(os.getcwd(), 'data', 'ch_est_pavel_40m_10mps_from_big_rec_autopilot_phase2.txt'),
        #os.path.join(os.getcwd(), 'data', 'ch_est_pavel_40m_20mps_lange_route_from_big_rec_autopilot.txt'),
        #os.path.join(os.getcwd(), 'data', 'ch_est_pavel_40m_20mps_versuch_phase1_from_big_rec_autopilot.txt'),
        #os.path.join(os.getcwd(), 'data', 'ch_est_pavel_40m_20mps_versuch_phase2_from_big_rec_autopilot.txt'),
        #os.path.join(os.getcwd(), 'data', 'ch_est_pavel_60m_10mps_from_big_rec_autopilot.txt'),
        #os.path.join(os.getcwd(), 'data', 'ch_est_pavel_60m_15mps_20mps_from_big_rec_autopilot.txt'),
        os.path.join(os.getcwd(), 'data', 'channel_estimates_20250917_140636_bolek_40m_5mps_autopilot.txt'),
    ]
    
    output_mat_filename = 'ch_est_40m_5mps_autopilot.mat'
    padding_timesteps = 100 
    
    all_processed_data = []
    num_active_subcarriers = 0

    try:
        # --- 2. 逐一处理每个文件 ---
        for i, log_path in enumerate(channel_log_files):
            processed_array, num_active_sc = process_log_file_optimally(log_path, padding_timesteps)
            
            if processed_array.size > 0:
                all_processed_data.append(processed_array)
                if num_active_sc > 0: num_active_subcarriers = num_active_sc
                #print(f"DEBUG: Found {num_active_sc} active subcarriers for block.")
                # 在不同文件之间总是添加安全间隔
                if i < len(channel_log_files) - 1 and num_active_subcarriers > 0:
                    print(f" -> 在文件 {os.path.basename(log_path)} 和下一个文件之间插入安全间隔...")
                    padding_length = padding_timesteps * num_active_subcarriers * 2
                    all_processed_data.append(np.zeros(padding_length))
            print("-" * 70)

        # --- 3. 合并所有数据 ---
        if not all_processed_data:
            raise ValueError("所有文件中都没有找到任何有效的完整数据帧，无法生成最终文件。")

        print("\n正在合并所有已处理的数据...")
        final_combined_clean_H = np.concatenate(all_processed_data)
        print(f"所有数据合并完成。最终干净数据集的总长度为: {len(final_combined_clean_H)}")

        # 将最终的合并数组保存为.mat文件
        scipy.io.savemat(output_mat_filename, {'H': final_combined_clean_H})
        print(f" -> 干净的合并数据已成功保存到: {output_mat_filename}")

    except (FileNotFoundError, ValueError) as e:
        print(f"\n错误: {e}")
    except Exception as e:
        print(f"\n发生未知错误: {e}")
