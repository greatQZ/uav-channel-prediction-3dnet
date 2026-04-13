# txt_to_sp_np.py
# 最终版: 提取H, SP和NP

import os
import re
import numpy as np
import scipy.io
import pandas as pd

def process_log_file_optimally(file_path, padding_timesteps=100):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"信道文件未找到: {file_path}")

    print(f"正在以【单遍扫描模式】高效处理文件: {os.path.basename(file_path)}...")
    
    # <<< --- 关键改动 1: 更新正则表达式以捕获 SP 和 NP --- >>>
    header_pattern = re.compile(r"SRS Frame (\d+),.* Real: ([\d\.]+),.* SP: (\d+), NP:(\d+),")
    sc_pattern = re.compile(r"Sc (\d+): Re = (-?\d+), Im = (-?\d+)")

    all_processed_blocks = []
    current_continuous_block_data = []
    
    all_sp_np_blocks = []
    current_continuous_block_sp_np = []

    last_valid_frame_index = None
    current_timestamp = None
    current_sp_np = None
    num_subcarriers = 1024
    current_sc_data = {}
    num_active_subcarriers = 0
    FRAMES_PER_CYCLE = 1024
    EXPECTED_FRAME_DIFF = 1

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            header_match = header_pattern.search(line)
            if header_match:
                if current_timestamp is not None and current_sc_data:
                    if len(current_sc_data) == num_subcarriers:
                        is_discontinuous = False
                        if last_valid_frame_index is not None:
                            frame_diff = (current_frame_index - last_valid_frame_index + FRAMES_PER_CYCLE) % FRAMES_PER_CYCLE
                            if frame_diff > 1:    
                                is_discontinuous = True

                        if is_discontinuous:
                            if current_continuous_block_data:
                                block_df = pd.DataFrame(current_continuous_block_data)
                                processed_array, num_active_sc = convert_to_interleaved_1d_array(block_df)
                                all_processed_blocks.append(processed_array)
                                if num_active_sc > 0: num_active_subcarriers = num_active_sc
                                padding_length = padding_timesteps * num_active_subcarriers * 2
                                all_processed_blocks.append(np.zeros(padding_length))
                                all_sp_np_blocks.append(np.array(current_continuous_block_sp_np))
                            current_continuous_block_data = []
                            current_continuous_block_sp_np = []

                        h_vector = np.zeros(num_subcarriers, dtype=np.complex128)
                        for sc_idx, value in current_sc_data.items(): h_vector[sc_idx] = value
                        current_continuous_block_data.append({'h_vector': h_vector})
                        current_continuous_block_sp_np.append(current_sp_np)
                        last_valid_frame_index = current_frame_index
                    else:
                        print(f"!!! 警告: 帧 {current_frame_index} 的数据不完整，已丢弃。")

                current_frame_index = int(header_match.group(1))
                current_sp_np = (int(header_match.group(3)), int(header_match.group(4)))
                current_timestamp = float(header_match.group(2))
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
        all_sp_np_blocks.append(np.array(current_continuous_block_sp_np))

    h_array = np.concatenate(all_processed_blocks) if all_processed_blocks else np.array([])
    sp_np_array = np.concatenate(all_sp_np_blocks) if all_sp_np_blocks else np.array([])
    return h_array, sp_np_array, num_active_subcarriers

def convert_to_interleaved_1d_array(df):
    H_complex_matrix = np.stack(df['h_vector'].values)
    avg_power_per_sc = np.mean(np.abs(H_complex_matrix)**2, axis=0)
    active_sc_indices = np.where(avg_power_per_sc > 1e-9)[0]
    if len(active_sc_indices) == 0: return np.array([]), 0
    H_active_sc = H_complex_matrix[:, active_sc_indices]
    H_real = np.real(H_active_sc); H_imag = np.imag(H_active_sc)
    num_timesteps, num_active_subcarriers = H_active_sc.shape
    H_interleaved = np.zeros((num_timesteps, num_active_subcarriers * 2))
    H_interleaved[:, ::2] = H_real; H_interleaved[:, 1::2] = H_imag
    return H_interleaved.flatten(), num_active_subcarriers

if __name__ == '__main__':
    channel_log_files = ['data/channel_estimates_20250917_172122_5ms_bolek_40m_10mps_for_demo.txt']
    output_mat_filename = 'data/my_H_real_data_bolek_for_5ms_40m_10mps_for_demo.mat'
    output_npy_filename_sp_np = 'data/my_SP_NP_real_data_bolek_for_5ms_40m_10mps_for_demo.npy'
    padding_timesteps = 100 
    all_processed_data = []; all_sp_np_data = []
    num_active_subcarriers = 0

    try:
        for i, log_path in enumerate(channel_log_files):
            processed_array, sp_np_array, num_active_sc = process_log_file_optimally(log_path, padding_timesteps)
            if processed_array.size > 0:
                all_processed_data.append(processed_array)
                all_sp_np_data.append(sp_np_array)
                if num_active_sc > 0: num_active_subcarriers = num_active_sc
                if i < len(channel_log_files) - 1 and num_active_subcarriers > 0:
                    padding_length = padding_timesteps * num_active_subcarriers * 2
                    all_processed_data.append(np.zeros(padding_length))
        
        if not all_processed_data: raise ValueError("未找到有效数据")
        
        final_combined_clean_H = np.concatenate(all_processed_data)
        final_sp_np_data = np.concatenate(all_sp_np_data)
        
        scipy.io.savemat(output_mat_filename, {'H': final_combined_clean_H})
        print(f"H数据已保存到: {output_mat_filename}")

        np.save(output_npy_filename_sp_np, final_sp_np_data)
        print(f"SP/NP数据已保存到: {output_npy_filename_sp_np}")

    except Exception as e:
        print(f"\n错误: {e}")