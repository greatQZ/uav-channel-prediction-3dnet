import os
import re
import numpy as np
import scipy.io
import pandas as pd

def process_log_file_optimally(file_path, padding_timesteps=100):
    """
    采用单遍扫描算法，高效地解析、验证、分割并转换信道数据。
    [已升级]：现在会同时提取信道H和同步的SNR值。

    Args:
        file_path (str): 信道数据文件的路径。
        padding_timesteps (int): 在不连续的数据块之间插入的安全间隔长度（以时间点计）。

    Returns:
        np.ndarray: 包含所有处理好的、由安全间隔分隔开的H数据的一维数组。
        np.ndarray: 与H数据帧一一对应的SNR值的一维数组。
        int: 在数据中识别出的有效子载波数量。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"信道文件未找到: {file_path}")

    print(f"正在以【单遍扫描模式】高效处理文件: {os.path.basename(file_path)}...")
    
    # <<< --- 关键改动 1: 更新正则表达式以捕获SNR值 --- >>>
    # 新增了第三个捕获组 (\d+) 来匹配 "SNR: 40 dB" 中的 40
    header_pattern = re.compile(r"SRS Frame (\d+),.* Real: ([\d\.]+),.* SNR: (\d+) dB")
    sc_pattern = re.compile(r"Sc (\d+): Re = (-?\d+), Im = (-?\d+)")

    all_processed_blocks = []
    current_continuous_block_data = []
    
    # <<< --- 关键改动 2: 新增列表用于存储SNR数据 --- >>>
    all_snr_blocks = []
    current_continuous_block_snr = []

    last_valid_frame_index = None
    current_timestamp = None
    current_snr = None # 用于临时存储当前帧的SNR
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
                            if current_frame_index > last_valid_frame_index:
                                frame_diff = current_frame_index - last_valid_frame_index
                            else:
                                frame_diff = (current_frame_index + FRAMES_PER_CYCLE) - last_valid_frame_index
                            
                            if frame_diff != EXPECTED_FRAME_DIFF:    
                                is_discontinuous = True

                        if is_discontinuous:
                            print(f" -> 检测到数据中断，正在处理上一个连续块...")
                            if current_continuous_block_data:
                                # 处理H数据块
                                block_df = pd.DataFrame(current_continuous_block_data)
                                processed_array, num_active_sc = convert_to_interleaved_1d_array(block_df)
                                all_processed_blocks.append(processed_array)
                                if num_active_sc > 0: num_active_subcarriers = num_active_sc
                                
                                # 添加H数据的安全间隔
                                padding_length = padding_timesteps * num_active_subcarriers * 2
                                all_processed_blocks.append(np.zeros(padding_length))
                                print(f" -> 已在上一个数据块后插入长度为 {padding_length} 的安全间隔。")
                                
                                # <<< --- 关键改动 3: 同步处理SNR数据块 --- >>>
                                # SNR数据不需要padding，直接添加到列表中
                                all_snr_blocks.append(np.array(current_continuous_block_snr))

                            current_continuous_block_data = []
                            current_continuous_block_snr = [] # 清空SNR块

                        h_vector = np.zeros(num_subcarriers, dtype=np.complex128)
                        for sc_idx, value in current_sc_data.items(): h_vector[sc_idx] = value
                        current_continuous_block_data.append({'h_vector': h_vector})
                        
                        # <<< --- 关键改动 4: 将当前帧的SNR值添加到连续块列表中 --- >>>
                        current_continuous_block_snr.append(current_snr)
                        
                        last_valid_frame_index = current_frame_index
                    else:
                        print(f"!!! 警告: 帧 {current_frame_index} 的数据不完整，已丢弃。")

                # 准备下一个新帧
                current_frame_index = int(header_match.group(1))
                ts_unix_nano = float(header_match.group(2))
                current_timestamp = pd.to_datetime(ts_unix_nano, unit='s')
                # <<< --- 关键改动 5: 提取并存储当前帧的SNR值 --- >>>
                current_snr = int(header_match.group(3))
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
        
        # <<< --- 关键改动 6: 处理最后一个SNR数据块 --- >>>
        all_snr_blocks.append(np.array(current_continuous_block_snr))

    print(f"文件 {os.path.basename(file_path)} 处理完成。")
    
    # <<< --- 关键改动 7: 同时返回处理好的H和SNR数据 --- >>>
    h_array = np.concatenate(all_processed_blocks) if all_processed_blocks else np.array([])
    snr_array = np.concatenate(all_snr_blocks) if all_snr_blocks else np.array([])
    return h_array, snr_array, num_active_subcarriers

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
        os.path.join(os.getcwd(), 'data', 'channel_estimates_20250827_174423_heli_flight_for_training_testing.txt'),
    ]
    
    output_mat_filename = 'channel_estimates_20250827_174423_heli_flight_for_training_testing.mat'
    # <<< --- 关键改动 8: 新增SNR输出文件名 --- >>>
    output_npy_filename_snr = 'heli_flight_SNR_real_data.npy'
    
    padding_timesteps = 100 
    
    all_processed_data = []
    # <<< --- 关键改动 9: 新增列表用于聚合所有SNR数据 --- >>>
    all_snr_data = []
    num_active_subcarriers = 0

    try:
        # --- 2. 逐一处理每个文件 ---
        for i, log_path in enumerate(channel_log_files):
            # <<< --- 关键改动 10: 接收函数返回的SNR数组 --- >>>
            processed_array, snr_array, num_active_sc = process_log_file_optimally(log_path, padding_timesteps)
            
            if processed_array.size > 0:
                all_processed_data.append(processed_array)
                # <<< --- 关键改动 11: 将当前文件的SNR数组添加到聚合列表中 --- >>>
                all_snr_data.append(snr_array)

                if num_active_sc > 0: num_active_subcarriers = num_active_sc
                
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
        # <<< --- 关键改动 12: 合并所有SNR数据 --- >>>
        final_snr_data = np.concatenate(all_snr_data)
        
        print(f"所有数据合并完成。")
        print(f" - 最终H数据集总长度: {len(final_combined_clean_H)}")
        print(f" - 最终SNR数据集总长度: {len(final_snr_data)}")
        
        # 保存H数据
        scipy.io.savemat(output_mat_filename, {'H': final_combined_clean_H})
        print(f" -> H数据已成功保存到: {output_mat_filename}")

        # <<< --- 关键改动 13: 保存SNR数据 --- >>>
        np.save(output_npy_filename_snr, final_snr_data)
        print(f" -> SNR数据已成功保存到: {output_npy_filename_snr}")


    except (FileNotFoundError, ValueError) as e:
        print(f"\n错误: {e}")
    except Exception as e:
        print(f"\n发生未知错误: {e}")