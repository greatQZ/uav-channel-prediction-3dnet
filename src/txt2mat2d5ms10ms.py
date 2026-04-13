import os
import re
import numpy as np
import scipy.io
import traceback

# --- 配置 ---
# 我们只关心有 1024 个子载波的“完整”帧
NUM_SUBCARRIERS = 1024

def process_log_file_V_FINAL(file_path):
    """
    【最终 V5 版本 - 适用于 5ms 和 10ms 数据】
    读取一个文件中的 *所有* 有效帧（1024 SCs），并将它们
    *按顺序* 收集到一个列表中，作为 2D 矩阵返回。
    *移除了所有基于 frame_index 的逻辑*
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Processing file (V_FINAL): {os.path.basename(file_path)}...")
    
    # 模式匹配
    header_pattern = re.compile(r"SRS Frame (\d+),.* Real: ([\d\.]+)")
    sc_pattern = re.compile(r"Sc (\d+): Re = (-?\d+), Im = (-?\d+)")

    all_valid_frames_data = [] # 存储所有有效 h_vectors
    
    current_timestamp = None
    current_sc_data = {}

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for line_num, line in enumerate(file):
            header_match = header_pattern.search(line)
            if header_match:
                # 1. 首先处理并验证上一个完整的数据帧
                if current_timestamp is not None:
                    # 验证：这是否是一个包含 1024 个 SC 的完整帧？
                    if len(current_sc_data) == NUM_SUBCARRIERS:
                        # 是，构建 h_vector
                        h_vector = np.zeros(NUM_SUBCARRIERS, dtype=np.complex128)
                        for sc_idx, value in current_sc_data.items():
                            h_vector[sc_idx] = value
                        all_valid_frames_data.append(h_vector) # 直接按顺序添加
                    else:
                        # 否，这是一个不完整的 slot，我们丢弃它
                        if len(current_sc_data) > 0: 
                            frame_idx_for_warning = int(header_match.group(1)) # 获取帧号用于警告
                            print(f"  INFO: Dropping incomplete slot (found {len(current_sc_data)}/{NUM_SUBCARRIERS} SCs) near frame {frame_idx_for_warning}.")

                # 2. 准备下一个新帧/Slot
                current_timestamp = float(header_match.group(2)) # 仅用于检查
                current_sc_data = {}
            
            elif current_timestamp: # 仅在找到第一个 header 后才开始收集 SC
                sc_match = sc_pattern.search(line)
                if sc_match:
                    sc_index, re_val, im_val = map(int, sc_match.groups())
                    if sc_index < NUM_SUBCARRIERS:
                        current_sc_data[sc_index] = re_val + 1j * im_val

    # --- 循环结束后，处理最后一个数据块 ---
    if current_timestamp is not None:
        if len(current_sc_data) == NUM_SUBCARRIERS:
            h_vector = np.zeros(NUM_SUBCARRIERS, dtype=np.complex128)
            for sc_idx, value in current_sc_data.items():
                h_vector[sc_idx] = value
            all_valid_frames_data.append(h_vector)
        elif len(current_sc_data) > 0:
            print(f"  INFO: Dropping final incomplete slot (found {len(current_sc_data)}/{NUM_SUBCARRIERS} SCs).")

    print(f"  Found {len(all_valid_frames_data)} total valid (1024-SC) frames/slots in file.")

    if not all_valid_frames_data:
        return None

    # --- 将所有帧堆叠成一个 2D 矩阵 ---
    H_matrix_full = np.stack(all_valid_frames_data) # Shape [T, 1024]
    
    print(f"  File processed. Full matrix shape: {H_matrix_full.shape}")
    return H_matrix_full


if __name__ == '__main__':
    # --- 1. 用户配置区域 ---
    base_data_path = 'data' # 假设 .txt 文件在 'data' 目录中
    base_output_path = 'data' # !! 新的 V_FINAL 输出目录 !!
    
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)
        print(f"Created output directory: {base_output_path}")

    # 定义要处理的 *所有* 实验
    experiments_to_process = {

        # --- 40m 组 (10ms) ---
        'ch_est_40m_5mps_autopilot_2d.mat': [
            os.path.join(base_data_path, 'channel_estimates_20250917_140636_bolek_40m_5mps_autopilot.txt')
        ],
        'ch_est_40m_10mps_autopilot_2d.mat': [
            # 示例：合并两个文件
            os.path.join(base_data_path, 'ch_est_pavel_40m_10mps_from_big_rec_autopilot_phase1.txt'),
            os.path.join(base_data_path, 'ch_est_pavel_40m_10mps_from_big_rec_autopilot_phase2.txt')
        ],
        'ch_est_40m_15mps_autopilot_2d.mat': [
            os.path.join(base_data_path, 'ch_est_pavel_40m_20mps_versuch_phase1_from_big_rec_autopilot.txt'),
            os.path.join(base_data_path, 'ch_est_pavel_40m_20mps_versuch_phase2_from_big_rec_autopilot.txt')
        ],
        
        # --- 50m 组 (5ms) ---
        'ch_est_50m_5mps_5ms_autopilot_2d.mat': [
            os.path.join(base_data_path, 'channel_estimates_20250917_150803_bolek_50m_5mps_5ms_autopilot.txt') 
            # 确保这是 50m 5mps 的 .txt 文件名
        ],
        'ch_est_50m_10mps_5ms_autopilot_2d.mat': [
            os.path.join(base_data_path, 'channel_estimates_20250917_150803_bolek_50m_10mps_5ms_autopilot.txt')
            # 确保这是 50m 10mps 的 .txt 文件名
        ],
        
        # --- 60m 组 (10ms) ---
        'ch_est_60m_10mps_autopilot_2d.mat': [
             os.path.join(base_data_path, 'ch_est_pavel_60m_10mps_from_big_rec_autopilot.txt')
        ],
        'ch_est_60m_15mps_autopilot_2d.mat': [
             os.path.join(base_data_path, 'ch_est_pavel_60m_15mps_20mps_from_big_rec_autopilot.txt')
        ],
    }

    try:
        all_H_matrices = [] # 存储 *所有* 实验的 H 矩阵
        all_filenames = [] # 存储对应的文件名

        # --- 2. 逐一处理每个实验 ---
        for output_filename, input_files_list in experiments_to_process.items():
            print(f"\n--- Processing Experiment: {output_filename} ---")
            
            all_h_vectors_for_exp = [] # 单个实验的所有 H_vectors
            
            for log_path in input_files_list:
                H_matrix_full = process_log_file_V_FINAL(log_path)
                if H_matrix_full is not None:
                    all_h_vectors_for_exp.append(H_matrix_full)
            
            if not all_h_vectors_for_exp:
                print(f"  No valid data found for {output_filename}, skipping.")
                print("-" * 70)
                continue
            
            # --- 3. 合并来自多个 .txt 文件的 H 矩阵 (例如 phase1 和 phase2) ---
            H_combined_exp = np.vstack(all_h_vectors_for_exp)
            all_H_matrices.append(H_combined_exp) # 添加到 *总* 列表
            all_filenames.append(output_filename) # 记录文件名
            print(f"  Experiment combined matrix shape: {H_combined_exp.shape}")
            
        
        # --- 4. 找到 *所有* 数据的 *统一* 活跃子载波 ---
        if not all_H_matrices:
             raise ValueError("所有文件中都没有找到任何有效的完整数据帧。")

        print("\nConcatenating ALL data from ALL experiments to find common active subcarriers...")
        H_global_combined = np.vstack(all_H_matrices)
        print(f"Global combined matrix shape: {H_global_combined.shape}")
        
        # 仅计算一次平均功率和活跃索引
        avg_power_per_sc = np.mean(np.abs(H_global_combined)**2, axis=0)
        active_indices = np.where(avg_power_per_sc > 1e-9)[0]
        num_active_sc = len(active_indices)
        
        print(f"Found {num_active_sc} common active subcarriers across all files (e.g., {active_indices[:3]}...{active_indices[-3:]}).")

        # --- 5. 逐一保存 *已过滤* 的 .mat 文件 ---
        for H_matrix_exp, output_filename in zip(all_H_matrices, all_filenames):
            output_mat_path = os.path.join(base_output_path, output_filename)
            
            # *仅* 选择活跃的子载波
            H_active_matrix = H_matrix_exp[:, active_indices] # Shape [T_total, N_active]

            # --- 6. 将 2D 复数矩阵和索引一起保存 ---
            print(f"  Saving filtered matrix {H_active_matrix.shape} for {output_filename} to {output_mat_path}")
            scipy.io.savemat(output_mat_path, {
                'H_active_matrix': H_active_matrix, 
                'active_indices': active_indices
            })
            
        print("\n--- 所有文件处理完成！---")

    except (FileNotFoundError, ValueError) as e:
        print(f"\n错误: {e}")
    except Exception as e:
        print(f"\n发生未知错误: {e}")
        traceback.print_exc()