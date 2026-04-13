import os
import re
import numpy as np
import scipy.io
import traceback

# --- 配置 ---
# OAI 在 1024 个帧后会回绕
FRAMES_PER_CYCLE = 1024
# 我们只关心有 1024 个子载波的“完整”帧
NUM_SUBCARRIERS = 1024

def process_log_file_V_FINAL(file_path):
    """
    【最终 V_FINAL 版本】
    读取一个文件中的 *所有* 有效帧（1024 SCs），并将它们
    作为 (frame_index, h_vector) 元组的列表返回。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Processing file (V_FINAL): {os.path.basename(file_path)}...")
    
    # 模式匹配
    header_pattern = re.compile(r"SRS Frame (\d+),.* Real: ([\d\.]+)")
    sc_pattern = re.compile(r"Sc (\d+): Re = (-?\d+), Im = (-?\d+)")

    all_valid_frames = [] # 存储 (frame_index, h_vector) 元组
    
    current_frame_index = None
    current_timestamp = None
    current_sc_data = {}

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for line_num, line in enumerate(file):
            header_match = header_pattern.search(line)
            if header_match:
                # 1. 首先处理并验证上一个完整的数据帧
                if current_timestamp is not None:
                    # 验证：这是否是一个包含 1024 个 SC 的完整帧？ (规则 #1 和 #3)
                    if len(current_sc_data) == NUM_SUBCARRIERS:
                        # 是，构建 h_vector
                        h_vector = np.zeros(NUM_SUBCARRIERS, dtype=np.complex128)
                        for sc_idx, value in current_sc_data.items():
                            h_vector[sc_idx] = value
                        # 保存 (frame_index, h_vector)
                        all_valid_frames.append((current_frame_index, h_vector))
                    else:
                        # 否，这是一个不完整的 slot，我们丢弃它
                        if len(current_sc_data) > 0: # 避免在文件开头或空块后报错
                            print(f"  INFO: Dropping incomplete slot (found {len(current_sc_data)}/{NUM_SUBCARRIERS} SCs) at frame {current_frame_index}.")

                # 2. 准备下一个新帧/Slot
                current_frame_index = int(header_match.group(1))
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
            all_valid_frames.append((current_frame_index, h_vector))
        elif len(current_sc_data) > 0:
            print(f"  INFO: Dropping final incomplete slot (found {len(current_sc_data)}/{NUM_SUBCARRIERS} SCs).")

    print(f"  Found {len(all_valid_frames)} total valid (1024-SC) frames in file.")
    return all_valid_frames


if __name__ == '__main__':
    # --- 1. 用户配置区域 ---
    # !! 确保这些路径和文件名是正确的 !!
    base_data_path = 'data' # 假设 .txt 文件在 'data' 目录中
    base_output_path = 'data' # !! 新的 V_FINAL 输出目录 !!
    
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)
        print(f"Created output directory: {base_output_path}")

    # 定义要处理的 40m 实验
    # 键 = *输出* 的 .mat 文件名
    # 值 = *输入* 的 .txt 文件列表
    experiments_to_process = {
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
        # ... 为您所有的 7 个实验添加条目 ...
        # 'ch_est_50m_5mps_autopilot.mat': [ ... ],
        # 'ch_est_50m_10mps_autopilot.mat': [ ... ],
    }

    try:
        # --- 2. 逐一处理每个实验 ---
        for output_filename, input_files_list in experiments_to_process.items():
            print(f"\n--- Processing Experiment: {output_filename} ---")
            
            all_frames_dict = {} # 使用字典按 frame_index 自动去重 (规则 #4)
            
            for log_path in input_files_list:
                frames_list = process_log_file_V_FINAL(log_path)
                for frame_idx, h_vec in frames_list:
                    if frame_idx not in all_frames_dict: # 避免重复
                        all_frames_dict[frame_idx] = h_vec
            
            if not all_frames_dict:
                print(f"  No valid data found for {output_filename}, skipping.")
                print("-" * 70)
                continue

            # --- 3. 排序并检测丢帧 (Padding) (规则 #4) ---
            print(f"  Found {len(all_frames_dict)} total unique frames. Sorting and checking for drops...")
            sorted_frame_indices = sorted(all_frames_dict.keys())
            
            padded_h_vectors = []
            if not sorted_frame_indices:
                 print("  No frames to process after sorting.")
                 continue

            last_frame_idx = sorted_frame_indices[0] - 1 # 确保第一帧总被添加
            
            for frame_idx in sorted_frame_indices:
                # 计算帧差异，包括回绕
                if frame_idx >= last_frame_idx:
                    frame_diff = frame_idx - last_frame_idx
                else: # 回绕 (例如, 从 1023 -> 0)
                    frame_diff = (frame_idx + FRAMES_PER_CYCLE) - last_frame_idx
                
                # 如果丢帧，插入 0
                if frame_diff > 1:
                    num_dropped = frame_diff - 1
                    print(f"  WARNING: Detected drop of {num_dropped} frame(s) between frame {last_frame_idx} and {frame_idx}. Inserting padding.")
                    # 创建 (num_dropped, 1024) 的 0 矩阵
                    padding = np.zeros((num_dropped, NUM_SUBCARRIERS), dtype=np.complex128)
                    padded_h_vectors.extend(padding) # 添加多个 0 向量
                
                # 添加当前帧
                padded_h_vectors.append(all_frames_dict[frame_idx])
                last_frame_idx = frame_idx

            # --- 4. 堆叠、过滤、保存 ---
            H_matrix_full = np.stack(padded_h_vectors) # Shape [T_total_padded, 1024]
            print(f"  Padded matrix shape: {H_matrix_full.shape}")

            # --- 5. 将 *完整* 的 2D 复数矩阵保存 ---
            # 我们不再在这里过滤活跃子载波
            output_mat_path = os.path.join(base_output_path, output_filename)
            print(f"  Saving FULL matrix {H_matrix_full.shape} to {output_mat_path}")
            scipy.io.savemat(output_mat_path, {
                'H_matrix_full': H_matrix_full 
            })
            print("-" * 70)

        print("\n--- 所有文件处理完成！---")

    except (FileNotFoundError, ValueError) as e:
        print(f"\n错误: {e}")
    except Exception as e:
        print(f"\n发生未知错误: {e}")
        traceback.print_exc()