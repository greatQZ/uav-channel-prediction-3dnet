# -*- coding: utf-8 -*-
import scipy.io
import numpy as np
import os

def convert_mat_1d_to_2d(input_file, output_file, num_features=576):
    """
    将 1D 存储的 .mat 数据转换为 2D 格式 [Time, Features]
    
    Args:
        input_file: 输入的 .mat 文件路径
        output_file: 输出的 .mat 文件路径
        num_features: 特征数 (子载波数)，例如 576。
                      脚本会自动计算 Time = Total_Length / num_features
    """
    
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        return

    print(f"--- 正在读取: {input_file} ---")
    try:
        mat = scipy.io.loadmat(input_file)
        
        # 1. 自动寻找数据变量名 (寻找最大的数组)
        # 排除 __header__, __version__ 等元数据
        valid_keys = [k for k in mat.keys() if not k.startswith('__')]
        if not valid_keys:
            print("错误: .mat 文件中没有有效变量")
            return
            
        # 假设数据是最大的那个变量
        var_name = max(valid_keys, key=lambda k: mat[k].size)
        raw_data = mat[var_name]
        
        print(f"找到变量: '{var_name}', 原始形状: {raw_data.shape}")
        
        # 2. 扁平化数据 (确保是 1D)
        data_flat = raw_data.flatten()
        total_len = data_flat.size
        print(f"数据总长度: {total_len}")
        
        # 3. 检查能否整除
        if total_len % num_features != 0:
            print(f"\n[错误] 无法重塑数据！")
            print(f"总长度 {total_len} 不能被特征数 {num_features} 整除。")
            print(f"余数: {total_len % num_features}")
            print("请检查 num_features (子载波数) 是否正确。")
            return
            
        num_timesteps = total_len // num_features
        
        # 4. 执行重塑 (Reshape)
        # 默认使用 C-order (行优先)，即先填满第一行的576个特征，再填下一行
        # 如果您的数据是列优先存储的，请改用 order='F'
        data_2d = data_flat.reshape(num_timesteps, num_features)
        
        print(f"转换成功！新形状: {data_2d.shape} (Time={num_timesteps}, Features={num_features})")
        
        # 5. 保存
        # 保存为通用的变量名 'H_active_matrix'，方便后续代码直接读取
        scipy.io.savemat(output_file, {'H_active_matrix': data_2d})
        print(f"已保存至: {output_file}")
        print(f"变量名已重命名为: 'H_active_matrix'")
        
    except Exception as e:
        print(f"发生异常: {e}")

if __name__ == "__main__":
    # --- 配置区域 ---
    
    # 您的 1D 源文件路径
    INPUT_PATH = 'data/my_H_real_data_bolek_for_training_testing.mat'  # <--- 请修改这里
    
    # 您想要保存的 2D 文件路径
    OUTPUT_PATH = 'data/my_H_real_data_bolek_for_training_testing_2d.mat' 
    
    # 特征数量 (子载波数)
    # 根据您之前的日志，这个数字应该是 576
    FEATURES = 576 
    
    # --- 执行转换 ---
    convert_mat_1d_to_2d(INPUT_PATH, OUTPUT_PATH, num_features=FEATURES)