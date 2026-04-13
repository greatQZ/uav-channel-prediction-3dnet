import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, sin, cos, sqrt, atan2

# ==============================================================================
#  函数定义部分
# ==============================================================================

def parse_channel_log(file_path):
    """
    解析包含高精度时间戳的SRS信道估计文件。
    此版本会正确处理柏林时区。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"信道文件未找到: {file_path}")
    print(f"正在解析信道文件: {os.path.basename(file_path)}...")
    header_pattern = re.compile(r"SRS Frame .* Real: ([\d\.]+)")
    sc_pattern = re.compile(r"Sc \d+: Re = (-?\d+), Im = (-?\d+)")
    records = []
    current_timestamp = None
    current_powers = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            header_match = header_pattern.search(line)
            sc_match = sc_pattern.search(line)
            if header_match:
                if current_timestamp and current_powers:
                    records.append({'timestamp': current_timestamp, 'avg_power': np.mean(current_powers)})
                ts_unix_nano = float(header_match.group(1))
                ts_utc = pd.to_datetime(ts_unix_nano, unit='s', utc=True)
                current_timestamp = ts_utc.tz_convert('Europe/Berlin')
                current_powers = []
            elif sc_match and current_timestamp:
                re_val, im_val = int(sc_match.group(1)), int(sc_match.group(2))
                power = re_val**2 + im_val**2
                if power > 0:
                    current_powers.append(power)
    if current_timestamp and current_powers:
        records.append({'timestamp': current_timestamp, 'avg_power': np.mean(current_powers)})
    if not records:
        raise ValueError("未能在信道文件中解析出任何有效数据。")
    df = pd.DataFrame(records).set_index('timestamp')
    print(f" -> 完成，共找到 {len(df)} 个有效时间快照。")
    return df

def parse_gps_log(file_path):
    """
    解析CSV格式的GPS飞行日志。
    此版本会正确处理柏林时区，并强制转换关键列为数值类型以进行数据清洗。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"GPS文件未找到: {file_path}")
    print(f"正在解析GPS文件: {os.path.basename(file_path)}...")
    df = pd.read_csv(file_path)
    
    cols_to_convert = ['gps.lat', 'gps.lon', 'groundSpeed', 'altitudeRelative']
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"!!! 警告: GPS文件中缺少列 '{col}'")

    original_rows = len(df)
    df.dropna(subset=['gps.lat', 'gps.lon'], inplace=True)
    if len(df) < original_rows:
        print(f" -> 已清理 {original_rows - len(df)} 行无效的GPS记录。")

    df['timestamp'] = pd.to_datetime(df['Timestamp'])
    df['timestamp'] = df['timestamp'].dt.tz_localize('Europe/Berlin', ambiguous='infer')
    df = df.set_index('timestamp').drop(columns=['Timestamp'])
    print(f" -> 完成，共找到 {len(df)} 条有效记录。")
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    """计算两个GPS坐标点之间的距离（单位：米）。"""
    R = 6371000
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def analyze_and_plot_correlation(df, mission_name):
    """计算并可视化关键参数之间的相关性。"""
    print(f"\n" + "="*50)
    print(f"开始为任务 '{mission_name}' (整个飞行过程) 进行关键参数相关性分析...")
    
    # 定义用于分析的列
    columns_for_corr = [
        'power_db', 'groundSpeed', 'altitudeRelative', 'distance_from_start',
        'power_change_rate', 'speed_change_rate', 'distance_change_rate'
    ]
    corr_df = df[columns_for_corr].dropna()

    corr_df.rename(columns={
        'power_db': 'Channel Power (dB)',
        'groundSpeed': 'Groundspeed (m/s)',
        'altitudeRelative': 'Altitude (m)',
        'distance_from_start': 'Distance (m)',
        'power_change_rate': 'Power Fluctuation (dB)',
        'speed_change_rate': 'Speed Change (m/s)',
        'distance_change_rate': 'Distance Change (m)'
    }, inplace=True)

    correlation_matrix = corr_df.corr()

    print("\n参数相关性矩阵:")
    print(correlation_matrix)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(f'Correlation Matrix for Mission (Entire Flight): {mission_name}', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    print("="*50)

# ==============================================================================
#  主程序执行部分
# ==============================================================================
if __name__ == '__main__':
    # --- 1. 用户配置区域 ---
    # 在这里定义您要分析的单次飞行任务的文件路径
    mission_name = "60m, 10m/s"
    channel_log_path = os.path.join(os.getcwd(), 'data', 'ch_est_pavel_60m_10mps_from_big_rec.txt')
    gps_log_path = os.path.join(os.getcwd(), 'data', 'pavel_60m_10mps.csv')

    try:
        # --- 2. 解析数据 ---
        channel_df_full = parse_channel_log(channel_log_path)
        gps_df_full = parse_gps_log(gps_log_path)

        # --- 3. 数据时间范围自检 ---
        print("\n" + "="*50)
        print("开始进行数据时间范围自检...")
        
        ch_start, ch_end = channel_df_full.index.min(), channel_df_full.index.max()
        gps_start, gps_end = gps_df_full.index.min(), gps_df_full.index.max()

        print(f"信道数据时间范围: {ch_start}  TO  {ch_end}")
        print(f"GPS 数据时间范围: {gps_start}  TO  {gps_end}")

        overlap_start = max(ch_start, gps_start)
        overlap_end = min(ch_end, gps_end)

        if overlap_start >= overlap_end:
            raise ValueError("!!! 严重错误：两个文件的时间范围没有重叠！!!!")
        else:
            print(f"\n✅ 自检通过：数据存在重叠区域: {overlap_start} TO {overlap_end}")
            print("="*50)
            
            # --- 4. 同步整个飞行过程的数据 ---
            print("正在同步整个飞行过程的GPS和信道数据...")
            merged_df = pd.merge_asof(
                left=gps_df_full.sort_index(),
                right=channel_df_full.sort_index(),
                on='timestamp',
                direction='nearest',
                tolerance=pd.Timedelta('1s')
            )
            merged_df.dropna(subset=['avg_power'], inplace=True)
            
            if merged_df.empty:
                 raise ValueError("未能匹配任何数据点。")
            
            print(f"数据同步完成，共得到 {len(merged_df)} 个匹配的数据点。")

            # --- 5. 计算衍生指标 ---
            print("正在计算衍生指标...")
            merged_df['power_db'] = 10 * np.log10(merged_df['avg_power'])
            start_lat = merged_df['gps.lat'].iloc[0]
            start_lon = merged_df['gps.lon'].iloc[0]
            merged_df['distance_from_start'] = merged_df.apply(
                lambda row: haversine_distance(start_lat, start_lon, row['gps.lat'], row['gps.lon']),
                axis=1
            )
            
            # 计算变化率指标
            merged_df['power_change_rate'] = merged_df['power_db'].diff().abs()
            merged_df['speed_change_rate'] = merged_df['groundSpeed'].diff().abs()
            merged_df['distance_change_rate'] = merged_df['distance_from_start'].diff().abs()
            print("变化率指标计算完成。")
            print("指标计算完成。")

            # --- 6. 生成相关性分析 ---
            analyze_and_plot_correlation(merged_df, mission_name)

    except (FileNotFoundError, ValueError) as e:
        print(f"\n错误: {e}")
    except Exception as e:
        print(f"\n发生未知错误: {e}")
