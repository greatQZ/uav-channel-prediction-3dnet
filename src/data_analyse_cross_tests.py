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
    处理柏林时区。
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
    
    # --- 核心修正：确保关键列是数值类型，并处理错误 ---
    cols_to_convert = ['gps.lat', 'gps.lon', 'groundSpeed', 'altitudeRelative']
    for col in cols_to_convert:
        if col in df.columns:
            # errors='coerce' 会将无法转换的字符串变为NaN (空值)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"!!! 警告: GPS文件中缺少列 '{col}'")

    # 删除因为转换失败而产生的、缺少GPS坐标的关键行
    original_rows = len(df)
    df.dropna(subset=['gps.lat', 'gps.lon'], inplace=True)
    if len(df) < original_rows:
        print(f" -> 已清理 {original_rows - len(df)} 行无效的GPS记录。")
    # --- 修正结束 ---

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

# ==============================================================================
#  核心功能函数 (已更新)
# ==============================================================================

def plot_multiple_flight_paths_heatmap(flight_data_list):
    """
    【新功能】在同一张2D地图上绘制多次飞行的轨迹热力图。
    """
    print("\n开始生成多飞行任务统一对比图...")
    plt.figure(figsize=(16, 14))
    
    markers = ['o', '^', 's', 'P', 'D', 'X'] # 为不同飞行任务准备不同的标记符号
    
    # --- 从字典中正确提取DataFrame ---
    # 统一所有数据的颜色和大小范围
    all_power_db = pd.concat([flight_data['data']['power_db'] for flight_data in flight_data_list if not flight_data['data'].empty]).dropna()
    all_speed = pd.concat([flight_data['data']['groundSpeed'] for flight_data in flight_data_list if not flight_data['data'].empty]).dropna()
    # --- 修正结束 ---
    
    vmin = all_power_db.quantile(0.05)
    vmax = all_power_db.quantile(0.95)
    min_speed, max_speed = all_speed.min(), all_speed.max()
    
    # 绘制每个飞行任务的轨迹
    for i, flight_data in enumerate(flight_data_list):
        df = flight_data['data']
        name = flight_data['name']
        marker = markers[i % len(markers)]

        if df.empty:
            continue

        # 计算点的大小
        speed_range = max_speed - min_speed
        if speed_range > 0.1:
            speeds_normalized = (df['groundSpeed'].clip(0) - min_speed) / speed_range
            sizes = 15 + 250 * (speeds_normalized ** 2)
        else:
            sizes = 50
        
        # 绘制散点图
        sc = plt.scatter(df['gps.lon'], df['gps.lat'], c=df['power_db'], cmap='inferno',
                         s=sizes, vmin=vmin, vmax=vmax,
                         marker=marker, label=name, alpha=0.7)

    # 配置图表
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.title('Multi-Flight Comparison: Power (Color), Speed (Size), Mission (Marker)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # 添加图例和颜色条
    plt.legend(title="mission", loc='upper left')
    cbar = plt.colorbar(sc, shrink=0.7, ax=plt.gca())
    cbar.set_label('Average Channel Power (dB)')
    
    plt.tight_layout()
    plt.show()

def analyze_and_plot_correlation(combined_df):
    """
    在所有飞行任务的合并数据上进行相关性分析。
    """
    print("\n" + "="*50)
    print("开始在【所有】飞行数据上进行关键参数相关性分析...")
    
    columns_for_corr = ['power_db', 'groundSpeed', 'altitudeRelative', 'distance_from_start']
    corr_df = combined_df[columns_for_corr].dropna()

    corr_df.rename(columns={
        'power_db': 'Channel Power (dB)',
        'groundSpeed': 'Groundspeed (m/s)',
        'altitudeRelative': 'Altitude (m)',
        'distance_from_start': 'Distance (m)'
    }, inplace=True)

    correlation_matrix = corr_df.corr()

    print("\n参数相关性矩阵 (综合所有飞行):")
    print(correlation_matrix)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Overall Correlation Matrix (All Flights Combined)', fontsize=16)
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
    # 在这里定义您所有要分析的飞行任务
    # 每个任务是一个字典，包含名称和两个文件的路径
    flight_experiments = [
        {
            "name": "40m, 10m/s",
            "channel_log": os.path.join(os.getcwd(), 'data', 'ch_est_pavel_40m_10mps_from_big_rec.txt'), # 示例文件名
            "gps_log": os.path.join(os.getcwd(), 'data', 'pavel_40m_10mps.csv')      # 示例文件名
        },
        {
            "name": "40m, 20m/s",
            "channel_log": os.path.join(os.getcwd(), 'data', 'ch_est_pavel_40m_20mps_lange_route_from_big_rec.txt'), # 示例文件名
            "gps_log": os.path.join(os.getcwd(), 'data', 'pavel_40ms_20mps_lange_Route.csv')      # 示例文件名
        },
        {
            "name": "60m, 10m/s",
            "channel_log": os.path.join(os.getcwd(), 'data', 'ch_est_pavel_60m_10mps_from_big_rec.txt'), # 示例文件名
            "gps_log": os.path.join(os.getcwd(), 'data', 'pavel_60m_10mps.csv')      # 示例文件名
        },
        {
            "name": "60m, 15-20m/s",
            "channel_log": os.path.join(os.getcwd(), 'data', 'ch_est_pavel_60m_15mps_20mps_from_big_rec.txt'), # 示例文件名
            "gps_log": os.path.join(os.getcwd(), 'data', 'pavel_60m_15mps_20mps.csv')      # 示例文件名
        },
        # ... 可以在这里添加更多的飞行任务 ...
    ]

    all_flight_data = []
    
    # --- 2. 批量处理每个飞行任务 ---
    for experiment in flight_experiments:
        print("\n" + "#"*70)
        print(f"### 开始处理任务: {experiment['name']} ###")
        print("#"*70)
        try:
            channel_df = parse_channel_log(experiment['channel_log'])
            gps_df = parse_gps_log(experiment['gps_log'])

            # 时间范围自检
            ch_start, ch_end = channel_df.index.min(), channel_df.index.max()
            gps_start, gps_end = gps_df.index.min(), gps_df.index.max()
            if max(ch_start, gps_start) >= min(ch_end, gps_end):
                print(f"!!! 警告：任务 '{experiment['name']}' 的文件时间范围不重叠，已跳过。")
                continue

            # 同步数据
            merged_df = pd.merge_asof(
                left=gps_df.sort_index(),
                right=channel_df.sort_index(),
                on='timestamp',
                direction='nearest',
                tolerance=pd.Timedelta('1s')
            )
            merged_df.dropna(subset=['avg_power'], inplace=True)
            
            if merged_df.empty:
                print(f"!!! 警告：任务 '{experiment['name']}' 未能匹配任何数据点，已跳过。")
                continue
            
            # 计算衍生指标
            merged_df['power_db'] = 10 * np.log10(merged_df['avg_power'])
            start_lat = merged_df['gps.lat'].iloc[0]
            start_lon = merged_df['gps.lon'].iloc[0]
            merged_df['distance_from_start'] = merged_df.apply(
                lambda row: haversine_distance(start_lat, start_lon, row['gps.lat'], row['gps.lon']),
                axis=1
            )
            
            all_flight_data.append({"name": experiment['name'], "data": merged_df})
            print(f"--- 任务 '{experiment['name']}' 处理成功 ---")

        except (FileNotFoundError, ValueError) as e:
            print(f"!!! 错误：处理任务 '{experiment['name']}' 时失败: {e}")
        except Exception as e:
            print(f"!!! 未知错误：处理任务 '{experiment['name']}' 时失败: {e}")

    # --- 3. 统一分析和可视化 ---
    if all_flight_data:
        # 统一地理热力图
        plot_multiple_flight_paths_heatmap(all_flight_data)
        
        # 统一相关性分析
        combined_df = pd.concat([flight['data'] for flight in all_flight_data])
        analyze_and_plot_correlation(combined_df)
    else:
        print("\n没有成功处理任何飞行任务，无法生成对比图。")

