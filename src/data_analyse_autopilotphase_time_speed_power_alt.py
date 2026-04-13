import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
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

def plot_power_speed_altitude_profile(gps_df, channel_df, mission_name):
    """
    为单次飞行任务绘制包含功率分布(点云)、平均功率、速度和高度的详细时间剖面图。
    """
    print(f"\n--- 正在为任务 '{mission_name}' (巡航阶段) 生成详细时间剖面图 ---")
    fig, ax1 = plt.subplots(figsize=(18, 9))

    # 计算每个信道测量点的功率dB值
    channel_df['power_db'] = 10 * np.log10(channel_df['avg_power'])

    # --- 开始绘图 ---
    color1 = 'darkviolet'
    ax1.set_xlabel('Time (Europe/Berlin)', fontsize=14)
    ax1.set_ylabel('Channel Power (dB)', color=color1, fontsize=14)
    
    # --- 核心修正：直接在真实的时间戳上绘制散点图 ---
    ax1.scatter(channel_df.index, channel_df['power_db'], color=color1, alpha=0.15, s=10, label='Instantaneous Power')
    
    # 绘制平均功率趋势线
    mean_power_trend = channel_df['power_db'].resample('1S').mean()
    ax1.plot(mean_power_trend.index, mean_power_trend, color='purple', lw=2.5, label='Average Power Trend')

    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 绘制速度
    ax2 = ax1.twinx()
    color2 = 'tomato'
    ax2.set_ylabel('Groundspeed (m/s)', color=color2, fontsize=14)
    ax2.plot(gps_df.index, gps_df['groundSpeed'], color=color2, label='Groundspeed', lw=2, alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # 绘制高度
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.08))
    color3 = 'deepskyblue'
    ax3.set_ylabel('Relative Altitude (m)', color=color3, fontsize=14)
    ax3.plot(gps_df.index, gps_df['altitudeRelative'], color=color3, label='Altitude', lw=2)
    ax3.tick_params(axis='y', labelcolor=color3)

    fig.suptitle(f'Cruise Phase Time Profile for Mission: {mission_name}', fontsize=18)
    
    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax3.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper right')
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax1.xaxis.set_major_locator(plt.MaxNLocator(15))
    fig.autofmt_xdate()

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.show()

def plot_flight_path_heatmap(df, mission_name):
    """在2D地图上绘制带功率热力和速度大小的飞行轨迹。"""
    print(f"\n--- 正在为任务 '{mission_name}' (巡航阶段) 生成地理热力图 ---")
    plt.figure(figsize=(14, 12))
    
    min_speed = df['groundSpeed'].min()
    max_speed = df['groundSpeed'].max()
    speed_range = max_speed - min_speed
    
    if speed_range > 0.1:
        speeds_normalized = (df['groundSpeed'].clip(0) - min_speed) / speed_range
        sizes = 15 + 250 * (speeds_normalized ** 2)
    else:
        sizes = 50
    
    power_db_filtered = df['power_db'].dropna()
    vmin = power_db_filtered.quantile(0.05)
    vmax = power_db_filtered.quantile(0.95)
    
    sc = plt.scatter(df['gps.lon'], df['gps.lat'], c=df['power_db'], cmap='inferno',
                     s=sizes, vmin=vmin, vmax=vmax,
                     label='Flight Path', alpha=0.7)
    
    plt.plot(df['gps.lon'].iloc[0], df['gps.lat'].iloc[0], 'go', markersize=12, label='Start', markeredgecolor='black')
    plt.plot(df['gps.lon'].iloc[-1], df['gps.lat'].iloc[-1], 'rX', markersize=12, label='End', markeredgecolor='black')
    
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.title(f'Cruise Phase Flight Path for {mission_name}: Power (Color) & Speed (Size)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    
    cbar = plt.colorbar(sc, shrink=0.7)
    cbar.set_label('Average Channel Power (dB)')
    
    size_legend_speeds = np.linspace(min_speed, max_speed, 4)
    legend_sizes = 15 + 250 * ((size_legend_speeds - min_speed) / speed_range)**2 if speed_range > 0.1 else [50]*4
    
    legend_handles = [plt.scatter([],[], s=s, edgecolors='k', color='grey') for s in legend_sizes]
    legend_labels = [f'{speed:.1f} m/s' for speed in size_legend_speeds]
    
    orig_legend = plt.legend(loc='upper left')
    plt.gca().add_artist(orig_legend)
    
    plt.legend(legend_handles, legend_labels, title="Groundspeed", loc="lower left")
    
    plt.tight_layout()
    plt.show()

def analyze_and_plot_correlation(df, mission_name):
    """计算并可视化关键参数之间的相关性。"""
    print(f"\n" + "="*50)
    print(f"开始为任务 '{mission_name}' (巡航阶段) 进行关键参数相关性分析...")
    
    columns_for_corr = ['power_db', 'groundSpeed', 'altitudeRelative', 'distance_from_start']
    corr_df = df[columns_for_corr].dropna()

    corr_df.rename(columns={
        'power_db': 'Channel Power (dB)',
        'groundSpeed': 'Groundspeed (m/s)',
        'altitudeRelative': 'Altitude (m)',
        'distance_from_start': 'Distance (m)'
    }, inplace=True)

    correlation_matrix = corr_df.corr()

    print("\n参数相关性矩阵:")
    print(correlation_matrix)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(f'Correlation Matrix for Mission (Cruise Phase): {mission_name}', fontsize=16)
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
    # 在这里定义您要分析的单次飞行任务的文件路径和巡航高度
    mission_name = "60m, 15-20m/s"
    target_cruise_altitude = 60  # ★★★ 目标巡航高度 (米) ★★★
    altitude_margin = 5        # ★★★ 高度容差 (米)，例如上下5米 ★★★
    
    channel_log_path = os.path.join(os.getcwd(), 'data', 'ch_est_pavel_60m_15mps_20mps_from_big_rec.txt')
    gps_log_path = os.path.join(os.getcwd(), 'data', 'pavel_60m_15mps_20mps.csv')

    try:
        # --- 2. 解析数据 ---
        channel_df_full = parse_channel_log(channel_log_path)
        gps_df_full = parse_gps_log(gps_log_path)

        # --- 3. 【新功能】根据巡航高度筛选数据 ---
        print("\n" + "="*50)
        print(f"正在根据目标巡航高度 {target_cruise_altitude}m (容差 ±{altitude_margin}m) 筛选巡航阶段...")
        
        # 筛选出巡航阶段的GPS数据
        cruise_gps_df = gps_df_full[
            (gps_df_full['altitudeRelative'] >= target_cruise_altitude - altitude_margin) &
            (gps_df_full['altitudeRelative'] <= target_cruise_altitude + altitude_margin)
        ]

        if cruise_gps_df.empty:
            raise ValueError(f"在指定的高度范围 [{target_cruise_altitude - altitude_margin}m, {target_cruise_altitude + altitude_margin}m] 内未找到任何GPS数据。")

        # 获取巡航阶段的开始和结束时间
        cruise_start_time = cruise_gps_df.index.min()
        cruise_end_time = cruise_gps_df.index.max()
        print(f"巡航阶段已确定: 从 {cruise_start_time} 到 {cruise_end_time}")
        
        # 筛选出巡航阶段的信道数据
        cruise_channel_df = channel_df_full[
            (channel_df_full.index >= cruise_start_time) &
            (channel_df_full.index <= cruise_end_time)
        ]
        
        if cruise_channel_df.empty:
            raise ValueError("在确定的巡航时间段内，未能找到任何对应的信道数据。")
        
        print(f"已成功筛选出 {len(cruise_channel_df)} 个巡航阶段的信道数据点。")
        print("="*50)

        # --- 4. 同步巡航阶段的数据 ---
        print("正在同步巡航阶段的GPS和信道数据...")
        merged_df_cruise = pd.merge_asof(
            left=cruise_gps_df.sort_index(),
            right=cruise_channel_df.sort_index(),
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta('1s')
        )
        merged_df_cruise.dropna(subset=['avg_power'], inplace=True)
        
        if merged_df_cruise.empty:
             raise ValueError("未能匹配任何巡航阶段的数据点。")
        
        print(f"数据同步完成，共得到 {len(merged_df_cruise)} 个匹配的巡航数据点。")

        # --- 5. 计算衍生指标 ---
        print("正在计算衍生指标...")
        merged_df_cruise['power_db'] = 10 * np.log10(merged_df_cruise['avg_power'])
        start_lat = merged_df_cruise['gps.lat'].iloc[0]
        start_lon = merged_df_cruise['gps.lon'].iloc[0]
        merged_df_cruise['distance_from_start'] = merged_df_cruise.apply(
            lambda row: haversine_distance(start_lat, start_lon, row['gps.lat'], row['gps.lon']),
            axis=1
        )
        print("指标计算完成。")

        # --- 6. 生成图表 ---
        # 注意：现在所有绘图函数都使用筛选后的巡航数据
        plot_power_speed_altitude_profile(cruise_gps_df, cruise_channel_df, mission_name)
        plot_flight_path_heatmap(merged_df_cruise, mission_name)
        analyze_and_plot_correlation(merged_df_cruise, mission_name)

    except (FileNotFoundError, ValueError) as e:
        print(f"\n错误: {e}")
    except Exception as e:
        print(f"\n发生未知错误: {e}")
