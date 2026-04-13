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
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"GPS文件未找到: {file_path}")
    print(f"正在解析GPS CSV日志文件: {os.path.basename(file_path)}...")

    try:
        column_names = [
            'id', 'date', '0', 'TimeUS', 'I', 'Status', 'GMS', 'gWk', 'NSats',
            'Hdop', 'Lat', 'Lng', 'Alt', 'Spd', 'GCrs', 'VZ', 'Yaw', 'U'
        ]
        
        df = pd.read_csv(
            file_path,
            sep=',',
            decimal='.',
            engine='python',
            header=0,
            names=column_names,
            usecols=column_names
        )

        df.rename(columns={
            'date': 'Timestamp', 'Lat': 'gps.lat', 'Lng': 'gps.lon',
            'Alt': 'altitudeAMSL', 'Spd': 'groundSpeed'
        }, inplace=True)

        cols_to_convert = ['gps.lat', 'gps.lon', 'altitudeAMSL', 'groundSpeed']
        for col in cols_to_convert:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        original_rows = len(df)
        df.dropna(subset=['gps.lat', 'gps.lon', 'altitudeAMSL'], inplace=True)
        if len(df) < original_rows:
            print(f" -> 已清理 {original_rows - len(df)} 行无效的GPS记录。")
        
        if df.empty:
            raise ValueError("清理数据后，未能找到任何包含有效GPS坐标和海拔的行。")

        df['altitudeRelative'] = df['altitudeAMSL'] - df['altitudeAMSL'].iloc[0]
        df['timestamp'] = pd.to_datetime(df['Timestamp'])
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')
        df = df.set_index('timestamp').drop(columns=['Timestamp'])
        
        print(f" -> 完成，共找到 {len(df)} 条有效记录。")
        return df

    except Exception as e:
        print(f"解析GPS文件时出错: {e}")
        raise

def plot_full_profile(gps_df_full, channel_df_full, merged_df, mission_name):
    print(f"\n--- 正在为任务 '{mission_name}' 生成最终时间剖面图 ---")
    fig, ax1 = plt.subplots(figsize=(18, 10))
    plot_merged_df = merged_df.copy()
    plot_merged_df.index = plot_merged_df.index.tz_localize(None)
    plot_channel_df = channel_df_full.copy()
    plot_channel_df.index = plot_channel_df.index.tz_localize(None)

    color1 = 'darkviolet'
    ax1.set_xlabel('Time (Local)', fontsize=14)
    ax1.set_ylabel('Channel Power (dB)', color=color1, fontsize=14)
    plot_channel_df['power_db_scatter'] = 10 * np.log10(plot_channel_df['avg_power'])
    ax1.scatter(plot_channel_df.index, plot_channel_df['power_db_scatter'], color=color1, alpha=0.1, s=10, label='Instantaneous Power')
    ax1.plot(plot_merged_df.index, plot_merged_df['power_db'], color='purple', lw=2.5, label='Average Power Trend')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax2 = ax1.twinx()
    color2_ground = 'tomato'
    ax2.set_ylabel('Groundspeed (m/s)', color=color2_ground, fontsize=14)
    ax2.plot(plot_merged_df.index, plot_merged_df['groundSpeed'], color=color2_ground, label='Groundspeed', lw=2, alpha=0.9)
    ax2.tick_params(axis='y', labelcolor=color2_ground)
    
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.08))
    color3 = 'deepskyblue'
    ax3.set_ylabel('Relative Altitude (m)', color=color3, fontsize=14)
    ax3.plot(plot_merged_df.index, plot_merged_df['altitudeRelative'], color=color3, label='Altitude', lw=2.5)
    ax3.tick_params(axis='y', labelcolor=color3)

    fig.suptitle(f'Full Time Profile for Mission: {mission_name}', fontsize=18)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax3.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper right')
    plot_start_time = plot_merged_df.index.min()
    plot_end_time = plot_merged_df.index.max()
    ax1.set_xlim(plot_start_time, plot_end_time)

    locator = mdates.AutoDateLocator(minticks=10, maxticks=20)
    formatter = mdates.ConciseDateFormatter(locator)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(f'{mission_name}_full_profile.png')
    print(f"时间剖面图已保存为 {mission_name}_full_profile.png")
    plt.close()

def plot_flight_path_heatmap(df, mission_name):
    print(f"\n--- 正在为任务 '{mission_name}' 生成带有地面速度标注的地理热力图 ---")
    fig, ax = plt.subplots(figsize=(16, 14))
    min_speed = df['groundSpeed'].min()
    max_speed = df['groundSpeed'].max()
    speed_range = max_speed - min_speed
    
    if speed_range > 0.1:
        sizes = 15 + 250 * (((df['groundSpeed'] - min_speed) / speed_range) ** 2)
    else:
        sizes = 50
    
    power_db_filtered = df['power_db'].dropna()
    vmin = power_db_filtered.quantile(0.05)
    vmax = power_db_filtered.quantile(0.95)
    
    sc = ax.scatter(df['gps.lon'], df['gps.lat'], c=df['power_db'], cmap='inferno', s=sizes, vmin=vmin, vmax=vmax, label='Flight Path', alpha=0.7)
    ax.plot(df['gps.lon'].iloc[0], df['gps.lat'].iloc[0], 'go', markersize=12, label='Start', markeredgecolor='black')
    ax.plot(df['gps.lon'].iloc[-1], df['gps.lat'].iloc[-1], 'rX', markersize=12, label='End', markeredgecolor='black')
    
    annotation_step = max(1, len(df) // 100)
    for i in range(0, len(df), annotation_step):
        point = df.iloc[i]
        ax.text(point['gps.lon'], point['gps.lat'], f"{point['groundSpeed']:.1f}", fontsize=8, color='white', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.5))
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Flight Path for {mission_name}: Power (Color), Groundspeed (Size & Labels)', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')
    
    cbar = fig.colorbar(sc, shrink=0.7, ax=ax)
    cbar.set_label('Average Channel Power (dB)')
    
    size_legend_speeds = np.linspace(min_speed, max_speed, 4)
    legend_sizes = 15 + 250 * (((size_legend_speeds - min_speed) / speed_range)**2) if speed_range > 0.1 else [50]*4
    legend_handles = [plt.scatter([],[], s=s, edgecolors='k', color='grey') for s in legend_sizes]
    legend_labels = [f'{vel:.1f} m/s' for vel in size_legend_speeds]
    orig_legend = ax.legend(loc='upper left')
    ax.add_artist(orig_legend)
    ax.legend(legend_handles, legend_labels, title="Groundspeed", loc="lower left")
    
    plt.tight_layout()
    plt.savefig(f'{mission_name}_flight_path_heatmap.png')
    print(f"地理热力图已保存为 {mission_name}_flight_path_heatmap.png")
    plt.close()

# ==============================================================================
#  主程序执行部分
# ==============================================================================
if __name__ == '__main__':
    mission_name = "Bolek_40m_5mps"
    
    if not os.path.exists('data'):
        os.makedirs('data')
        print("创建 'data' 文件夹。请将您的数据文件放入其中。")

    channel_log_path = os.path.join('data', 'channel_estimates_20250917_140636_bolek_40m_5mps.txt')
    gps_log_path = os.path.join('data', '2025-09-17_11-31-48b_bolek.csv')

    try:
        channel_df_full = parse_channel_log(channel_log_path)
        gps_df_full = parse_gps_log(gps_log_path)

        gs_lat = gps_df_full['gps.lat'].iloc[0]
        gs_lon = gps_df_full['gps.lon'].iloc[0]
        gs_alt = gps_df_full['altitudeAMSL'].iloc[0] + 5.0
        print("\n" + "="*50)
        print("根据GPS起点估算基站坐标:")
        print(f"  - 纬度 (Lat): {gs_lat}")
        print(f"  - 经度 (Lon): {gs_lon}")
        print(f"  - 海拔 (Alt): {gs_alt} m")
        print("="*50)

        print("正在同步GPS和信道数据...")
        merged_df = pd.merge_asof(
            left=gps_df_full.sort_index(),
            right=channel_df_full.sort_index(),
            left_index=True,
            right_index=True,
            direction='nearest',
            tolerance=pd.Timedelta('1s')
        )
        merged_df.dropna(subset=['avg_power'], inplace=True)
        if merged_df.empty:
             raise ValueError("数据同步失败。")
        print(f"数据同步完成，共得到 {len(merged_df)} 个匹配的数据点。")

        print("正在计算衍生指标...")
        merged_df['power_db'] = 10 * np.log10(merged_df['avg_power'])
        merged_df['power_change_rate'] = merged_df['power_db'].diff().abs()
        merged_df['speed_change_rate'] = merged_df['groundSpeed'].diff().abs()
        print("指标计算完成。")

        plot_full_profile(gps_df_full, channel_df_full, merged_df, mission_name)
        plot_flight_path_heatmap(merged_df, mission_name)
        
        # 移除了相关性分析以简化脚本
        print("\n分析完成。")

    except (FileNotFoundError, ValueError) as e:
        print(f"\n[错误]: {e}")
    except Exception as e:
        print(f"\n[发生未知错误]: {e}")