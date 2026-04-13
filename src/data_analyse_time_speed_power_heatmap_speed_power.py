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
    解析CSV格式的GPS飞行日志，并进行数据清洗。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"GPS文件未找到: {file_path}")
    print(f"正在解析GPS文件: {os.path.basename(file_path)}...")
    df = pd.read_csv(file_path)
    
    cols_to_convert = [
        'gps.lat', 'gps.lon', 'altitudeAMSL', 'groundSpeed', 'altitudeRelative',
        'localPosition.vx', 'localPosition.vy', 'localPosition.vz'
    ]
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"!!! 警告: GPS文件中缺少列 '{col}'")

    original_rows = len(df)
    df.dropna(subset=['gps.lat', 'gps.lon', 'altitudeAMSL', 'localPosition.vx', 'localPosition.vy', 'localPosition.vz'], inplace=True)
    if len(df) < original_rows:
        print(f" -> 已清理 {original_rows - len(df)} 行无效的GPS记录。")

    df['timestamp'] = pd.to_datetime(df['Timestamp'])
    df['timestamp'] = df['timestamp'].dt.tz_localize('Europe/Berlin', ambiguous='infer')
    df = df.set_index('timestamp').drop(columns=['Timestamp'])
    print(f" -> 完成，共找到 {len(df)} 条有效记录。")
    return df

def calculate_radial_velocity(df, gs_lat, gs_lon, gs_alt):
    """
    计算无人机相对于地面基站的位置矢量和【绝对】相对径向速度。
    """
    print("正在计算相对径向速度...")
    
    df['pos_x_m'] = (df['gps.lon'] - gs_lon) * 40075000 * np.cos(np.radians(gs_lat)) / 360
    df['pos_y_m'] = (df['gps.lat'] - gs_lat) * 40008000 / 360
    df['pos_z_m'] = df['altitudeAMSL'] - gs_alt
    
    pos_vectors = df[['pos_x_m', 'pos_y_m', 'pos_z_m']].values
    vel_vectors = df[['localPosition.vx', 'localPosition.vy', 'localPosition.vz']].values
    
    dot_product = np.sum(vel_vectors * pos_vectors, axis=1)
    distance = np.linalg.norm(pos_vectors, axis=1)
    
    radial_velocity_with_direction = np.divide(dot_product, distance, out=np.zeros_like(dot_product), where=distance!=0)
    
    df['radial_velocity_abs'] = np.abs(radial_velocity_with_direction)
    
    print("相对径向速度计算完成。")
    return df

def plot_full_profile(gps_df_full, channel_df_full, merged_df, mission_name):
    """
    为单次飞行任务绘制包含所有关键参数的最终时间剖面图。
    """
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
    color2_radial = 'limegreen'
    ax2.set_ylabel('Speed (m/s)', color=color2_ground, fontsize=14)
    ax2.plot(plot_merged_df.index, plot_merged_df['groundSpeed'], color=color2_ground, label='Groundspeed', lw=2, alpha=0.9)
    ax2.plot(plot_merged_df.index, plot_merged_df['radial_velocity_abs'], color=color2_radial, label='Abs. Radial Velocity', lw=2, linestyle=':', alpha=0.9)
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
    plt.show()

def plot_flight_path_heatmap(df, mission_name):
    """
    【已更新】在2D地图上绘制带功率热力、径向速度大小和径向速度标注的飞行轨迹。
    """
    print(f"\n--- 正在为任务 '{mission_name}' 生成带有径向速度标注的地理热力图 ---")
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # --- 核心修正 1：点的大小现在由【径向速度】决定 ---
    min_radial_vel = df['radial_velocity_abs'].min()
    max_radial_vel = df['radial_velocity_abs'].max()
    radial_vel_range = max_radial_vel - min_radial_vel
    
    if radial_vel_range > 0.1:
        radial_vel_normalized = (df['radial_velocity_abs'] - min_radial_vel) / radial_vel_range
        sizes = 15 + 250 * (radial_vel_normalized ** 2)
    else:
        sizes = 50
    
    power_db_filtered = df['power_db'].dropna()
    vmin = power_db_filtered.quantile(0.05)
    vmax = power_db_filtered.quantile(0.95)
    
    sc = ax.scatter(df['gps.lon'], df['gps.lat'], c=df['power_db'], cmap='inferno',
                     s=sizes, vmin=vmin, vmax=vmax,
                     label='Flight Path', alpha=0.7)
    
    ax.plot(df['gps.lon'].iloc[0], df['gps.lat'].iloc[0], 'go', markersize=12, label='Start', markeredgecolor='black')
    ax.plot(df['gps.lon'].iloc[-1], df['gps.lat'].iloc[-1], 'rX', markersize=12, label='End', markeredgecolor='black')
    
    max_radial_vel_idx = df['radial_velocity_abs'].idxmax()
    max_rv_point = df.loc[max_radial_vel_idx]
    ax.plot(max_rv_point['gps.lon'], max_rv_point['gps.lat'], '*', color='cyan', markersize=15, label=f'Max Radial Vel: {max_rv_point["radial_velocity_abs"]:.1f} m/s', markeredgecolor='black')
    
    # --- 核心修正 2：更新标注频率 ---
    annotation_step = 5 
    for i in range(0, len(df), annotation_step):
        point = df.iloc[i]
        ax.text(point['gps.lon'], point['gps.lat'], f"{point['radial_velocity_abs']:.1f}", fontsize=8, color='white', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.5))
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Flight Path for {mission_name}: Power (Color), Abs. Radial Vel. (Size & Labels)', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')
    
    cbar = fig.colorbar(sc, shrink=0.7, ax=ax)
    cbar.set_label('Average Channel Power (dB)')
    
    # --- 核心修正 3：更新图例以反映径向速度 ---
    size_legend_radial_vels = np.linspace(min_radial_vel, max_radial_vel, 4)
    legend_sizes = 15 + 250 * ((size_legend_radial_vels - min_radial_vel) / radial_vel_range)**2 if radial_vel_range > 0.1 else [50]*4
    
    legend_handles = [plt.scatter([],[], s=s, edgecolors='k', color='grey') for s in legend_sizes]
    legend_labels = [f'{vel:.1f} m/s' for vel in size_legend_radial_vels]
    
    orig_legend = ax.legend(loc='upper left')
    ax.add_artist(orig_legend)
    
    ax.legend(legend_handles, legend_labels, title="Abs. Radial Velocity", loc="lower left")
    
    plt.tight_layout()
    plt.show()


def analyze_and_plot_correlation(df, mission_name):
    """计算并可视化关键参数之间的相关性。"""
    print(f"\n" + "="*50)
    print(f"开始为任务 '{mission_name}' (整个飞行过程) 进行关键参数相关性分析...")
    
    columns_for_corr = [
        'power_db', 'groundSpeed', 'altitudeRelative', 'radial_velocity_abs',
        'power_change_rate', 'speed_change_rate'
    ]
    corr_df = df[columns_for_corr].dropna()

    corr_df.rename(columns={
        'power_db': 'Channel Power (dB)',
        'groundSpeed': 'Groundspeed (m/s)',
        'altitudeRelative': 'Altitude (m)',
        'radial_velocity_abs': 'Abs. Radial Velocity (m/s)',
        'power_change_rate': 'Power Fluctuation (dB)',
        'speed_change_rate': 'Speed Change (m/s)'
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
    mission_name = "40m, 10m/s"
    channel_log_path = os.path.join(os.getcwd(), 'data', 'ch_est_pavel_40m_10mps_from_big_rec_autopilot.txt')
    gps_log_path = os.path.join(os.getcwd(), 'data', 'pavel_40m_10mps.csv')

    try:
        # --- 2. 解析数据 ---
        channel_df_full = parse_channel_log(channel_log_path)
        gps_df_full = parse_gps_log(gps_log_path)

        # --- 3. 确定基站坐标 ---
        if not gps_df_full.empty:
            gs_lat = gps_df_full['gps.lat'].iloc[0]
            gs_lon = gps_df_full['gps.lon'].iloc[0]
            gs_alt = gps_df_full['altitudeAMSL'].iloc[0] + 3.0 # 高度加3米
            print("\n" + "="*50)
            print("根据GPS起点估算基站坐标:")
            print(f"  - 纬度 (Lat): {gs_lat}")
            print(f"  - 经度 (Lon): {gs_lon}")
            print(f"  - 海拔 (Alt): {gs_alt} m")
            print("="*50)
        else:
            raise ValueError("GPS数据为空，无法确定基站坐标。")

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
        
        merged_df.set_index('timestamp', inplace=True)

        print(f"数据同步完成，共得到 {len(merged_df)} 个匹配的数据点。")

        # --- 5. 计算衍生指标 ---
        print("正在计算衍生指标...")
        merged_df['power_db'] = 10 * np.log10(merged_df['avg_power'])
        
        merged_df = calculate_radial_velocity(merged_df, gs_lat, gs_lon, gs_alt)
        
        merged_df['power_change_rate'] = merged_df['power_db'].diff().abs()
        merged_df['speed_change_rate'] = merged_df['groundSpeed'].diff().abs()
        print("指标计算完成。")

        # --- 6. 生成图表和分析 ---
        plot_full_profile(gps_df_full, channel_df_full, merged_df, mission_name)
        plot_flight_path_heatmap(merged_df, mission_name)
        analyze_and_plot_correlation(merged_df, mission_name)

    except (FileNotFoundError, ValueError) as e:
        print(f"\n错误: {e}")
    except Exception as e:
        print(f"\n发生未知错误: {e}")
