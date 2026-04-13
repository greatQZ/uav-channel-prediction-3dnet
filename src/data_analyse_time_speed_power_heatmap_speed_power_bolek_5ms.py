import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from math import radians, sin, cos, sqrt, atan2
import csv 
import traceback
import pytz # <-- 解决方案：导入 Pytz

# ==============================================================================
#  函数定义部分
# ==============================================================================

def parse_channel_log(file_path):
    """
    解析信道日志。
    【正确版本】UNIX时间戳 = UTC。 转换为柏林时间。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"信道文件未找到: {file_path}")
    print(f"正在解析信道文件: {os.path.basename(file_path)}...")
    header_pattern = re.compile(r"SRS Frame .*?, Real: ([\d\.]+)")
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
    【“干净”版本】GPS时间戳 = 柏林时间。
    使用 Pandas 默认解析器，并让 Pandas *自动推断* 时间格式。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"GPS文件未找到: {file_path}")
    print(f"正在解析GPS文件: {os.path.basename(file_path)}...")

    try:
        print(" -> 使用标准 'c' 引擎进行CSV解析...")
        df = pd.read_csv(file_path, on_bad_lines='skip')
    except Exception as e:
        print(f"!!! C引擎读取CSV时发生错误: {e}。尝试回退到 'python' 引擎...")
        try:
             df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
        except Exception as e2:
             print(f"!!! Python 引擎也失败了: {e2}")
             raise e2

    if 'Timestamp' not in df.columns:
         print(f"!!! 错误: CSV文件中未找到 'Timestamp' 列。")
         print(f" -> 读取到的列为: {df.columns.tolist()}")
         raise ValueError("CSV文件中未找到 'Timestamp' 列。")

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
    print(f" -> 初始读取到 {original_rows} 行数据。")
    df.dropna(subset=['gps.lat', 'gps.lon', 'altitudeAMSL', 'localPosition.vx', 'localPosition.vy', 'localPosition.vz'], inplace=True)

    df['Timestamp'] = df['Timestamp'].astype(str).str.strip()
    print(" -> 正在解析 'Timestamp' 列...")
    df['timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    rows_before_ts_drop = len(df)
    df.dropna(subset=['timestamp'], inplace=True)
    rows_after_ts_drop = len(df)
    if rows_before_ts_drop > rows_after_ts_drop:
        print(f" -> 在时间戳转换/清理中移除了 {rows_before_ts_drop - rows_after_ts_drop} 行。")

    if df.empty:
        raise ValueError("在清理时间戳转换失败的行之后，没有剩余的有效GPS记录。")

    df['timestamp'] = df['timestamp'].dt.tz_localize('Europe/Berlin', ambiguous='infer')
    df = df.set_index('timestamp').drop(columns=['Timestamp'])
    cleaned_rows = len(df)
    total_cleaned = original_rows - cleaned_rows
    print(f" -> (总共清理了 {total_cleaned} 行无效数据)")
    print(f" -> 清理后，共找到 {len(df)} 条有效记录。")

    if cleaned_rows == 0: 
        raise ValueError("在清理后，没有剩余的有效GPS记录。")
    return df


# --- 其他绘图和计算函数 (英文注释) ---
def calculate_radial_velocity(df, gs_lat, gs_lon, gs_alt):
    print("Calculating relative radial velocity...")
    df['pos_x_m'] = (df['gps.lon'] - gs_lon) * 40075000 * np.cos(np.radians(gs_lat)) / 360
    df['pos_y_m'] = (df['gps.lat'] - gs_lat) * 40008000 / 360
    df['pos_z_m'] = df['altitudeAMSL'] - gs_alt
    pos_vectors = df[['pos_x_m', 'pos_y_m', 'pos_z_m']].values
    vel_vectors = df[['localPosition.vx', 'localPosition.vy', 'localPosition.vz']].values
    dot_product = np.sum(vel_vectors * pos_vectors, axis=1)
    distance = np.linalg.norm(pos_vectors, axis=1)
    distance[distance == 0] = 1e-6
    radial_velocity_with_direction = np.divide(dot_product, distance, out=np.zeros_like(dot_product), where=distance!=0)
    df['radial_velocity_abs'] = np.abs(radial_velocity_with_direction)
    print("Relative radial velocity calculation complete.")
    return df

def plot_full_profile(gps_df_full, channel_df_full, merged_df, mission_name):
    print(f"\n--- Generating Full Time Profile for mission '{mission_name}' ---")
    fig, ax1 = plt.subplots(figsize=(18, 10))
    plot_merged_df = merged_df.copy()
    if not isinstance(plot_merged_df.index, pd.DatetimeIndex):
         plot_merged_df.index = pd.to_datetime(plot_merged_df.index)
    if not isinstance(channel_df_full.index, pd.DatetimeIndex):
         channel_df_full.index = pd.to_datetime(channel_df_full.index)
    plot_channel_df = channel_df_full.copy()

    color1 = 'darkviolet'
    ax1.set_xlabel('Time (Berlin Local Time)', fontsize=14)
    ax1.set_ylabel('Channel Power (dB)', color=color1, fontsize=14)
    valid_power = plot_channel_df['avg_power'] > 0
    plot_channel_df.loc[valid_power, 'power_db_scatter'] = 10 * np.log10(plot_channel_df.loc[valid_power, 'avg_power'])
    plot_channel_df.loc[~valid_power, 'power_db_scatter'] = np.nan
    ax1.scatter(plot_channel_df.index, plot_channel_df['power_db_scatter'], color=color1, alpha=0.1, s=10, label='Instantaneous Power')

    ax1.plot(plot_merged_df.index, plot_merged_df['power_db'], color='purple', lw=2.5, label='Average Power Trend')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    ax2 = ax1.twinx()
    color2_ground = 'tomato'
    color2_radial = 'limegreen'
    ax2.set_ylabel('Speed (m/s)', color=color2_ground, fontsize=14)
    if 'groundSpeed' in plot_merged_df.columns:
         ax2.plot(plot_merged_df.index, plot_merged_df['groundSpeed'], color=color2_ground, label='Ground Speed', lw=2, alpha=0.9)
    if 'radial_velocity_abs' in plot_merged_df.columns:
         ax2.plot(plot_merged_df.index, plot_merged_df['radial_velocity_abs'], color=color2_radial, label='Absolute Radial Velocity', lw=2, linestyle=':', alpha=0.9)
    ax2.tick_params(axis='y', labelcolor=color2_ground)
    
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.08))
    color3 = 'deepskyblue'
    ax3.set_ylabel('Relative Altitude (m)', color=color3, fontsize=14)
    if 'altitudeRelative' in plot_merged_df.columns:
         ax3.plot(plot_merged_df.index, plot_merged_df['altitudeRelative'], color=color3, label='Altitude', lw=2.5)
    ax3.tick_params(axis='y', labelcolor=color3)
    
    fig.suptitle(f'Mission Full Time Profile: {mission_name}', fontsize=18)
    handles = []; labels = []
    for ax in [ax1, ax2, ax3]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h); labels.extend(l)
    if handles:
        by_label = dict(zip(labels, handles))
        ax3.legend(by_label.values(), by_label.keys(), loc='upper right')

    if not plot_merged_df.empty:
        plot_start_time = plot_merged_df.index.min()
        plot_end_time = plot_merged_df.index.max()
        ax1.set_xlim(plot_start_time, plot_end_time)

    # =======================================================================
    #                       ↓↓↓ 时区修复处 ↓↓↓
    # =======================================================================
    # 1. 定义柏林时区
    berlin_tz = pytz.timezone('Europe/Berlin')
    
    # 2. 将时区信息传递给 Locator 和 Formatter
    locator = mdates.AutoDateLocator(minticks=10, maxticks=20, tz=berlin_tz)
    formatter = mdates.ConciseDateFormatter(locator, tz=berlin_tz)
    # =======================================================================
    
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.show()


def plot_flight_path_heatmap(df, mission_name):
    print(f"\n--- Generating Flight Path Heatmap for mission '{mission_name}' ---")
    fig, ax = plt.subplots(figsize=(16, 14))
    if df.empty or 'radial_velocity_abs' not in df.columns or df['radial_velocity_abs'].isnull().all() or 'gps.lon' not in df.columns or 'gps.lat' not in df.columns:
        print("!!! WARNING: Cannot generate heatmap, missing essential GPS or radial velocity data."); plt.close(fig); return
    min_radial_vel = df['radial_velocity_abs'].min(); max_radial_vel = df['radial_velocity_abs'].max()
    radial_vel_range = max_radial_vel - min_radial_vel
    sizes = 50 if radial_vel_range <= 0.1 else 15 + 250 * (((df['radial_velocity_abs'] - min_radial_vel) / radial_vel_range) ** 2)
    if 'power_db' not in df.columns or df['power_db'].isnull().all():
        print("!!! WARNING: Cannot colorize heatmap by power."); cmap = None; c_val = 'blue'; vmin=None; vmax=None
    else:
        cmap = 'inferno'; c_val = df['power_db']; power_db_filtered = df['power_db'].dropna()
        try:
            vmin = power_db_filtered.quantile(0.05) if not power_db_filtered.empty else None
            vmax = power_db_filtered.quantile(0.95) if not power_db_filtered.empty else None
            if vmin == vmax: vmin = vmin - 1 if vmin is not None else None; vmax = vmax + 1 if vmax is not None else None
        except IndexError: vmin = None; vmax = None

    sc = ax.scatter(df['gps.lon'], df['gps.lat'], c=c_val, cmap=cmap, s=sizes, vmin=vmin, vmax=vmax, label='Flight Path', alpha=0.7)
    ax.plot(df['gps.lon'].iloc[0], df['gps.lat'].iloc[0], 'go', markersize=12, label='Start', markeredgecolor='black')
    ax.plot(df['gps.lon'].iloc[-1], df['gps.lat'].iloc[-1], 'rX', markersize=12, label='End', markeredgecolor='black')
    if not df['radial_velocity_abs'].isnull().all():
        try:
            max_radial_vel_idx = df['radial_velocity_abs'].idxmax(); max_rv_point = df.loc[max_radial_vel_idx]
            ax.plot(max_rv_point['gps.lon'], max_rv_point['gps.lat'], '*', color='cyan', markersize=15, label=f'Max Radial Velocity: {max_rv_point["radial_velocity_abs"]:.1f} m/s', markeredgecolor='black')
        except ValueError: print("!!! WARNING: Could not plot max radial velocity point.")

    annotation_step = max(1, len(df) // 20)
    for i in range(0, len(df), annotation_step):
        point = df.iloc[i]
        if pd.notna(point['radial_velocity_abs']): ax.text(point['gps.lon'], point['gps.lat'], f"{point['radial_velocity_abs']:.1f}", fontsize=8, color='white', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.5))

    ax.set_xlabel('Longitude', fontsize=12); ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Mission Flight Path: {mission_name} (Color: Power, Size & Label: Abs. Radial Velocity)', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6); ax.set_aspect('equal', adjustable='box')
    if cmap is not None and vmin is not None and vmax is not None: cbar = fig.colorbar(sc, shrink=0.7, ax=ax); cbar.set_label('Average Channel Power (dB)')
    handles, labels = ax.get_legend_handles_labels()
    if radial_vel_range > 0.1 :
        size_legend_radial_vels = np.linspace(min_radial_vel, max_radial_vel, 4)
        legend_sizes = 15 + 250 * (((size_legend_radial_vels - min_radial_vel) / radial_vel_range)**2)
        size_handles = [plt.scatter([],[], s=s, edgecolors='k', color='grey') for s in legend_sizes]
        size_labels = [f'{vel:.1f} m/s' for vel in size_legend_radial_vels]
        legend2 = ax.legend(size_handles, size_labels, title="Absolute Radial Velocity", loc="lower left")
        ax.add_artist(legend2)
    ax.legend(handles=handles, labels=labels, loc='upper left')
    plt.tight_layout(); plt.show()

def analyze_and_plot_correlation(df, mission_name):
    print(f"\n" + "="*50); print(f"Starting correlation analysis for mission '{mission_name}'...")
    required_cols = ['power_db', 'groundSpeed', 'altitudeRelative', 'radial_velocity_abs']
    if not all(col in df.columns for col in required_cols) or df[required_cols].isnull().all().any(): print("!!! WARNING: Missing data for correlation analysis."); return
    df['power_change_rate'] = df['power_db'].diff().abs(); df['speed_change_rate'] = df['groundSpeed'].diff().abs()
    columns_for_corr = ['power_db', 'groundSpeed', 'altitudeRelative', 'radial_velocity_abs', 'power_change_rate', 'speed_change_rate']
    corr_df = df[columns_for_corr].dropna()
    if len(corr_df) < 2: print("!!! WARNING: Not enough data (<2 rows) to compute correlation matrix."); return
    corr_df.rename(columns={
        'power_db': 'Channel Power (dB)', 
        'groundSpeed': 'Ground Speed (m/s)', 
        'altitudeRelative': 'Altitude (m)', 
        'radial_velocity_abs': 'Abs. Radial Velocity (m/s)', 
        'power_change_rate': 'Power Fluctuation (dB)', 
        'speed_change_rate': 'Speed Change (m/s)'
    }, inplace=True)
    try:
        correlation_matrix = corr_df.corr(); print("\nCorrelation Matrix:"); print(correlation_matrix)
        plt.figure(figsize=(12, 10)); sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title(f'Mission Correlation Matrix (Full): {mission_name}', fontsize=16); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
        plt.tight_layout(); plt.show(); print("="*50)
    except Exception as corr_e:
        print(f"!!! Error calculating or plotting correlation matrix: {corr_e}")


# ==============================================================================
#  主程序执行部分
# ==============================================================================
if __name__ == '__main__':
    mission_name = "flight_mission_analysis"
    
    # =======================================================================
    #                       ↓↓↓ 使用您指定的路径 ↓↓↓
    # =======================================================================
    # 1. Channel Log: 使用您提供的绝对路径
    channel_log_path = os.path.join('/home/sixgnext/database/6gnext_messdaten/Measurement/EDAZ_0917', 'channel_estimates_20250917_150803_5ms_bolek_50m_5mps_10mps_withsuddenstop.txt')
    
    # 2. GPS Log: 使用 os.getcwd() 逻辑和您提供的文件名
    # 假设您从 /home/sixgnext/Nextcloud/Documents/Works/Github_Projects/channel_predictor_prepro 运行
    gps_log_path = os.path.join(os.getcwd(), 'data', '2025-09-17_14-11-56_bolek.csv')
    # =======================================================================

    try:
        channel_df_full = parse_channel_log(channel_log_path)
        gps_df_full = parse_gps_log(gps_log_path) # 使用“干净”的解析器

        if not gps_df_full.empty:
            if not all(col in gps_df_full.columns for col in ['gps.lat', 'gps.lon', 'altitudeAMSL']):
                 raise ValueError("Parsed GPS DataFrame is missing required coordinate columns.")
            gs_lat = gps_df_full['gps.lat'].iloc[0]; gs_lon = gps_df_full['gps.lon'].iloc[0]
            gs_alt_raw = gps_df_full['altitudeAMSL'].iloc[0]
            if pd.isna(gs_alt_raw):
                 raise ValueError("GPS Start Altitude (altitudeAMSL) is invalid (NaN).")
            gs_alt = gs_alt_raw + 5.0

            print("\n" + "="*50); print("Estimating Base Station coordinates from GPS start:")
            print(f"  - Latitude (Lat): {gs_lat}"); print(f"  - Longitude (Lon): {gs_lon}"); print(f"  - Altitude (Alt): {gs_alt} m"); print("="*50)
        else: raise ValueError("GPS data is empty after cleaning, cannot determine base station coordinates.")

        print("Synchronizing GPS and Channel data based on channel estimates timestamp...")
        print("\n" + "="*20 + " [ Final Pre-Merge Check ] " + "="*20)
        if not channel_df_full.empty: print(f"Channel Head:\n{channel_df_full.head(3).index}\nChannel Tail:\n{channel_df_full.tail(3).index}")
        else: print("Channel DataFrame is empty")
        if not gps_df_full.empty: print(f"GPS Head:\n{gps_df_full.head(3).index}\nGPS Tail:\n{gps_df_full.tail(3).index}")
        else: print("GPS DataFrame is empty")
        print("="*56 + "\n")
        if channel_df_full.empty or gps_df_full.empty: raise ValueError("Cannot merge: One or both input DataFrames are empty.")

        merged_df = pd.merge_asof(
            left=channel_df_full.sort_index(), right=gps_df_full.sort_index(),
            on='timestamp', direction='backward', tolerance=pd.Timedelta('1s')
        )
        if 'avg_power' not in merged_df.columns or 'gps.lat' not in merged_df.columns:
            raise ValueError("Merged DataFrame is missing 'avg_power' or 'gps.lat' columns.")
        merged_df.dropna(subset=['avg_power', 'gps.lat'], inplace=True)
        if merged_df.empty: raise ValueError("No matching data points found. Please re-check timestamps and timezones.")
        
        if 'timestamp' in merged_df.columns:
             merged_df.set_index('timestamp', inplace=True)
        elif not isinstance(merged_df.index, pd.DatetimeIndex):
             raise ValueError("Merge failed to produce a 'timestamp' column or index.")

        print(f"Data synchronization complete. Found {len(merged_df)} matching data points.")
        print("Calculating derived metrics...")
        merged_df['power_db'] = np.nan
        valid_power_mask = merged_df['avg_power'] > 0
        merged_df.loc[valid_power_mask, 'power_db'] = 10 * np.log10(merged_df.loc[valid_power_mask, 'avg_power'])

        required_velo_cols = ['gps.lon', 'gps.lat', 'altitudeAMSL', 'localPosition.vx', 'localPosition.vy', 'localPosition.vz']
        if not all(col in merged_df.columns for col in required_velo_cols):
             missing = [col for col in required_velo_cols if col not in merged_df.columns]
             raise ValueError(f"Merged DataFrame is missing required columns for velocity calculation: {missing}")

        merged_df = calculate_radial_velocity(merged_df, gs_lat, gs_lon, gs_alt)
        print("Metric calculation complete.")

        plot_full_profile(gps_df_full, channel_df_full, merged_df, mission_name)
        plot_flight_path_heatmap(merged_df, mission_name)
        analyze_and_plot_correlation(merged_df, mission_name)

    except (FileNotFoundError, ValueError) as e: print(f"\nERROR: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}"); traceback.print_exc()