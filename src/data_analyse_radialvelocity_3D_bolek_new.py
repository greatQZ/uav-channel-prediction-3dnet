import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
# 新增：导入Plotly库
import plotly.graph_objects as go

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
    # 增加 on_bad_lines='skip' 来处理可能存在的格式错误的行
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
    except Exception as e:
        print(f"!!! 读取CSV时发生错误: {e}")
        print("!!! 尝试使用'python'引擎读取，速度可能较慢...")
        df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')

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
        print(f" -> 已清理 {original_rows - len(df)} 行无效或格式错误的GPS记录。")

    df['timestamp'] = pd.to_datetime(df['Timestamp'])
    df['timestamp'] = df['timestamp'].dt.tz_localize('Europe/Berlin', ambiguous='infer')
    df = df.set_index('timestamp').drop(columns=['Timestamp'])
    print(f" -> 完成，共找到 {len(df)} 条有效记录。")
    return df

def calculate_derived_metrics(df, gs_lat, gs_lon, gs_alt):
    """
    计算无人机相对于地面基站的3D距离和相对径向速度。
    """
    print("正在计算衍生指标 (距离和径向速度)...")
    
    # 位置矢量 P (ENU坐标系: East, North, Up)
    df['pos_x_m'] = (df['gps.lon'] - gs_lon) * 40075000 * np.cos(np.radians(gs_lat)) / 360
    df['pos_y_m'] = (df['gps.lat'] - gs_lat) * 40008000 / 360
    df['pos_z_m'] = df['altitudeAMSL'] - gs_alt
    pos_vectors = df[['pos_x_m', 'pos_y_m', 'pos_z_m']].values
    
    # --- 核心修正：将速度矢量从NED转换为ENU ---
    # 原始速度矢量 V (NED坐标系: North, East, Down)
    vx_ned = df['localPosition.vx']
    vy_ned = df['localPosition.vy']
    vz_ned = df['localPosition.vz']
    # 转换后的速度矢量 V' (ENU坐标系: East, North, Up)
    vel_vectors_enu = pd.concat([vy_ned, vx_ned, -vz_ned], axis=1, ignore_index=True).values
    # --- 修正结束 ---
    
    # 3D距离
    df['distance_to_gs_3d'] = np.linalg.norm(pos_vectors, axis=1)

    # 径向速度 (使用正确的坐标系进行点积)
    dot_product = np.sum(vel_vectors_enu * pos_vectors, axis=1)
    radial_velocity_with_direction = np.divide(dot_product, df['distance_to_gs_3d'], out=np.zeros_like(dot_product), where=df['distance_to_gs_3d']!=0)
    df['radial_velocity'] = radial_velocity_with_direction
    df['radial_velocity_abs'] = np.abs(radial_velocity_with_direction)
    
    # 功率dB值
    df['power_db'] = 10 * np.log10(df['avg_power'])
    
    print(" -> 衍生指标计算完成。")
    return df

def plot_full_profile(gps_df_full, channel_df_full, merged_df, mission_name, save_path=None):
    """
    为单次飞行任务绘制包含所有关键参数的最终时间剖面图。
    【已应用最终修正】在绘图前强行将柏林时间转换为无时区的朴素时间。
    """
    print(f"\n--- 正在为任务 '{mission_name}' 生成最终时间剖面图 ---")
    fig, ax1 = plt.subplots(figsize=(18, 10))

    # =======================================================================
    #  *** 代码修改处：应用最终的时间戳固化方案 ***
    # =======================================================================
    plot_merged_df = merged_df.copy()
    plot_merged_df.index = pd.to_datetime(plot_merged_df.index.strftime('%Y-%m-%d %H:%M:%S.%f'))

    plot_channel_df = channel_df_full.copy()
    plot_channel_df.index = pd.to_datetime(plot_channel_df.index.strftime('%Y-%m-%d %H:%M:%S.%f'))
    # =======================================================================

    color1 = 'darkviolet'
    ax1.set_xlabel('Time (Berlin Local)', fontsize=14)
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
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" -> 时间剖面图已保存至: {save_path}")
        
    plt.show()

def plot_3d_flight_path_interactive(df, mission_name, save_path=None):
    """
    【最终版本】使用Plotly在3D空间中绘制飞行轨迹。
    特征：颜色代表功率，只保留径向速度矢量箭头。图例位置已调整。
    """
    print(f"\n--- 正在为任务 '{mission_name}' 生成可交互的3D飞行轨迹图 ---")
    
    # 准备位置数据
    x = df['pos_x_m']
    y = df['pos_y_m']
    z = df['altitudeRelative']
    
    # 颜色：信道功率
    power_db_filtered = df['power_db'].dropna()
    vmin = power_db_filtered.quantile(0.05)
    vmax = power_db_filtered.quantile(0.95)
    
    # 创建3D散点图轨迹 (大小固定)
    trace_path = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=df['power_db'],
            colorscale='Inferno',
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(title='Channel Power (dB)'),
            showscale=True
        ),
        text=[f"Speed: {gs:.1f} m/s<br>Radial Vel: {rv:.1f} m/s" for gs, rv in zip(df['groundSpeed'], df['radial_velocity_abs'])],
        hoverinfo='text',
        name='Flight Path'
    )

    # 只创建径向速度矢量箭头
    step = 10
    quiver_df = df.iloc[::step]
    arrow_scale = 1.5 
    
    radial_lines_x, radial_lines_y, radial_lines_z = [], [], []
    
    for i, row in quiver_df.iterrows():
        start_point = np.array([row['pos_x_m'], row['pos_y_m'], row['altitudeRelative']])
        pos_vector = start_point
        pos_dir = pos_vector / (np.linalg.norm(pos_vector) + 1e-9)
        radial_length = row['radial_velocity'] * arrow_scale
        radial_vel_end = start_point + pos_dir * radial_length
        radial_lines_x.extend([start_point[0], radial_vel_end[0], None])
        radial_lines_y.extend([start_point[1], radial_vel_end[1], None])
        radial_lines_z.extend([start_point[2], radial_vel_end[2], None])

    trace_radial_arrows = go.Scatter3d(
        x=radial_lines_x, y=radial_lines_y, z=radial_lines_z,
        mode='lines', line=dict(color='magenta', width=5),
        hoverinfo='none', name='Radial Velocity Vector'
    )

    # 创建起点和终点标记
    trace_start = go.Scatter3d(x=[x.iloc[0]], y=[y.iloc[0]], z=[z.iloc[0]], mode='markers', marker=dict(color='lime', size=10, symbol='circle'), name='Start')
    trace_end = go.Scatter3d(x=[x.iloc[-1]], y=[y.iloc[-1]], z=[z.iloc[-1]], mode='markers', marker=dict(color='red', size=10, symbol='x'), name='End')

    layout = go.Layout(
        title=f'3D Flight Path: Power (Color), Radial Velocity Vector (Arrow)',
        scene=dict(
            xaxis_title='East-West Distance from Start (m)',
            yaxis_title='North-South Distance from Start (m)',
            zaxis_title='Relative Altitude (m)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=True,
        
        # =======================================================================
        #  *** 代码修改处：新增 legend 参数来控制图例位置 ***
        # =======================================================================
        legend=dict(
            yanchor="top",     # y 锚点在图例的顶部
            y=0.99,            # 图例的顶部位于绘图区域从上往下 1% 的位置 (1=最顶, 0=最底)
            xanchor="left",    # x 锚点在图例的左侧
            x=0.01,            # 图例的左侧位于绘图区域从左往右 1% 的位置 (0=最左, 1=最右)
            bgcolor="rgba(255, 255, 255, 0.6)", # 背景设为半透明白色
            bordercolor="Black", # 边框为黑色
            borderwidth=1        # 边框宽度
        )
        # =======================================================================
    )

    fig = go.Figure(data=[trace_path, trace_start, trace_end, trace_radial_arrows], layout=layout)
    
    if save_path:
        fig.write_html(save_path)
        print(f" -> 可交互的3D轨迹图已保存至: {save_path}")
        
    fig.show()

def analyze_and_plot_correlation(df, mission_name, save_path=None):
    """计算并可视化关键参数之间的相关性。"""
    print(f"\n" + "="*50)
    print(f"开始为任务 '{mission_name}' (整个飞行过程) 进行关键参数相关性分析...")
    
    columns_for_corr = [
        'power_db', 'groundSpeed', 'altitudeRelative', 'radial_velocity_abs', 'distance_to_gs_3d'
    ]
    corr_df = df[columns_for_corr].dropna()

    corr_df.rename(columns={
        'power_db': 'Channel Power (dB)',
        'groundSpeed': 'Groundspeed (m/s)',
        'altitudeRelative': 'Altitude (m)',
        'radial_velocity_abs': 'Abs. Radial Velocity (m/s)',
        'distance_to_gs_3d': 'Distance to GS (3D, m)'
    }, inplace=True)

    correlation_matrix = corr_df.corr()

    print("\n参数相关性矩阵:")
    print(correlation_matrix)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(f'Correlation Matrix for Mission (Entire Flight): {mission_name}', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" -> 相关性热力图已保存至: {save_path}")

    plt.show()
    print("="*50)

# ==============================================================================
#  主程序执行部分
# ==============================================================================
if __name__ == '__main__':
    # --- 1. 用户配置区域 ---
    mission_name = "40m_5mps" # 使用一个适合做文件名的名称
    channel_log_path = os.path.join(os.getcwd(), 'data', 'channel_estimates_20250917_140636_bolek_40m_5mps.txt')
    gps_log_path = os.path.join(os.getcwd(), 'data', '2025-09-17_14-11-56_bolek.csv')

    # --- 新增：保存图表配置 ---
    SAVE_PLOTS = True # 设置为 True 来保存图表，False 则不保存
    output_dir = os.path.join(os.getcwd(), 'results') # 图片保存的文件夹
    if SAVE_PLOTS and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # --- 配置结束 ---

    try:
        # --- 2. 解析数据 ---
        channel_df_full = parse_channel_log(channel_log_path)
        gps_df_full = parse_gps_log(gps_log_path)

        # --- 3. 确定基站坐标 ---
        if not gps_df_full.empty:
            gs_lat = gps_df_full['gps.lat'].iloc[0]
            gs_lon = gps_df_full['gps.lon'].iloc[0]
            gs_alt = gps_df_full['altitudeAMSL'].iloc[0] + 5.0
            print("\n" + "="*50)
            print("根据GPS起点估算基站坐标:")
            print(f"  - 纬度 (Lat): {gs_lat}")
            print(f"  - 经度 (Lon): {gs_lon}")
            print(f"  - 海拔 (Alt): {gs_alt} m")
            print("="*50)
        else:
            raise ValueError("GPS数据为空，无法确定基站坐标。")

        # =======================================================================
        #  *** 代码修改处：应用新的同步逻辑 ***
        # =======================================================================
        # --- 4. 同步整个飞行过程的数据 ---
        print("正在以channel estimates的时间为基准，同步GPS和信道数据...")
        merged_df = pd.merge_asof(
            left=channel_df_full.sort_index(),      # 以 channel_df 作为同步的左表（基准）
            right=gps_df_full.sort_index(),         # 将 gps_df 匹配到 channel_df
            on='timestamp',
            direction='nearest',                    # 寻找最近的时间点
            tolerance=pd.Timedelta('1s')          # 允许的最大时间差为1秒
        )
        # 确保合并后的数据同时具有有效的功率和GPS位置信息
        merged_df.dropna(subset=['avg_power', 'gps.lat'], inplace=True)
        
        if merged_df.empty:
             raise ValueError("未能匹配任何数据点。请检查GPS和信道日志的时间戳是否对齐。")
        
        merged_df.set_index('timestamp', inplace=True)

        print(f"数据同步完成，共得到 {len(merged_df)} 个匹配的数据点。")
        # =======================================================================

        # --- 5. 计算衍生指标 ---
        # 注意：函数名已更新，确保与新函数定义匹配
        merged_df = calculate_derived_metrics(merged_df, gs_lat, gs_lon, gs_alt)

        # --- 6. 生成图表和分析 ---
        # 构造保存路径
        save_path_profile = os.path.join(output_dir, f"{mission_name}_time_profile.png") if SAVE_PLOTS else None
        save_path_3d_html = os.path.join(output_dir, f"{mission_name}_3d_path.html") if SAVE_PLOTS else None # 保存为HTML
        save_path_corr = os.path.join(output_dir, f"{mission_name}_correlation.png") if SAVE_PLOTS else None

        plot_full_profile(gps_df_full, channel_df_full, merged_df, mission_name, save_path=save_path_profile)
        plot_3d_flight_path_interactive(merged_df, mission_name, save_path=save_path_3d_html) # 调用新的交互式3D绘图函数
        analyze_and_plot_correlation(merged_df, mission_name, save_path=save_path_corr)

    except (FileNotFoundError, ValueError) as e:
        print(f"\n错误: {e}")
    except Exception as e:
        print(f"\n发生未知错误: {e}")