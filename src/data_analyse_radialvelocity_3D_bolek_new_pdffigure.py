# -*- coding: utf-8 -*-
"""
DNN Channel Prediction Analysis Script (4-Axis Plot Version)
Status:
1. Plotting: ADDED 4th Axis for '3D Distance'. Now plots Power, Alt, Vel, Dist together.
2. Logic: Original NED->ENU calculation preserved.
3. Sync: +2h UTC offset maintained.
4. Robustness: Bad CSV lines skipped automatically.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
# Import Plotly
import plotly.graph_objects as go

# ==============================================================================
#  Function Definitions
# ==============================================================================

def parse_channel_log(file_path, time_offset_hours=0.0):
    """
    Parses SRS channel estimation file.
    Applies an optional timestamp offset in hours.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Channel file not found: {file_path}")
    print(f"Parsing channel file: {os.path.basename(file_path)}...")
    
    header_pattern = re.compile(r"SRS Frame .* Real: ([\d\.]+)")
    sc_pattern = re.compile(r"Sc \d+: Re = (-?\d+), Im = (-?\d+)")
    
    records = []
    current_timestamp = None
    current_powers = []
    
    # Optional time offset (hours), default no shift
    time_offset = pd.Timedelta(hours=float(time_offset_hours))
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            header_match = header_pattern.search(line)
            sc_match = sc_pattern.search(line)
            
            if header_match:
                if current_timestamp and current_powers:
                    records.append({'timestamp': current_timestamp, 'avg_power': np.mean(current_powers)})
                
                ts_unix_nano = float(header_match.group(1))
                ts_utc = pd.to_datetime(ts_unix_nano, unit='s', utc=True) + time_offset
                
                current_timestamp = ts_utc
                current_powers = []
            
            elif sc_match:
                re_val = int(sc_match.group(1))
                im_val = int(sc_match.group(2))
                power = re_val**2 + im_val**2
                current_powers.append(power)
        
        if current_timestamp and current_powers:
            records.append({'timestamp': current_timestamp, 'avg_power': np.mean(current_powers)})
    
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("Failed to parse any data from channel file.")
    
    # Convert to dB
    df['power_db'] = 10 * np.log10(df['avg_power'] + 1e-9)
    print(f"Channel data parsed, {len(df)} records found.")
    return df

def auto_align_channel_time(channel_df, gps_df, candidate_hours=range(-4, 5), tolerance='1s'):
    """
    Automatically find integer-hour offset for channel timestamps by maximizing
    timestamp match count against GPS data.
    """
    if channel_df.empty or gps_df.empty:
        return channel_df, 0, 0

    ch = channel_df.sort_values('timestamp').copy()
    gps = gps_df[['timestamp']].dropna().sort_values('timestamp').copy()
    gps['gps_marker'] = 1
    tol = pd.Timedelta(tolerance)

    best_hour = 0
    best_matches = -1

    for h in candidate_hours:
        shifted = ch[['timestamp']].copy()
        shifted['timestamp'] = shifted['timestamp'] + pd.Timedelta(hours=h)
        merged_probe = pd.merge_asof(
            left=shifted,
            right=gps,
            on='timestamp',
            direction='nearest',
            tolerance=tol
        )
        matches = int(merged_probe['gps_marker'].notna().sum())
        if matches > best_matches:
            best_matches = matches
            best_hour = h

    aligned = ch.copy()
    aligned['timestamp'] = aligned['timestamp'] + pd.Timedelta(hours=best_hour)
    return aligned, best_hour, best_matches

def estimate_ground_station_reference(gps_df, samples=30):
    """
    Estimate ground-station reference from the first valid GPS segment using median.
    """
    cols = ['gps.lat', 'gps.lon', 'altitudeAMSL']
    valid = gps_df.dropna(subset=cols).copy()
    if valid.empty:
        raise ValueError("No valid GPS rows to estimate ground-station reference.")

    anchor = valid.head(samples)
    gs_lat = float(anchor['gps.lat'].median())
    gs_lon = float(anchor['gps.lon'].median())
    gs_alt = float(anchor['altitudeAMSL'].median())
    return gs_lat, gs_lon, gs_alt

def parse_gps_log(file_path):
    """
    Parses CSV GPS log file with robustness fixes.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"GPS file not found: {file_path}")
    print(f"Parsing GPS file: {os.path.basename(file_path)}...")
    
    # Skip bad lines (Fixes 'Expected 161 fields...' error)
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
    except TypeError:
        df = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines=True)

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {'Timestamp': 'timestamp', 'Time': 'timestamp', 'time': 'timestamp'}
    df.rename(columns=rename_map, inplace=True)

    # Check for required columns
    required_cols = [
        'timestamp', 'gps.lat', 'gps.lon', 'altitudeAMSL', 'groundSpeed', 'altitudeRelative',
        'localPosition.vx', 'localPosition.vy', 'localPosition.vz'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Current columns (first 10): {list(df.columns)[:10]}")
        raise ValueError(f"GPS file missing required columns: {missing}")
    
    # Smart timestamp parsing
    if pd.api.types.is_numeric_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # Convert numeric columns
    num_cols = ['gps.lat', 'gps.lon', 'altitudeAMSL', 'groundSpeed', 'altitudeRelative', 
                'localPosition.vx', 'localPosition.vy', 'localPosition.vz']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    print(f"GPS data parsed, {len(df)} records found.")
    return df

def calculate_derived_metrics(df, gs_lat, gs_lon, gs_alt):
    """
    Calculates derived metrics using the ORIGINAL method (NED->ENU).
    """
    print("Calculating derived metrics (Distance & Radial Velocity)...")
    
    # 1. Position Vector P (ENU Frame: East, North, Up)
    df['pos_x_m'] = (df['gps.lon'] - gs_lon) * 40075000 * np.cos(np.radians(gs_lat)) / 360 # East
    df['pos_y_m'] = (df['gps.lat'] - gs_lat) * 40008000 / 360 # North
    df['pos_z_m'] = df['altitudeAMSL'] - gs_alt # Up
    
    pos_vectors = df[['pos_x_m', 'pos_y_m', 'pos_z_m']].values
    
    # 2. Velocity Vector Transformation (NED -> ENU)
    # Original logic: vx=North, vy=East, vz=Down
    vx_ned = df['localPosition.vx']
    vy_ned = df['localPosition.vy']
    vz_ned = df['localPosition.vz']
    
    # Build ENU velocity vectors [East, North, Up]
    # Note: Logic preserved from your original script
    vel_vectors_enu = pd.concat([vy_ned, vx_ned, -vz_ned], axis=1, ignore_index=True).values
    
    # 3. 3D Distance
    df['distance_to_gs_3d'] = np.linalg.norm(pos_vectors, axis=1)

    # 4. Radial Velocity (Dot Product)
    dot_product = np.sum(vel_vectors_enu * pos_vectors, axis=1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        radial_velocity = np.divide(dot_product, df['distance_to_gs_3d'])
        radial_velocity[df['distance_to_gs_3d'] == 0] = 0.0
        
    df['radial_velocity'] = radial_velocity
    df['radial_velocity_abs'] = np.abs(radial_velocity)
    
    return df

def plot_full_profile(gps_df, channel_df, merged_df, mission_name, save_path=None):
    """
    Draws the combined time profile plot.
    Added Axis 4 for '3D Distance'.
    """
    print(f"\nPlotting combined time profile for '{mission_name}'...")
    
    # Increase figure width to accommodate extra y-axis
    fig, ax1 = plt.subplots(figsize=(18, 9))
    
    # --- Axis 1 (Left): Channel Power ---
    color_pwr = 'tab:red'
    ax1.set_xlabel('Time (Berlin)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Channel Power (dB)', color=color_pwr, fontsize=14, fontweight='bold')
    l1, = ax1.plot(merged_df.index, merged_df['power_db'], color=color_pwr, linewidth=1.5, label='Channel Power (dB)')
    ax1.tick_params(axis='y', labelcolor=color_pwr)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # --- Axis 2 (Right): Altitude ---
    ax2 = ax1.twinx()
    color_alt = 'tab:blue'
    ax2.set_ylabel('Relative Altitude (m)', color=color_alt, fontsize=14, fontweight='bold')
    l2, = ax2.plot(merged_df.index, merged_df['altitudeRelative'], color=color_alt, linewidth=2, linestyle='-', label='Rel. Altitude')
    ax2.tick_params(axis='y', labelcolor=color_alt)
    
    # --- Axis 3 (Right, Offset 1): Speed ---
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.08)) # Offset 8%
    color_spd = 'tab:green'
    ax3.set_ylabel('Velocity (m/s)', color=color_spd, fontsize=14, fontweight='bold')
    l3, = ax3.plot(merged_df.index, merged_df['radial_velocity'], color=color_spd, linewidth=2.5, linestyle='-', label='Radial Velocity')
    color_gspd = 'tab:orange'
    l4, = ax3.plot(merged_df.index, merged_df['groundSpeed'], color=color_gspd, linewidth=1.5, linestyle='--', label='Ground Speed')
    ax3.tick_params(axis='y', labelcolor=color_spd)
    
    # --- Axis 4 (Right, Offset 2): 3D Distance [新增] ---
    ax4 = ax1.twinx()
    ax4.spines["right"].set_position(("axes", 1.16)) # Offset 16% to avoid overlap
    color_dist = 'tab:brown'
    ax4.set_ylabel('3D Distance (m)', color=color_dist, fontsize=14, fontweight='bold')
    l5, = ax4.plot(merged_df.index, merged_df['distance_to_gs_3d'], color=color_dist, linewidth=1.5, linestyle='-.', label='3D Distance')
    ax4.tick_params(axis='y', labelcolor=color_dist)

    # Combined Legend
    lines = [l1, l2, l3, l4, l5]
    labels = [l.get_label() for l in lines]
    # Place legend outside to save space or upper left
    ax1.legend(lines, labels, loc='upper left', fontsize=11, frameon=True, framealpha=0.9, ncol=2)
    
    plt.title(f'Integrated Flight Profile: {mission_name}', fontsize=20, fontweight='bold', pad=20)
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()
    
    # Adjust layout to make room for the extra spine
    plt.tight_layout() 
    # Extra adjustment for right margin
    plt.subplots_adjust(right=0.85)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f" -> Time profile saved to: {save_path}")

def analyze_and_plot_correlation(df, mission_name, save_path=None):
    """
    Plots correlation matrix (PDF).
    """
    print(f"\n" + "="*50)
    print(f"Calculating correlation for '{mission_name}'...")
    
    columns_for_corr = [
        'power_db', 'groundSpeed', 'altitudeRelative', 'radial_velocity_abs', 'distance_to_gs_3d'
    ]
    valid_cols = [c for c in columns_for_corr if c in df.columns]
    corr_df = df[valid_cols].dropna()

    rename_map = {
        'power_db': 'Channel Power',
        'groundSpeed': 'Ground Speed',
        'altitudeRelative': 'Altitude',
        'radial_velocity_abs': 'Abs Radial Vel',
        'distance_to_gs_3d': '3D Distance'
    }
    corr_df.rename(columns=rename_map, inplace=True)

    correlation_matrix = corr_df.corr()

    print("\nCorrelation Matrix:")
    print(correlation_matrix)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
                linewidths=.5, annot_kws={"size": 16}, cbar_kws={"shrink": .8})
    
    plt.title(f'Correlation Matrix: {mission_name}', fontsize=20, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f" -> Correlation matrix saved to: {save_path}")
    print("="*50)

def plot_3d_flight_path_interactive(df, mission_name, save_path_html=None, save_path_pdf=None):
    """
    Plots 3D flight path with scaled arrows (PDF & HTML).
    """
    print(f"\n--- Generating 3D Flight Path for '{mission_name}' ---")
    
    x = df['pos_x_m']
    y = df['pos_y_m']
    z = df['pos_z_m']
    power_db_filtered = df['power_db'].dropna()
    vmin, vmax = power_db_filtered.quantile(0.05), power_db_filtered.quantile(0.95)
    
    # 1. Path Markers
    trace_path = go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=4, color=df['power_db'], colorscale='Jet', cmin=vmin, cmax=vmax,
                    colorbar=dict(title='Power (dB)'), showscale=True),
        text=[f"RelAlt: {a:.1f}m<br>Dist: {d:.1f}m<br>RadVel: {rv:.1f}m/s"
              for a, d, rv in zip(df['altitudeRelative'], df['distance_to_gs_3d'], df['radial_velocity'])],
        hoverinfo='text', name='Flight Path'
    )
    
    # 2. Arrows (Radial Velocity)
    # Dynamic scaling
    scene_range = max(x.max()-x.min(), y.max()-y.min(), z.max()-z.min())
    avg_vel = df['radial_velocity_abs'].mean()
    if avg_vel < 0.1: avg_vel = 1.0 
    
    arrow_scale = (scene_range * 0.08) / avg_vel 
    print(f"DEBUG: 3D Arrow Scale: {arrow_scale:.2f}")

    step = 20 
    quiver_df = df.iloc[::step].copy()
    
    lines_x, lines_y, lines_z = [], [], []
    cone_x, cone_y, cone_z = [], [], [] 
    cone_u, cone_v, cone_w = [], [], []
    
    for i, row in quiver_df.iterrows():
        sx, sy, sz = row['pos_x_m'], row['pos_y_m'], row['pos_z_m']
        dist = row['distance_to_gs_3d']
        if dist < 1: continue
        
        ux, uy, uz = sx/dist, sy/dist, sz/dist
        v_rad = row['radial_velocity']
        
        ex = sx + ux * v_rad * arrow_scale
        ey = sy + uy * v_rad * arrow_scale
        ez = sz + uz * v_rad * arrow_scale
        
        lines_x.extend([sx, ex, None])
        lines_y.extend([sy, ey, None])
        lines_z.extend([sz, ez, None])
        
        cone_x.append(ex)
        cone_y.append(ey)
        cone_z.append(ez)
        cone_u.append(ux * v_rad)
        cone_v.append(uy * v_rad)
        cone_w.append(uz * v_rad)

    trace_lines = go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z,
        mode='lines',
        line=dict(color='black', width=3),
        name='Radial Vel Vector'
    )
    
    trace_cones = go.Cone(
        x=cone_x, y=cone_y, z=cone_z,
        u=cone_u, v=cone_v, w=cone_w,
        sizemode="absolute",
        sizeref=2, 
        anchor="tip",
        colorscale=[[0, 'black'], [1, 'black']],
        showscale=False,
        name='Direction'
    )
    
    trace_start = go.Scatter3d(x=[x.iloc[0]], y=[y.iloc[0]], z=[z.iloc[0]], mode='markers', marker=dict(color='green', size=8, symbol='diamond'), name='Start')
    trace_end = go.Scatter3d(x=[x.iloc[-1]], y=[y.iloc[-1]], z=[z.iloc[-1]], mode='markers', marker=dict(color='red', size=8, symbol='x'), name='End')

    layout = go.Layout(
        title=f'3D Flight Path: {mission_name}',
        scene=dict(xaxis_title='East (m)', yaxis_title='North (m)', zaxis_title='Up wrt GS (m)', aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig = go.Figure(data=[trace_path, trace_lines, trace_cones, trace_start, trace_end], layout=layout)
    
    if save_path_html:
        fig.write_html(save_path_html)
        print(f" -> 3D Path (HTML) saved to: {save_path_html}")
    
    if save_path_pdf:
        try:
            fig.write_image(save_path_pdf, format="pdf")
            print(f" -> 3D Path (PDF) saved to: {save_path_pdf}")
        except Exception as e:
            print(f"!!! PDF Save Failed: {e}")

# ==============================================================================
#  Main Execution
# ==============================================================================

if __name__ == '__main__':
    mission_name = "40m_5mps"
    CHANNEL_TIME_OFFSET_HOURS = 2.0  # Berlin local vs GPS UTC
    USE_AUTO_ALIGN = False            # Keep False for production runs; True only for diagnostics
    
    base_dir = os.getcwd()
    channel_filename = 'channel_estimates_20250917_140636_bolek_40m_5mps.txt'
    gps_filename = '2025-09-17_14-11-56_bolek.csv'
    
    channel_log_path = os.path.join(base_dir, 'data', channel_filename)
    gps_log_path = os.path.join(base_dir, 'data', gps_filename)

    SAVE_PLOTS = True
    output_dir = os.path.join(base_dir, 'paper_results')
    if SAVE_PLOTS and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Work Dir: {base_dir}")
    print(f"Channel: {channel_log_path}")
    print(f"GPS: {gps_log_path}")

    try:
        # 1. Parse
        channel_df_full = parse_channel_log(channel_log_path, time_offset_hours=CHANNEL_TIME_OFFSET_HOURS)
        gps_df_full = parse_gps_log(gps_log_path)
        
        # 2. Merge
        channel_df_full.sort_values('timestamp', inplace=True)
        gps_df_full.sort_values('timestamp', inplace=True)

        if USE_AUTO_ALIGN:
            # Optional diagnostic mode only
            channel_df_full, best_offset_h, matched_probe = auto_align_channel_time(
                channel_df_full,
                gps_df_full,
                candidate_hours=range(-4, 5),
                tolerance='1s'
            )
            print(f"\n[DEBUG] Auto time alignment enabled:")
            print(f"Best channel offset: {best_offset_h:+d} h, probe matches: {matched_probe}")
        else:
            print(f"\n[DEBUG] Fixed channel offset mode: {CHANNEL_TIME_OFFSET_HOURS:+.1f} h")

        # DEBUG Time Ranges (after auto alignment)
        t_ch_min, t_ch_max = channel_df_full['timestamp'].min(), channel_df_full['timestamp'].max()
        t_gps_min, t_gps_max = gps_df_full['timestamp'].min(), gps_df_full['timestamp'].max()
        print(f"Channel: {t_ch_min} -> {t_ch_max}")
        print(f"GPS    : {t_gps_min} -> {t_gps_max}")
        
        print("\nMerging data...")
        merged_df = pd.merge_asof(
            left=channel_df_full, 
            right=gps_df_full,
            on='timestamp', 
            direction='backward',
            tolerance=pd.Timedelta('1s')
        )
        
        merged_df.dropna(subset=['avg_power', 'gps.lat'], inplace=True)
        
        if merged_df.empty:
            raise ValueError("Merge result is empty! Check timestamps.")
            
        merged_df.set_index('timestamp', inplace=True)
        print(f"Merge complete. Samples: {len(merged_df)}")
        
        # 3. Calculate Metrics (Original Method)
        gs_lat, gs_lon, gs_alt = estimate_ground_station_reference(gps_df_full, samples=30)
        print(f"Estimated GS reference (median first 30 samples): lat={gs_lat:.7f}, lon={gs_lon:.7f}, alt={gs_alt:.2f}m")
        
        merged_df = calculate_derived_metrics(merged_df, gs_lat, gs_lon, gs_alt)

        # 4. Plotting (PDF)
        save_path_profile = os.path.join(output_dir, f"{mission_name}_time_profile_combined.pdf") if SAVE_PLOTS else None
        save_path_corr = os.path.join(output_dir, f"{mission_name}_correlation.pdf") if SAVE_PLOTS else None
        save_path_3d_html = os.path.join(output_dir, f"{mission_name}_3d_path.html") if SAVE_PLOTS else None
        save_path_3d_pdf = os.path.join(output_dir, f"{mission_name}_3d_path.pdf") if SAVE_PLOTS else None

        plot_full_profile(gps_df_full, channel_df_full, merged_df, mission_name, save_path=save_path_profile)
        
        plot_3d_flight_path_interactive(merged_df, mission_name, save_path_html=save_path_3d_html, save_path_pdf=save_path_3d_pdf)
        
        analyze_and_plot_correlation(merged_df, mission_name, save_path=save_path_corr)
        
        print("\nAll done! Results saved to 'paper_results' folder.")

    except Exception as e:
        print(f"\nError: {e}")
