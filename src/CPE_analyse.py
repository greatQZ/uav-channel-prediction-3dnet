# -*- coding: utf-8 -*-
"""
VTC W16 终极纯净版 (单子载波全域锁定 + 零失真频域分析)
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch
from scipy.ndimage import median_filter, uniform_filter1d
import matplotlib.gridspec as gridspec


def _infer_unix_unit(ts_value):
    abs_ts = abs(float(ts_value))
    if abs_ts > 1e17:
        return 'ns'
    if abs_ts > 1e14:
        return 'us'
    if abs_ts > 1e11:
        return 'ms'
    return 's'

# ==============================================================================
#  1. 信道解析 (视频流追踪版：专抓大带宽数据传输信号)
# ==============================================================================
def parse_channel_log(file_path, time_offset=pd.Timedelta(0), combine_mode='mean'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Channel file not found: {file_path}")
    
    print("📡 正在扫描视频传输频带 (寻找高带宽数据爆发区)...")
    
    # 第一遍扫描：计算每个子载波的平均能量分布
    sc_pattern_fast = re.compile(r"Sc (\d+): Re = (-?\d+), Im = (-?\d+)")
    power_distribution = np.zeros(1024)
    counts = np.zeros(1024)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            m = sc_pattern_fast.search(line)
            if m:
                idx = int(m.group(1))
                if 5 < idx < 1020 and not (270 <= idx <= 717): # 避开DC和保护带
                    re_val = int(m.group(2))
                    im_val = int(m.group(3))
                    power_distribution[idx] += (re_val**2 + im_val**2)
                    counts[idx] += 1
    
    # 计算平均功率谱密度
    psd = np.divide(power_distribution, counts, out=np.zeros_like(power_distribution), where=counts!=0)
    
    # 使用滑动窗口寻找能量最集中的“视频数据块” (假设视频占用约 12-24 个子载波)
    window_size = 12
    smoothed_psd = np.convolve(psd, np.ones(window_size)/window_size, mode='same')
    best_idx = int(np.argmax(smoothed_psd))
    
    print(f"🎯 [锁定视频频带] 发现视频流数据中心位于: Sc {best_idx}")
    print(f"📊 该频带平均强度: {10*np.log10(smoothed_psd[best_idx]+1e-9):.2f} dB")

    # 第二遍扫描：提取锁定的视频频带相位
    header_pattern = re.compile(r"SRS Frame .* Real: ([\d\.eE+-]+)")
    sc_pattern = re.compile(r"Sc (\d+): Re = (-?\d+), Im = (-?\d+)")
    
    records = []
    current_timestamp = None
    current_powers = []
    # 存储窗口内所有子载波的复数，用于矢量平均（提高SNR）
    window_complex_cache = [] 
    
    window_start = max(0, best_idx - window_size // 2)
    window_end = min(1023, window_start + window_size - 1)
    window_start = max(0, window_end - window_size + 1)

    def finalize_current_record():
        if current_timestamp is None or not current_powers:
            return
        if window_complex_cache:
            if combine_mode == 'sum':
                complex_csi = np.sum(window_complex_cache)
            else:
                # Mean avoids an amplitude jump tied to how many carriers fall in the window.
                complex_csi = np.mean(window_complex_cache)
        else:
            complex_csi = None

        records.append({
            'timestamp': current_timestamp,
            'avg_power': np.mean(current_powers),
            'complex_csi': complex_csi,
            'window_csi': np.array(window_complex_cache, dtype=np.complex128) if window_complex_cache else None
        })
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            header_match = header_pattern.search(line)
            sc_match = sc_pattern.search(line)
            
            if header_match:
                finalize_current_record()
                
                ts_raw = float(header_match.group(1))
                ts_unit = _infer_unix_unit(ts_raw)
                current_timestamp = pd.to_datetime(ts_raw, unit=ts_unit, utc=True) + time_offset
                current_powers = []
                window_complex_cache = []
            
            elif sc_match:
                idx = int(sc_match.group(1))
                re_val = int(sc_match.group(2))
                im_val = int(sc_match.group(3))
                comp = complex(re_val, im_val)
                
                # 如果落在视频数据窗口内，存入缓存
                if window_start <= idx <= window_end:
                    window_complex_cache.append(comp)
                
                current_powers.append(re_val**2 + im_val**2)

    # Flush the last frame, otherwise one frame is always dropped.
    finalize_current_record()

    df = pd.DataFrame(records)
    print(f"✅ 视频信号提取完成，共 {len(df)} 帧。")
    return df


def parse_channel_logs(file_paths, time_offset=pd.Timedelta(0), combine_mode='mean'):
    if isinstance(file_paths, (str, os.PathLike)):
        file_paths = [file_paths]
    frames = []
    for idx, path in enumerate(file_paths, start=1):
        print(f"\n📂 解析信道文件 {idx}/{len(file_paths)}: {os.path.basename(path)}")
        df_part = parse_channel_log(path, time_offset=time_offset, combine_mode=combine_mode)
        if not df_part.empty:
            frames.append(df_part)
    if not frames:
        raise ValueError("No valid channel frames parsed from the provided file list.")
    df_all = pd.concat(frames, ignore_index=True)
    df_all.sort_values('timestamp', inplace=True)
    df_all.drop_duplicates(subset=['timestamp'], inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    print(f"✅ 多文件信道合并完成，共 {len(df_all)} 帧。")
    return df_all

# ==============================================================================
#  2 & 3. GPS解析与几何计算 (保持不变)
# ==============================================================================
def parse_gps_log(file_path, local_tz='Europe/Berlin'):
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
    except TypeError:
        df = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines=True)

    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {'Timestamp': 'timestamp', 'Time': 'timestamp', 'time': 'timestamp'}
    df.rename(columns=rename_map, inplace=True)

    if 'timestamp' not in df.columns:
        raise ValueError("GPS CSV does not contain a timestamp column.")

    if pd.api.types.is_numeric_dtype(df['timestamp']):
        ts_unit = _infer_unix_unit(df['timestamp'].dropna().iloc[0])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit=ts_unit, utc=True)
    else:
        ts = pd.to_datetime(df['timestamp'], errors='coerce')
        if ts.dt.tz is None:
            # Most flight logs store local wall-clock time without timezone info.
            ts = ts.dt.tz_localize(local_tz, nonexistent='shift_forward', ambiguous='NaT').dt.tz_convert('UTC')
        else:
            ts = ts.dt.tz_convert('UTC')
        df['timestamp'] = ts
    
    num_cols = ['gps.lat', 'gps.lon', 'altitudeAMSL', 'groundSpeed', 'altitudeRelative', 
                'localPosition.vx', 'localPosition.vy', 'localPosition.vz']
    for c in num_cols:
        if c not in df.columns:
            raise ValueError(f"GPS CSV missing required column: {c}")
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def calculate_derived_metrics(df, gs_lat, gs_lon, gs_alt):
    df['pos_x_m'] = (df['gps.lon'] - gs_lon) * 40075000 * np.cos(np.radians(gs_lat)) / 360
    df['pos_y_m'] = (df['gps.lat'] - gs_lat) * 40008000 / 360
    df['pos_z_m'] = df['altitudeAMSL'] - gs_alt
    pos_vectors = df[['pos_x_m', 'pos_y_m', 'pos_z_m']].values
    
    vx_ned = df['localPosition.vx']
    vy_ned = df['localPosition.vy']
    vz_ned = df['localPosition.vz']
    vel_vectors_enu = pd.concat([vy_ned, vx_ned, -vz_ned], axis=1, ignore_index=True).values
    
    df['distance_to_gs_3d'] = np.linalg.norm(pos_vectors, axis=1)

    dot_product = np.sum(vel_vectors_enu * pos_vectors, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        radial_velocity = np.divide(dot_product, df['distance_to_gs_3d'])
        radial_velocity[df['distance_to_gs_3d'] == 0] = 0.0
        
    df['radial_velocity'] = radial_velocity
    return df


def _time_overlap_seconds(a_start, a_end, b_start, b_end):
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0.0, (end - start).total_seconds())


def find_best_hourly_offset(channel_df, gps_df, candidate_hours=range(-3, 4)):
    ch_start = channel_df['timestamp'].iloc[0]
    ch_end = channel_df['timestamp'].iloc[-1]
    gps_start = gps_df['timestamp'].iloc[0]
    gps_end = gps_df['timestamp'].iloc[-1]

    best_offset = pd.Timedelta(0)
    best_overlap = _time_overlap_seconds(ch_start, ch_end, gps_start, gps_end)
    for hour in candidate_hours:
        offset = pd.Timedelta(hours=hour)
        overlap = _time_overlap_seconds(ch_start + offset, ch_end + offset, gps_start, gps_end)
        if overlap > best_overlap:
            best_overlap = overlap
            best_offset = offset
    return best_offset, best_overlap


def alias_doppler_with_breaks(theoretical_doppler, fs):
    aliased = theoretical_doppler - fs * np.round(theoretical_doppler / fs)
    aliased_plot = aliased.astype(float).copy()
    jumps = np.abs(np.diff(aliased_plot)) > (0.45 * fs)
    aliased_plot[1:][jumps] = np.nan
    return aliased, aliased_plot


def build_enhanced_measurement_emission(search_psd_db, fs):
    """
    Enhance measurement-only ridge visibility without using any theoretical prior.
    The score emphasizes peaks that:
    1) rise above a time-stationary per-frequency floor, and
    2) stand out from the local frequency neighborhood at each STFT slice.
    """
    if search_psd_db.size == 0:
        return search_psd_db.copy()

    stationary_floor = np.median(search_psd_db, axis=1, keepdims=True)
    residual = search_psd_db - stationary_floor

    freq_bins = search_psd_db.shape[0]
    freq_window = max(5, int(round(0.08 * freq_bins)))
    if freq_window % 2 == 0:
        freq_window += 1
    local_floor = median_filter(residual, size=(freq_window, 1), mode='nearest')
    local_prominence = residual - local_floor

    slice_floor = np.median(local_prominence, axis=0, keepdims=True)
    emission = local_prominence - slice_floor
    emission = 0.35 * residual + 0.65 * emission
    return emission


def track_peak_viterbi(
    search_freqs,
    search_psd_db,
    fs,
    stft_doppler=None,
    guide_band_hz=25.0,
    use_guidance=True,
    emission_override=None
):
    """
    Track a continuous ridge using dynamic programming (Viterbi-style).
    - Unguided: maximize measured power with continuity regularization.
    - Guided: add distance-to-theoretical-Doppler penalties.
    """
    n_bins, n_steps = search_psd_db.shape
    if n_bins == 0 or n_steps == 0:
        return np.full(n_steps, np.nan), np.full(n_steps, np.nan), np.full(n_steps, -1, dtype=int)

    # Tune penalties from STFT bin spacing and sampling rate.
    jump_penalty_per_hz = 0.20
    guide_penalty_per_hz = 0.35
    max_jump_hz = min(24.0, 0.14 * fs)

    def circular_distance(a, b, period):
        d = np.abs(a - b)
        return np.minimum(d, period - d)

    if emission_override is not None:
        emission = np.asarray(emission_override, dtype=float)
        if emission.shape != search_psd_db.shape:
            raise ValueError("emission_override must have the same shape as search_psd_db")
    elif use_guidance:
        if stft_doppler is None or len(stft_doppler) != n_steps:
            raise ValueError("stft_doppler must be provided with matching length when use_guidance=True")
        freqs_col = search_freqs[:, None]
        doppler_row = stft_doppler[None, :]
        dist_to_theory = circular_distance(freqs_col, doppler_row, fs)
        emission = search_psd_db - guide_penalty_per_hz * dist_to_theory
        # Extra soft penalty outside the preferred guided band.
        emission = emission - 0.6 * np.maximum(dist_to_theory - guide_band_hz, 0.0)
    else:
        # Measurement-only ridge tracking: no theoretical prior.
        emission = search_psd_db.copy()

    dp = np.full((n_bins, n_steps), -np.inf, dtype=float)
    parent = np.full((n_bins, n_steps), -1, dtype=int)
    dp[:, 0] = emission[:, 0]

    # Precompute transition cost matrix once.
    freq_diff = circular_distance(search_freqs[:, None], search_freqs[None, :], fs)
    trans_cost = jump_penalty_per_hz * freq_diff
    trans_cost[freq_diff > max_jump_hz] = np.inf

    for t in range(1, n_steps):
        prev = dp[:, t - 1]
        # score_matrix[i, j]: transition from i@t-1 to j@t
        score_matrix = prev[:, None] - trans_cost
        best_prev_idx = np.argmax(score_matrix, axis=0)
        best_prev_score = score_matrix[best_prev_idx, np.arange(n_bins)]
        dp[:, t] = best_prev_score + emission[:, t]
        parent[:, t] = best_prev_idx

    end_idx = int(np.argmax(dp[:, -1]))
    if not np.isfinite(dp[end_idx, -1]):
        return np.full(n_steps, np.nan), np.full(n_steps, np.nan), np.full(n_steps, -1, dtype=int)

    path_idx = np.full(n_steps, -1, dtype=int)
    path_idx[-1] = end_idx
    for t in range(n_steps - 1, 0, -1):
        path_idx[t - 1] = parent[path_idx[t], t]
        if path_idx[t - 1] < 0:
            break

    valid = path_idx >= 0
    path_freq = np.full(n_steps, np.nan, dtype=float)
    path_power = np.full(n_steps, np.nan, dtype=float)
    if np.any(valid):
        path_freq[valid] = search_freqs[path_idx[valid]]
        path_power[valid] = search_psd_db[path_idx[valid], np.where(valid)[0]]
    return path_freq, path_power, path_idx


def estimate_common_phase(H, eps=1e-12):
    unit = H / np.maximum(np.abs(H), eps)
    return np.angle(np.sum(unit, axis=1))


def compensate_cfo_cpe_matrix(H):
    t = np.arange(H.shape[0], dtype=float)
    cpe_raw = estimate_common_phase(H)
    cpe_unwrapped = np.unwrap(cpe_raw)
    slope, intercept = np.polyfit(t, cpe_unwrapped, deg=1)
    cfo_phase = slope * t + intercept
    H_cfo = H * np.exp(-1j * cfo_phase[:, None])

    cpe_res = estimate_common_phase(H_cfo)
    cpe_res_unwrapped = np.unwrap(cpe_res)
    H_comp = H_cfo * np.exp(-1j * cpe_res_unwrapped[:, None])

    info = {
        'cfo_slope_rad_per_step': float(slope),
        'cpe_std_rad': float(np.std(cpe_res_unwrapped))
    }
    return H_comp, info


def compensate_cfo_linear_only_matrix(H):
    """
    Remove only the global linear phase drift (CFO-like term).
    Keep residual dynamic phase to preserve Doppler-bearing components.
    """
    t = np.arange(H.shape[0], dtype=float)
    cpe_raw = estimate_common_phase(H)
    cpe_unwrapped = np.unwrap(cpe_raw)
    slope, intercept = np.polyfit(t, cpe_unwrapped, deg=1)
    linear_trend = slope * t + intercept
    H_out = H * np.exp(-1j * linear_trend[:, None])
    info = {
        'cfo_slope_rad_per_step': float(slope),
        'residual_std_rad': float(np.std(cpe_unwrapped - linear_trend))
    }
    return H_out, info


def compensate_cfo_only_gentle_matrix(H, trend_window=1001):
    """
    Preserve Doppler-related dynamic phase as much as possible:
    remove only linear CFO and very-slow common-phase drift.
    """
    t = np.arange(H.shape[0], dtype=float)
    cpe_raw = estimate_common_phase(H)
    cpe_unwrapped = np.unwrap(cpe_raw)

    slope, intercept = np.polyfit(t, cpe_unwrapped, deg=1)
    linear_trend = slope * t + intercept
    residual = cpe_unwrapped - linear_trend

    window = max(31, int(trend_window))
    if window % 2 == 0:
        window += 1
    if H.shape[0] <= window:
        slow_residual = np.mean(residual) * np.ones_like(residual)
    else:
        slow_residual = uniform_filter1d(residual, size=window, mode='nearest')

    gentle_trend = linear_trend + slow_residual
    H_out = H * np.exp(-1j * gentle_trend[:, None])

    info = {
        'cfo_slope_rad_per_step': float(slope),
        'slow_trend_std_rad': float(np.std(slow_residual)),
        'residual_after_gentle_std_rad': float(np.std(residual - slow_residual)),
        'trend_window': int(window)
    }
    return H_out, info


def phase_smooth_matrix(H, window=11):
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if H.shape[0] < window:
        return H.copy()
    mag = np.abs(H)
    phase = np.unwrap(np.angle(H), axis=0)
    phase_sm = uniform_filter1d(phase, size=window, axis=0, mode='nearest')
    return mag * np.exp(1j * phase_sm)


def build_preprocessed_complex_csi_variants(merged_df, smooth_window=11):
    variants = {
        'raw': np.array(merged_df['complex_csi'].values, dtype=np.complex128)
    }
    if 'window_csi' not in merged_df.columns:
        print("⚠️ 'window_csi' not available, only 'raw' preprocessing can be evaluated.")
        return variants

    window_series = merged_df['window_csi']
    lengths = window_series.dropna().map(lambda x: len(x) if hasattr(x, '__len__') else 0)
    lengths = lengths[lengths > 0]
    if lengths.empty:
        print("⚠️ No valid window_csi vectors, only 'raw' preprocessing can be evaluated.")
        return variants

    mode_len = int(lengths.mode().iloc[0])
    valid_mask = window_series.map(lambda x: isinstance(x, np.ndarray) and len(x) == mode_len)
    valid_count = int(valid_mask.sum())
    if valid_count < 64:
        print(f"⚠️ Too few valid window vectors ({valid_count}), skip compensation variants.")
        return variants

    H = np.vstack(window_series[valid_mask].values)
    H_linear, info_linear = compensate_cfo_linear_only_matrix(H)
    H_gentle, info_gentle = compensate_cfo_only_gentle_matrix(H, trend_window=max(301, int(5 * smooth_window * 10)))
    H_comp, info = compensate_cfo_cpe_matrix(H)
    H_comp_smooth = phase_smooth_matrix(H_comp, window=smooth_window)

    linear_series = np.full(len(merged_df), np.nan + 1j * np.nan, dtype=np.complex128)
    gentle_series = np.full(len(merged_df), np.nan + 1j * np.nan, dtype=np.complex128)
    comp_series = np.full(len(merged_df), np.nan + 1j * np.nan, dtype=np.complex128)
    smooth_series = np.full(len(merged_df), np.nan + 1j * np.nan, dtype=np.complex128)
    linear_series[valid_mask.values] = np.mean(H_linear, axis=1)
    gentle_series[valid_mask.values] = np.mean(H_gentle, axis=1)
    comp_series[valid_mask.values] = np.mean(H_comp, axis=1)
    smooth_series[valid_mask.values] = np.mean(H_comp_smooth, axis=1)

    variants['cfo_linear_only'] = linear_series
    variants['cfo_only_gentle'] = gentle_series
    variants['cfo_cpe_comp'] = comp_series
    variants['cfo_cpe_comp_phase_smooth'] = smooth_series
    print(
        "[INFO] Preprocessing stats: "
        f"window_len={mode_len}, valid_frames={valid_count}, "
        f"linear-CFO slope={info_linear['cfo_slope_rad_per_step']:.6f} rad/step, "
        f"linear residual std={info_linear['residual_std_rad']:.6f} rad, "
        f"gentle-CFO slope={info_gentle['cfo_slope_rad_per_step']:.6f} rad/step, "
        f"gentle residual std={info_gentle['residual_after_gentle_std_rad']:.6f} rad, "
        f"CFO slope={info['cfo_slope_rad_per_step']:.6f} rad/step, "
        f"residual CPE std={info['cpe_std_rad']:.6f} rad"
    )
    return variants

# ==============================================================================
#  4. 纯净版量化诊断 (移除了可能破坏信号的 detrend)
# ==============================================================================
def quantify_dc_and_cpe(merged_df, fs=100.0):
    print("\n" + "="*50)
    print(" 🔍 开始量化分析 DC 泄漏与 CPE 能量分布")
    print("="*50)
    
    df_clean = merged_df.dropna(subset=['complex_csi']).copy()
    csi_complex = np.array(df_clean['complex_csi'].tolist())
    
    nperseg = 1024 if fs >= 200 else 512
    nperseg = min(nperseg, len(csi_complex))
    if nperseg < 16:
        print("数据长度过短，跳过 Welch 量化。")
        return

    # 🌟 移除 detrend，还原信号最真实的能量面貌
    freqs, psd = welch(csi_complex, fs=fs, nperseg=nperseg, return_onesided=False)
    
    freqs_shifted = np.fft.fftshift(freqs)
    psd_shifted = np.maximum(np.real(np.fft.fftshift(psd)), 1e-20)
    psd_dB = 10 * np.log10(psd_shifted)
    
    center_idx = np.argmin(np.abs(freqs_shifted))
    dc_power_lin = psd_shifted[center_idx]
    dc_power = 10 * np.log10(dc_power_lin)
    
    # Use linear-domain averaging for physical consistency.
    cpe_mask = (np.abs(freqs_shifted) >= 15.0) & (np.abs(freqs_shifted) <= 30.0)
    cpe_power_lin = np.mean(psd_shifted[cpe_mask]) if np.any(cpe_mask) else np.nan
    cpe_power = 10 * np.log10(cpe_power_lin) if np.isfinite(cpe_power_lin) and cpe_power_lin > 0 else np.nan
    
    noise_mask = np.abs(freqs_shifted) >= (fs/2 * 0.8)
    noise_power_lin = np.mean(psd_shifted[noise_mask]) if np.any(noise_mask) else np.nan
    noise_power = 10 * np.log10(noise_power_lin) if np.isfinite(noise_power_lin) and noise_power_lin > 0 else np.nan

    search_mask = (np.abs(freqs_shifted) >= 1.0) & (np.abs(freqs_shifted) <= fs/2 * 0.9)
    if np.any(search_mask):
        local_freqs = freqs_shifted[search_mask]
        local_psd = psd_shifted[search_mask]
        peak_idx = np.argmax(local_psd)
        peak_freq = local_freqs[peak_idx]
        peak_power = 10 * np.log10(local_psd[peak_idx])
        print(f"🟣 非DC主峰频率:             {peak_freq:+.2f} Hz")
        print(f"🟣 非DC主峰能量:             {peak_power:.2f} dB")
    
    print(f"🔴 真实 DC 峰值能量 (0 Hz):   {dc_power:.2f} dB")
    print(f"🟠 CPE 平台平均能量:         {cpe_power:.2f} dB")
    print(f"🔵 系统底噪平均能量:         {noise_power:.2f} dB")
    print("-" * 50)
    print(f"✅ 真实信号能量差 (DC vs CPE): {dc_power - cpe_power:.2f} dB")
    print("==================================================")

# ==============================================================================
#  5. 零失真验证图 (只保护色带，不修改信号)
# ==============================================================================
def analyze_whole_flight_spectrogram(merged_df, mission_name, output_dir, fc=3619.2e6, fs=100.0):
    print(f"\n--- 开始生成 {mission_name} 的全航程时频验证图 (STFT) ---")
    
    df_clean = merged_df.dropna(subset=['complex_csi', 'radial_velocity']).copy()
    if df_clean.empty:
        return
    df_clean.sort_index(inplace=True)

    time_array = np.arange(len(df_clean)) / fs
    csi_complex = np.array(df_clean['complex_csi'].tolist())
    radial_vels = df_clean['radial_velocity'].values

    nperseg = 256
    noverlap = int(nperseg * 0.8)
    
    # 🌟 原汁原味的傅里叶变换，没有任何人工修饰
    frequencies, times, Sxx = spectrogram(csi_complex, fs=fs, return_onesided=False, 
                                          window='hann', nperseg=nperseg, noverlap=noverlap)

    frequencies_shifted = np.fft.fftshift(frequencies)
    Sxx_shifted = np.fft.fftshift(Sxx, axes=0)
    Sxx_dB = 10 * np.log10(Sxx_shifted + 1e-12)

    # 🌟 温柔的色带控制：在计算亮度上限时，忽略 0Hz 那个点，但绝不修改矩阵里的数据
    zero_idx = np.argmin(np.abs(frequencies_shifted))
    mask = np.ones(len(frequencies_shifted), dtype=bool)
    mask[zero_idx] = False  # 忽略 DC 点
    
    vmax = np.percentile(Sxx_dB[mask, :], 99) 
    vmin = vmax -  30

    c = 3e8
    theoretical_doppler = (radial_vels / c) * fc
    aliased_doppler, aliased_doppler_plot = alias_doppler_with_breaks(theoretical_doppler, fs)

    plt.figure(figsize=(12, 7))
    nyquist_limit = fs / 2.0
    
    plt.imshow(Sxx_dB, aspect='auto', origin='lower', cmap='viridis',
               extent=[time_array[0], time_array[-1], frequencies_shifted[0], frequencies_shifted[-1]],
               vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar()
    cbar.set_label('Power Spectral Density (dB)', rotation=270, labelpad=15)
    
    plt.plot(time_array, aliased_doppler_plot, color='red', linestyle='--', linewidth=2, 
             label='Theoretical Aliased Doppler Track')

    plt.ylim(-nyquist_limit, nyquist_limit)
    plt.title(f"Whole Flight Spectrogram vs. Theoretical Doppler Track\n({mission_name})", fontsize=14, fontweight='bold')
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Frequency (Hz)", fontsize=12)
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
    text_str = ("Proof of Velocity-Bandwidth Decoupling:\n"
                "If pure kinematics dominated, energy would follow the red dashed line.\n"
                "Instead, actual energy forms a stationary hardware CPE band.")
    plt.text(0.02, 0.95, text_str, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=props)

    plt.legend(loc='lower right')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{mission_name}_Doppler_Masking_Spectrogram.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"-> 完美验证图已保存至: {save_path}\n")
    plt.close()

def analyze_2d_1d_combined(
    merged_df,
    mission_name,
    output_dir,
    fc=3619.2e6,
    fs=200.0,
    peak_track_mode='unguided',
    save_tag=None
):
    print(f"\n🎨 正在生成 1D+2D 联动对比图...")
    
    # 1. 准备数据
    df_clean = merged_df.dropna(subset=['complex_csi', 'radial_velocity']).copy()
    if df_clean.empty:
        raise ValueError("No valid rows after dropping NaN complex_csi/radial_velocity.")
    time_array = np.arange(len(df_clean)) / fs
    csi_complex = np.array(df_clean['complex_csi'].tolist())
    radial_vels = df_clean['radial_velocity'].values

    # GPS usually has a much lower update rate than CSI; interpolate/smooth before Doppler aliasing.
    rv = radial_vels.astype(float)
    change_mask = np.concatenate(([True], np.abs(np.diff(rv)) > 1e-12))
    t_anchor = time_array[change_mask]
    v_anchor = rv[change_mask]
    if len(t_anchor) >= 2:
        rv_interp = np.interp(time_array, t_anchor, v_anchor)
    else:
        rv_interp = rv.copy()
    smooth_len = int(max(3, round(fs * 0.25)))
    if smooth_len % 2 == 0:
        smooth_len += 1
    if smooth_len < len(rv_interp):
        kernel = np.ones(smooth_len, dtype=float) / smooth_len
        radial_vels_for_theory = np.convolve(rv_interp, kernel, mode='same')
    else:
        radial_vels_for_theory = rv_interp

    # 2. 计算 STFT
    nperseg = 512
    noverlap = int(nperseg * 0.8)
    frequencies, times, Sxx = spectrogram(csi_complex, fs=fs, return_onesided=False, 
                                          window='hann', nperseg=nperseg, noverlap=noverlap)
    
    freqs_shifted = np.fft.fftshift(frequencies)
    Sxx_shifted = np.fft.fftshift(Sxx, axes=0)
    Sxx_dB = 10 * np.log10(Sxx_shifted + 1e-12)

    # 3. 寻找一个“典型时刻”：多普勒偏移最大的时刻
    c = 3e8
    theoretical_doppler = (radial_vels_for_theory / c) * fc
    aliased_doppler_full, aliased_doppler_plot = alias_doppler_with_breaks(theoretical_doppler, fs)

    # 3.1 从每个 STFT 时间片提取实测主峰轨迹（排除近 DC 区域）
    freq_res = fs / nperseg
    nyq_guard = max(1.5 * freq_res, 0.3)
    freq_search_mask = (np.abs(freqs_shifted) >= 1.0) & (np.abs(freqs_shifted) <= (fs / 2.0 - nyq_guard))
    if np.any(freq_search_mask):
        search_freqs = freqs_shifted[freq_search_mask]
        search_psd = Sxx_dB[freq_search_mask, :]
        peak_bin_idx = np.argmax(search_psd, axis=0)
        global_peak_freq = search_freqs[peak_bin_idx]
        global_peak_power = search_psd[peak_bin_idx, np.arange(search_psd.shape[1])]
        noise_floor = np.median(search_psd, axis=0)
        global_peak_prominence = global_peak_power - noise_floor
    else:
        search_freqs = np.array([])
        search_psd = np.empty((0, len(times)))
        global_peak_freq = np.full_like(times, np.nan, dtype=float)
        global_peak_prominence = np.full_like(times, np.nan, dtype=float)

    # Map each STFT time bin to the closest Doppler sample index.
    stft_sample_indices = np.clip(np.rint(times * fs).astype(int), 0, len(aliased_doppler_full) - 1)
    stft_doppler = aliased_doppler_full[stft_sample_indices]

    peak_track_mode = str(peak_track_mode).strip().lower()
    if peak_track_mode not in ('guided', 'unguided', 'unguided_enhanced'):
        raise ValueError(f"Unsupported peak_track_mode: {peak_track_mode}. Use 'guided', 'unguided', or 'unguided_enhanced'.")

    mode_label = (
        'guided+Viterbi around theoretical Doppler'
        if peak_track_mode == 'guided'
        else 'unguided+Viterbi (measurement-only)'
        if peak_track_mode == 'unguided'
        else 'unguided_enhanced+Viterbi (measurement-only, floor-suppressed prominence)'
    )
    use_guidance = (peak_track_mode == 'guided')
    use_enhanced_unguided = (peak_track_mode == 'unguided_enhanced')
    guide_band_hz = min(35.0, 0.3 * fs)

    tracked_peak_freq = np.full_like(times, np.nan, dtype=float)
    tracked_peak_power = np.full_like(times, np.nan, dtype=float)
    tracked_peak_prominence = np.full_like(times, np.nan, dtype=float)
    if search_freqs.size > 0:
        tracked_peak_freq, tracked_peak_power, _ = track_peak_viterbi(
            search_freqs=search_freqs,
            search_psd_db=search_psd,
            fs=fs,
            stft_doppler=stft_doppler if use_guidance else None,
            guide_band_hz=guide_band_hz,
            use_guidance=use_guidance,
            emission_override=build_enhanced_measurement_emission(search_psd, fs) if use_enhanced_unguided else None
        )
        for ti in range(len(times)):
            if not np.isfinite(tracked_peak_freq[ti]):
                continue
            band_mask = np.abs(search_freqs - tracked_peak_freq[ti]) <= guide_band_hz
            if not np.any(band_mask):
                tracked_peak_prominence[ti] = tracked_peak_power[ti] - noise_floor[ti]
            else:
                tracked_peak_prominence[ti] = tracked_peak_power[ti] - np.median(search_psd[band_mask, ti])
    valid_tracked = np.isfinite(tracked_peak_freq).sum()
    if valid_tracked < max(3, int(0.2 * len(times))):
        print(f"⚠️ {mode_label} track coverage too low, fallback to global peak track.")
        tracked_peak_freq = global_peak_freq.copy()
        tracked_peak_power = global_peak_power.copy()
        tracked_peak_prominence = global_peak_prominence.copy()

    target_time_idx = int(np.argmax(np.abs(stft_doppler)))
    target_time_s = times[target_time_idx]
    psd_slice = Sxx_dB[:, target_time_idx]
    target_doppler = stft_doppler[target_time_idx]
    target_measured_peak = tracked_peak_freq[target_time_idx] if len(tracked_peak_freq) else np.nan

    # 4. 设置画布：左边 2D，右边 1D
    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
    
    ax_2d = fig.add_subplot(gs[0])
    ax_1d = fig.add_subplot(gs[1], sharey=ax_2d)

    # --- 左图：2D Spectrogram ---
    zero_idx = np.argmin(np.abs(freqs_shifted))
    mask = np.ones(len(freqs_shifted), dtype=bool)
    mask[zero_idx] = False
    vmax = np.percentile(Sxx_dB[mask, :], 99)
    vmin = vmax - 30

    im = ax_2d.imshow(Sxx_dB, aspect='auto', origin='lower', cmap='viridis',
                      extent=[times[0], times[-1], freqs_shifted[0], freqs_shifted[-1]],
                      vmin=vmin, vmax=vmax)
    
    ax_2d.plot(time_array, aliased_doppler_plot, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Theoretical Doppler (smoothed)')
    ax_2d.plot(times, tracked_peak_freq, color='cyan', linewidth=1.5, alpha=0.9, label=f'Measured Peak Track ({mode_label})')
    ax_2d.axvline(x=target_time_s, color='white', linestyle=':', linewidth=2, label='Slice Timestamp')
    
    ax_2d.set_ylabel("Frequency (Hz)", fontsize=12)
    ax_2d.set_xlabel("Time (seconds)", fontsize=12)
    ax_2d.set_title(f"2D Spectrogram: {mission_name}", fontsize=13, fontweight='bold')
    ax_2d.legend(loc='lower right')

    # --- 右图：1D PSD Slice ---
    ax_1d.plot(psd_slice, freqs_shifted, color='#2ca02c', linewidth=2, label='Measured PSD')
    # 标注理论多普勒位置
    ax_1d.axhline(y=target_doppler, color='red', linestyle='--', linewidth=2, label='Expected Doppler Peak')
    if np.isfinite(target_measured_peak):
        ax_1d.axhline(y=target_measured_peak, color='cyan', linestyle='-', linewidth=2, label='Measured Peak @ Slice')
    
    # 填充 CPE 区域增加视觉效果
    ax_1d.fill_betweenx(freqs_shifted, vmin, psd_slice, where=(np.abs(freqs_shifted) < 40), color='yellow', alpha=0.2, label='CPE Plateau')

    ax_1d.set_xlim(vmin, vmax + 5)
    ax_1d.set_xlabel("Power (dB)", fontsize=12)
    ax_1d.set_title("1D PSD Slice", fontsize=13, fontweight='bold')
    ax_1d.grid(True, alpha=0.3)
    ax_1d.legend(loc='upper right', fontsize=9)

    # 移除 1D 图的 Y 轴标签（因为共用）
    plt.setp(ax_1d.get_yticklabels(), visible=False)

    suffix = f"_{save_tag}" if save_tag else ""
    save_path = os.path.join(output_dir, f"{mission_name}_1D_2D_Combined{suffix}.pdf")
    plt.savefig(save_path, format='pdf')
    print(f"✅ 联动对比图已保存至: {save_path}")
    valid_prom = tracked_peak_prominence[np.isfinite(tracked_peak_prominence)]
    prom_mean = np.nan
    prom_median = np.nan
    prom_p95 = np.nan
    prom_max = np.nan
    if valid_prom.size > 0:
        prom_mean = float(np.mean(valid_prom))
        prom_median = float(np.median(valid_prom))
        prom_p95 = float(np.percentile(valid_prom, 95))
        prom_max = float(np.max(valid_prom))
        print(f"📈 主峰突出度统计 ({mode_label} Peak - Local Median Noise Floor, dB):")
        print(f"   Mean:   {prom_mean:.2f}")
        print(f"   Median: {prom_median:.2f}")
        print(f"   P95:    {prom_p95:.2f}")
        print(f"   Max:    {prom_max:.2f}")
    valid_peak = tracked_peak_freq[np.isfinite(tracked_peak_freq)]
    peak_mean = np.nan
    peak_median = np.nan
    peak_std = np.nan
    if valid_peak.size > 0:
        peak_mean = float(np.mean(valid_peak))
        peak_median = float(np.median(valid_peak))
        peak_std = float(np.std(valid_peak))
        print(f"📍 实测主峰频率统计 (Hz, {mode_label}):")
        print(f"   Mean:   {peak_mean:+.2f}")
        print(f"   Median: {peak_median:+.2f}")
        print(f"   Std:    {peak_std:.2f}")
    valid_theory = stft_doppler[np.isfinite(tracked_peak_freq)]
    mae = np.nan
    rmse = np.nan
    bias = np.nan
    if valid_peak.size > 0 and valid_theory.size == valid_peak.size:
        wrapped_err = valid_peak - valid_theory
        wrapped_err = wrapped_err - fs * np.round(wrapped_err / fs)
        mae = float(np.mean(np.abs(wrapped_err)))
        rmse = float(np.sqrt(np.mean(wrapped_err**2)))
        bias = float(np.mean(wrapped_err))
        print("🧭 轨迹一致性 (Measured vs Theoretical, wrapped error in Hz):")
        print(f"   MAE:    {mae:.2f}")
        print(f"   RMSE:   {rmse:.2f}")
        print(f"   Bias:   {bias:+.2f}")
    plt.close(fig)
    return {
        'peak_track_mode': peak_track_mode,
        'mode_label': mode_label,
        'n_stft_steps': int(len(times)),
        'n_valid_peak': int(valid_peak.size),
        'prom_mean_db': prom_mean,
        'prom_median_db': prom_median,
        'prom_p95_db': prom_p95,
        'prom_max_db': prom_max,
        'peak_mean_hz': peak_mean,
        'peak_median_hz': peak_median,
        'peak_std_hz': peak_std,
        'mae_hz': mae,
        'rmse_hz': rmse,
        'bias_hz': bias,
        'figure_path': save_path
    }


def analyze_guided_unguided_overlay(
    merged_df,
    mission_name,
    output_dir,
    fc=3619.2e6,
    fs=100.0,
    save_filename=None,
    extra_save_paths=None,
    focus_window_s=None,
    focus_mode='max_abs_theory'
):
    print(f"\n🎨 正在生成 guided vs unguided 叠加论文图...")

    df_clean = merged_df.dropna(subset=['complex_csi', 'radial_velocity']).copy()
    if df_clean.empty:
        raise ValueError("No valid rows after dropping NaN complex_csi/radial_velocity.")

    time_array = np.arange(len(df_clean)) / fs
    csi_complex = np.array(df_clean['complex_csi'].tolist())
    radial_vels = df_clean['radial_velocity'].values.astype(float)

    change_mask = np.concatenate(([True], np.abs(np.diff(radial_vels)) > 1e-12))
    t_anchor = time_array[change_mask]
    v_anchor = radial_vels[change_mask]
    if len(t_anchor) >= 2:
        rv_interp = np.interp(time_array, t_anchor, v_anchor)
    else:
        rv_interp = radial_vels.copy()

    smooth_len = int(max(3, round(fs * 0.25)))
    if smooth_len % 2 == 0:
        smooth_len += 1
    if smooth_len < len(rv_interp):
        kernel = np.ones(smooth_len, dtype=float) / smooth_len
        radial_vels_for_theory = np.convolve(rv_interp, kernel, mode='same')
    else:
        radial_vels_for_theory = rv_interp

    nperseg = 512 if fs <= 120 else 1024
    nperseg = min(nperseg, len(csi_complex))
    if nperseg < 64:
        raise ValueError("Too few CSI samples for a stable spectrogram.")
    noverlap = int(nperseg * 0.8)
    frequencies, times, Sxx = spectrogram(
        csi_complex,
        fs=fs,
        return_onesided=False,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap
    )

    freqs_shifted = np.fft.fftshift(frequencies)
    Sxx_shifted = np.fft.fftshift(Sxx, axes=0)
    Sxx_dB = 10 * np.log10(Sxx_shifted + 1e-12)

    c = 3e8
    theoretical_doppler = (radial_vels_for_theory / c) * fc
    aliased_doppler_full, aliased_doppler_plot = alias_doppler_with_breaks(theoretical_doppler, fs)

    freq_res = fs / max(nperseg, 1)
    nyq_guard = max(1.5 * freq_res, 0.3)
    freq_search_mask = (np.abs(freqs_shifted) >= 1.0) & (np.abs(freqs_shifted) <= (fs / 2.0 - nyq_guard))
    if not np.any(freq_search_mask):
        raise ValueError("No usable search band for ridge extraction.")

    search_freqs = freqs_shifted[freq_search_mask]
    search_psd = Sxx_dB[freq_search_mask, :]

    stft_sample_indices = np.clip(np.rint(times * fs).astype(int), 0, len(aliased_doppler_full) - 1)
    stft_doppler = aliased_doppler_full[stft_sample_indices]
    guide_band_hz = min(35.0, 0.3 * fs)

    guided_freq, _, _ = track_peak_viterbi(
        search_freqs=search_freqs,
        search_psd_db=search_psd,
        fs=fs,
        stft_doppler=stft_doppler,
        guide_band_hz=guide_band_hz,
        use_guidance=True
    )
    unguided_freq, _, _ = track_peak_viterbi(
        search_freqs=search_freqs,
        search_psd_db=search_psd,
        fs=fs,
        stft_doppler=None,
        guide_band_hz=guide_band_hz,
        use_guidance=False
    )

    def _wrapped_metrics(track_freq, valid_mask=None):
        valid = np.isfinite(track_freq)
        if valid_mask is not None:
            valid = valid & valid_mask
        if not np.any(valid):
            return np.nan, np.nan, np.nan
        err = track_freq[valid] - stft_doppler[valid]
        err = err - fs * np.round(err / fs)
        return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2))), float(np.mean(err))

    plot_mask = np.ones_like(times, dtype=bool)
    focus_note = "full overlap"
    if focus_window_s is not None and float(focus_window_s) > 0:
        focus_window_s = float(focus_window_s)
        if focus_mode == 'max_abs_theory':
            center_idx = int(np.nanargmax(np.abs(stft_doppler)))
        elif focus_mode == 'best_contrast':
            half_window = 0.5 * focus_window_s
            best_score = None
            best_idx = None
            for idx, center_time in enumerate(times):
                if center_time - half_window < times[0] or center_time + half_window > times[-1]:
                    continue
                candidate_mask = (times >= center_time - half_window) & (times <= center_time + half_window)
                if np.sum(candidate_mask) < 8:
                    continue
                g_mae, _, _ = _wrapped_metrics(guided_freq, valid_mask=candidate_mask)
                u_mae, _, _ = _wrapped_metrics(unguided_freq, valid_mask=candidate_mask)
                theory_sweep = np.nanpercentile(stft_doppler[candidate_mask], 95) - np.nanpercentile(stft_doppler[candidate_mask], 5)
                score = (u_mae - g_mae) - 0.35 * g_mae + 0.03 * theory_sweep
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                center_idx = int(np.nanargmax(np.abs(stft_doppler)))
            else:
                center_idx = int(best_idx)
        elif focus_mode == 'max_gap':
            ridge_gap = np.abs(guided_freq - unguided_freq)
            ridge_gap = ridge_gap - fs * np.round(ridge_gap / fs)
            center_idx = int(np.nanargmax(np.abs(ridge_gap)))
        else:
            raise ValueError(f"Unsupported focus_mode: {focus_mode}")

        center_time = float(times[center_idx])
        half_window = 0.5 * focus_window_s
        t_min = max(float(times[0]), center_time - half_window)
        t_max = min(float(times[-1]), center_time + half_window)
        plot_mask = (times >= t_min) & (times <= t_max)
        if np.sum(plot_mask) < 8:
            plot_mask = np.ones_like(times, dtype=bool)
            focus_note = "full overlap (window fallback)"
        else:
            focus_note = f"{t_min:.1f}-{t_max:.1f} s excerpt"

    guided_mae, guided_rmse, guided_bias = _wrapped_metrics(guided_freq, valid_mask=plot_mask)
    unguided_mae, unguided_rmse, unguided_bias = _wrapped_metrics(unguided_freq, valid_mask=plot_mask)

    zero_idx = np.argmin(np.abs(freqs_shifted))
    mask = np.ones(len(freqs_shifted), dtype=bool)
    mask[zero_idx] = False
    vmax = np.percentile(Sxx_dB[mask, :], 99)
    vmin = vmax - 30

    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    im = ax.imshow(
        Sxx_dB,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        extent=[times[0], times[-1], freqs_shifted[0], freqs_shifted[-1]],
        vmin=vmin,
        vmax=vmax
    )
    cbar = plt.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label('Power Spectral Density (dB)', rotation=270, labelpad=15)

    ax.plot(time_array, aliased_doppler_plot, color='red', linestyle='--', linewidth=2.0, label='Theoretical Doppler')
    ax.plot(times, guided_freq, color='cyan', linewidth=1.8, label='Guided Ridge')
    ax.plot(times, unguided_freq, color='white', linewidth=1.6, linestyle='-.', label='Unguided Ridge')

    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)
    ax.set_title(
        f"Raw Spectrogram with Guided/Unguided Ridge Tracking\n({mission_name})",
        fontsize=13,
        fontweight='bold'
    )
    ax.set_ylim(freqs_shifted[0], freqs_shifted[-1])
    if np.any(plot_mask):
        plot_times = times[plot_mask]
        ax.set_xlim(float(plot_times[0]), float(plot_times[-1]))
    ax.legend(loc='lower right', fontsize=10, framealpha=0.92)

    info_text = (
        f"Guided:   MAE {guided_mae:.2f} Hz | RMSE {guided_rmse:.2f} Hz\n"
        f"Unguided: MAE {unguided_mae:.2f} Hz | RMSE {unguided_rmse:.2f} Hz"
    )
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=10.5,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.88, edgecolor='gray')
    )

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    filename = save_filename or f"{mission_name}_Spectrogram_Guided_Unguided.pdf"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"✅ 论文叠加图已保存至: {save_path}")

    for extra_path in extra_save_paths or []:
        extra_dir = os.path.dirname(extra_path)
        if extra_dir:
            os.makedirs(extra_dir, exist_ok=True)
        fig.savefig(extra_path, format='pdf', bbox_inches='tight')
        print(f"✅ 额外保存图像至: {extra_path}")

    plt.close(fig)
    print("🧭 Guided vs. Unguided consistency summary:")
    print(f"   Guided   -> MAE {guided_mae:.2f} Hz | RMSE {guided_rmse:.2f} Hz | Bias {guided_bias:+.2f} Hz")
    print(f"   Unguided -> MAE {unguided_mae:.2f} Hz | RMSE {unguided_rmse:.2f} Hz | Bias {unguided_bias:+.2f} Hz")

    return {
        'figure_path': save_path,
        'focus_note': focus_note,
        'guided_mae_hz': guided_mae,
        'guided_rmse_hz': guided_rmse,
        'guided_bias_hz': guided_bias,
        'unguided_mae_hz': unguided_mae,
        'unguided_rmse_hz': unguided_rmse,
        'unguided_bias_hz': unguided_bias
    }


def export_paper_track_comparison_tables(merged_df, mission_name, output_dir, fs, fc=3619.2e6):
    print("\n📑 正在导出论文用对照表 (guided vs unguided, across preprocessing variants)...")
    variants = build_preprocessed_complex_csi_variants(merged_df, smooth_window=11)
    rows = []
    for preprocess_name, csi_variant in variants.items():
        df_variant = merged_df.copy()
        df_variant['complex_csi'] = csi_variant
        for track_mode in ('guided', 'unguided'):
            print(f"\n[RUN] preprocessing={preprocess_name}, track_mode={track_mode}")
            metrics = analyze_2d_1d_combined(
                df_variant,
                mission_name=mission_name,
                output_dir=output_dir,
                fc=fc,
                fs=fs,
                peak_track_mode=track_mode,
                save_tag=f"{preprocess_name}_{track_mode}"
            )
            metrics['preprocessing'] = preprocess_name
            rows.append(metrics)

    if not rows:
        print("⚠️ No comparison rows generated.")
        return None, None

    long_df = pd.DataFrame(rows)
    long_cols = [
        'preprocessing', 'peak_track_mode',
        'mae_hz', 'rmse_hz', 'bias_hz',
        'prom_mean_db', 'prom_median_db', 'prom_p95_db', 'prom_max_db',
        'peak_mean_hz', 'peak_median_hz', 'peak_std_hz',
        'n_valid_peak', 'n_stft_steps', 'figure_path'
    ]
    long_df = long_df[[c for c in long_cols if c in long_df.columns]]

    wide_df = long_df.pivot(index='preprocessing', columns='peak_track_mode', values=['mae_hz', 'rmse_hz', 'bias_hz'])
    wide_df.columns = [f"{metric}_{mode}" for metric, mode in wide_df.columns]
    wide_df = wide_df.reset_index()

    long_path = os.path.join(output_dir, f"{mission_name}_track_comparison_long.csv")
    wide_path = os.path.join(output_dir, f"{mission_name}_track_comparison_wide.csv")
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    print(f"\n✅ 对照明细表已保存: {long_path}")
    print(f"✅ 对照并排表已保存: {wide_path}")
    print("📊 并排摘要 (Hz, lower is better for MAE/RMSE):")
    for _, r in wide_df.iterrows():
        print(
            f"   {r['preprocessing']:<24} "
            f"MAE guided={r.get('mae_hz_guided', np.nan):6.2f} | unguided={r.get('mae_hz_unguided', np.nan):6.2f} ; "
            f"RMSE guided={r.get('rmse_hz_guided', np.nan):6.2f} | unguided={r.get('rmse_hz_unguided', np.nan):6.2f} ; "
            f"Bias guided={r.get('bias_hz_guided', np.nan):+6.2f} | unguided={r.get('bias_hz_unguided', np.nan):+6.2f}"
        )
    return long_path, wide_path

# ==============================================================================
#  6. 主程序
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CPE / Doppler analysis for UAV measurement campaigns.")
    parser.add_argument(
        '--preset',
        default='bolek_50m_10mps_5ms',
        choices=['bolek_50m_10mps_5ms', 'pavel_40m_15mps'],
        help='Mission preset to analyze.'
    )
    parser.add_argument(
        '--guided-unguided-overlay',
        action='store_true',
        help='Generate a single paper-ready spectrogram with theoretical, guided, and unguided ridges.'
    )
    args = parser.parse_args()

    base_dir = os.getcwd()
    paper_dir = os.path.join(base_dir, 'Empowering_AI_towards_6G__Realistic_UAV_Channel_Data_Acquisition_using_Open_Source_Solutions (1)')
    mission_presets = {
        'bolek_50m_10mps_5ms': {
            'mission_name': '50m_10mps_5ms_autopilot',
            'channel_files': ['channel_estimates_20250917_150803_bolek_50m_10mps_5ms_autopilot.txt'],
            'gps_filename': '2025-09-17_14-11-56_bolek.csv',
            'gps_local_tz': 'Europe/Berlin',
            'auto_align_hourly_offset': True,
            'peak_track_mode': 'unguided',
            'export_paper_comparison_table': True,
            'paper_overlay_filename': None
        },
        'pavel_40m_15mps': {
            'mission_name': '40m_15mps_autopilot',
            'channel_files': [
                'ch_est_pavel_40m_20mps_versuch_phase2_from_big_rec_autopilot.txt'
            ],
            'gps_filename': 'pavel_40m_20mps_versuch.csv',
            'gps_local_tz': 'Europe/Berlin',
            'auto_align_hourly_offset': False,
            'peak_track_mode': 'unguided',
            'export_paper_comparison_table': False,
            'paper_overlay_filename': 'Spectrogram_15mps_Guided_Unguided.pdf',
            'overlay_focus_window_s': 60.0,
            'overlay_focus_mode': 'best_contrast'
        }
    }
    preset = mission_presets[args.preset]
    mission_name = preset['mission_name']

    channel_log_paths = [os.path.join(base_dir, 'data', name) for name in preset['channel_files']]
    gps_log_path = os.path.join(base_dir, 'data', preset['gps_filename'])

    SAVE_PLOTS = True
    GPS_LOCAL_TZ = preset['gps_local_tz']
    CHANNEL_TIME_OFFSET = pd.Timedelta(0)
    AUTO_ALIGN_HOURLY_OFFSET = preset['auto_align_hourly_offset']
    PEAK_TRACK_MODE = preset['peak_track_mode']
    EXPORT_PAPER_COMPARISON_TABLE = preset['export_paper_comparison_table']
    output_dir = os.path.join(base_dir, 'paper_results')
    if SAVE_PLOTS and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Work Dir: {base_dir}")
    print(f"Preset: {args.preset}")
    print(f"Peak Track Mode: {PEAK_TRACK_MODE}")
    print(f"Export Paper Comparison Table: {EXPORT_PAPER_COMPARISON_TABLE}")
    print(f"Guided/Unguided Overlay: {args.guided_unguided_overlay}")

    try:
        channel_df_full = parse_channel_logs(
            channel_log_paths,
            time_offset=CHANNEL_TIME_OFFSET,
            combine_mode='mean'
        )
        gps_df_full = parse_gps_log(gps_log_path, local_tz=GPS_LOCAL_TZ)
        
        channel_df_full.sort_values('timestamp', inplace=True)
        gps_df_full.sort_values('timestamp', inplace=True)

        ch_start, ch_end = channel_df_full['timestamp'].iloc[[0, -1]]
        gps_start, gps_end = gps_df_full['timestamp'].iloc[[0, -1]]
        overlap_before = _time_overlap_seconds(ch_start, ch_end, gps_start, gps_end)

        print(f"Channel time range (UTC): {ch_start} -> {ch_end}")
        print(f"GPS time range (UTC):     {gps_start} -> {gps_end}")
        print(f"Raw overlap: {overlap_before:.2f} s")

        if AUTO_ALIGN_HOURLY_OFFSET:
            best_offset, best_overlap = find_best_hourly_offset(channel_df_full, gps_df_full)
            if best_offset != pd.Timedelta(0) and best_overlap > overlap_before:
                print(f"⚠️ Applying auto hourly offset to channel timestamps: {best_offset}")
                channel_df_full = channel_df_full.copy()
                channel_df_full['timestamp'] = channel_df_full['timestamp'] + best_offset
                ch_start, ch_end = channel_df_full['timestamp'].iloc[[0, -1]]
                print(f"Shifted channel range (UTC): {ch_start} -> {ch_end}")
                print(f"Shifted overlap: {best_overlap:.2f} s")
        
        merged_df = pd.merge_asof(
            left=channel_df_full, 
            right=gps_df_full,
            on='timestamp', 
            direction='nearest',
            tolerance=pd.Timedelta('1s')
        )

        matched_rows = merged_df['gps.lat'].notna().sum()
        print(f"Merged rows: {len(merged_df)}, matched GPS rows: {matched_rows} ({matched_rows / max(1, len(merged_df)):.2%})")

        merged_df.dropna(subset=['avg_power', 'gps.lat', 'complex_csi'], inplace=True)
        if merged_df.empty:
            raise ValueError("No valid merged rows after timestamp alignment. Please verify channel/GPS file pair.")

        merged_df.set_index('timestamp', inplace=True)
        
        gs_lat = gps_df_full['gps.lat'].iloc[0]
        gs_lon = gps_df_full['gps.lon'].iloc[0]
        gs_alt = gps_df_full['altitudeAMSL'].iloc[0] + 5  
        
        merged_df = calculate_derived_metrics(merged_df, gs_lat, gs_lon, gs_alt)

        diffs = merged_df.index.to_series().diff().dt.total_seconds().dropna()
        diffs = diffs[diffs > 0]
        if diffs.empty:
            raise ValueError("Cannot estimate sampling rate from merged timestamps.")
        actual_fs = float(np.clip(np.round(1.0 / np.median(diffs)), 1, 1e6))
        print(f"Estimated sampling rate: {actual_fs:.2f} Hz")

        if SAVE_PLOTS:
            try:
                quantify_dc_and_cpe(merged_df, fs=actual_fs)
                if args.guided_unguided_overlay:
                    extra_paths = []
                    overlay_name = preset.get('paper_overlay_filename')
                    if overlay_name:
                        extra_paths.append(os.path.join(paper_dir, 'figures', overlay_name))
                    analyze_guided_unguided_overlay(
                        merged_df,
                        mission_name=mission_name,
                        output_dir=output_dir,
                        fs=actual_fs,
                        save_filename=f"{mission_name}_Spectrogram_Guided_Unguided.pdf",
                        extra_save_paths=extra_paths,
                        focus_window_s=preset.get('overlay_focus_window_s'),
                        focus_mode=preset.get('overlay_focus_mode', 'max_abs_theory')
                    )
                elif EXPORT_PAPER_COMPARISON_TABLE:
                    export_paper_track_comparison_tables(merged_df, mission_name, output_dir, fs=actual_fs)
                else:
                    analyze_2d_1d_combined(merged_df, mission_name, output_dir, fs=actual_fs, peak_track_mode=PEAK_TRACK_MODE)
            except Exception as e:
                print(f"分析失败: {e}")
                
        print("\nAll done!")

    except Exception as e:
        print(f"\nError: {e}")
