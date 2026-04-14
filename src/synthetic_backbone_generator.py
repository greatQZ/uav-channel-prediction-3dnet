import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BackboneConfig:
    num_frames: int = 47959
    num_subcarriers: int = 1024
    mean_frame_power_db: float = 44.37868338087512
    std_frame_power_db: float = 3.752178242292282
    frame_power_lag1_corr: float = 0.9751529762008562
    phase_increment_std: float = 1.8132303370950915
    frequency_smoothing_span: int = 33
    residual_freq_smoothing_span: int = 17
    residual_std: float = 0.16
    residual_time_corr: float = 0.985
    shape_variation_std: float = 0.4
    shape_variation_lag1_corr: float = 0.95
    innovation_burst_std: float = 0.65
    innovation_burst_lag1_corr: float = 0.88
    event_rate_per_1k_frames: float = 8.0
    event_duration_frames: int = 16
    event_strength: float = 1.25
    random_seed: int = 42


def ar1_standard_series(length: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    rho = float(np.clip(rho, -0.9999, 0.9999))
    series = np.zeros(length, dtype=np.float64)
    innovations = rng.standard_normal(length)
    scale = math.sqrt(max(1.0 - rho * rho, 1e-12))
    series[0] = innovations[0]
    for idx in range(1, length):
        series[idx] = rho * series[idx - 1] + scale * innovations[idx]
    std = float(np.std(series))
    if std > 0:
        series = (series - np.mean(series)) / std
    return series


def smooth_vector(values: np.ndarray, span: int) -> np.ndarray:
    if span <= 1:
        return values.copy()
    span = max(1, int(span))
    if span % 2 == 0:
        span += 1
    kernel = np.ones(span, dtype=np.float64) / span
    return np.convolve(values, kernel, mode="same")


def build_frequency_profile(num_subcarriers: int, smoothing_span: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.standard_normal(num_subcarriers)
    smooth = smooth_vector(raw, smoothing_span)
    smooth -= np.min(smooth)
    smooth += 0.1
    profile = smooth / np.mean(smooth)
    return profile.astype(np.float64)


def build_power_envelope(config: BackboneConfig, rng: np.random.Generator) -> np.ndarray:
    rho_db = calibrate_power_db_rho(
        target_lag1=config.frame_power_lag1_corr,
        std_db=config.std_frame_power_db,
    )
    base = ar1_standard_series(config.num_frames, rho_db, rng)
    power_db = config.mean_frame_power_db + config.std_frame_power_db * base
    return np.power(10.0, power_db / 10.0).astype(np.float64)


def build_residual_field(config: BackboneConfig, rng: np.random.Generator) -> np.ndarray:
    residual = np.zeros((config.num_frames, config.num_subcarriers), dtype=np.float64)
    scale = math.sqrt(max(1.0 - config.residual_time_corr * config.residual_time_corr, 1e-12))
    burst_base = ar1_standard_series(config.num_frames, config.innovation_burst_lag1_corr, rng)
    event_envelope = build_event_envelope(
        num_frames=config.num_frames,
        rate_per_1k_frames=config.event_rate_per_1k_frames,
        duration_frames=config.event_duration_frames,
        strength=config.event_strength,
        rng=rng,
    )
    burst_strength = np.clip(
        (1.0 + config.innovation_burst_std * burst_base) * event_envelope,
        0.15,
        None,
    )
    residual[0] = smooth_vector(rng.standard_normal(config.num_subcarriers), config.residual_freq_smoothing_span)
    for t_idx in range(1, config.num_frames):
        innovation = smooth_vector(
            rng.standard_normal(config.num_subcarriers),
            config.residual_freq_smoothing_span,
        )
        innovation *= burst_strength[t_idx]
        residual[t_idx] = config.residual_time_corr * residual[t_idx - 1] + scale * innovation
    residual_std = float(np.std(residual))
    if residual_std > 0:
        residual = residual / residual_std
    return residual


def calibrate_wrapped_phase_sigma(target_std: float) -> float:
    target_std = float(max(target_std, 1e-6))
    rng = np.random.default_rng(12345)

    def wrapped_std(raw_sigma: float) -> float:
        samples = rng.normal(loc=0.0, scale=raw_sigma, size=200000)
        wrapped = np.angle(np.exp(1j * samples))
        return float(np.std(wrapped))

    lo = 1e-3
    hi = 10.0
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        current = wrapped_std(mid)
        if current < target_std:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrate_power_db_rho(target_lag1: float, std_db: float) -> float:
    target_lag1 = float(np.clip(target_lag1, 0.0, 0.9999))
    std_db = float(max(std_db, 1e-6))

    def resulting_lag1(candidate_rho: float) -> float:
        rng = np.random.default_rng(24680)
        base = ar1_standard_series(4096, candidate_rho, rng)
        power_db = std_db * base
        power_linear = np.power(10.0, power_db / 10.0)
        return safe_lag1_correlation(power_linear) or 0.0

    lo = 0.0
    hi = 0.9999
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        current = resulting_lag1(mid)
        if current < target_lag1:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_shape_strength(config: BackboneConfig, rng: np.random.Generator) -> np.ndarray:
    base = ar1_standard_series(config.num_frames, config.shape_variation_lag1_corr, rng)
    strength = 1.0 + config.shape_variation_std * base
    return np.clip(strength, 0.15, None)


def build_event_envelope(
    num_frames: int,
    rate_per_1k_frames: float,
    duration_frames: int,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    envelope = np.ones(num_frames, dtype=np.float64)
    if num_frames <= 0:
        return envelope

    duration_frames = max(1, int(duration_frames))
    expected_events = max(1, int(round(rate_per_1k_frames * num_frames / 1000.0)))
    ramp = max(2, duration_frames // 4)

    for _ in range(expected_events):
        start = int(rng.integers(0, max(1, num_frames - duration_frames + 1)))
        end = min(num_frames, start + duration_frames)
        span = end - start
        if span <= 0:
            continue

        active_strength = strength * float(rng.uniform(0.85, 1.15))
        if span <= 2 * ramp:
            local = np.full(span, active_strength, dtype=np.float64)
        else:
            up = np.linspace(1.0, active_strength, ramp, dtype=np.float64)
            flat = np.full(span - 2 * ramp, active_strength, dtype=np.float64)
            down = np.linspace(active_strength, 1.0, ramp, dtype=np.float64)
            local = np.concatenate([up, flat, down])

        envelope[start:end] *= local[:span]

    return np.clip(envelope, 0.5, 3.5)


def generate_synthetic_backbone(config: BackboneConfig) -> np.ndarray:
    rng = np.random.default_rng(config.random_seed)

    frame_power_linear = build_power_envelope(config, rng)
    frequency_profile = build_frequency_profile(
        config.num_subcarriers,
        config.frequency_smoothing_span,
        rng,
    )
    residual = build_residual_field(config, rng)
    residual = residual - np.mean(residual, axis=1, keepdims=True)
    shape_strength = build_shape_strength(config, rng)

    # The backbone is intentionally simple: a smooth mission-level power envelope,
    # a stable frequency profile, and a mild residual field that keeps neighboring
    # subcarriers and adjacent frames strongly correlated.
    amplitude = np.sqrt(frame_power_linear)[:, None] * np.sqrt(frequency_profile)[None, :]
    amplitude = amplitude * np.exp((config.residual_std * shape_strength[:, None]) * residual)

    # Re-lock each frame to the target frame-power envelope so the mission-level
    # power statistics are governed by the calibrated envelope rather than by the
    # local frequency-shape perturbations.
    current_frame_power = np.mean(amplitude ** 2, axis=1)
    frame_scale = np.sqrt(frame_power_linear / np.maximum(current_frame_power, 1e-12))
    amplitude = amplitude * frame_scale[:, None]

    initial_phase = rng.uniform(-np.pi, np.pi, size=config.num_subcarriers)
    calibrated_phase_sigma = calibrate_wrapped_phase_sigma(config.phase_increment_std)
    phase_increments = rng.normal(
        loc=0.0,
        scale=calibrated_phase_sigma,
        size=(config.num_frames, config.num_subcarriers),
    )
    phase = np.cumsum(phase_increments, axis=0)
    phase += initial_phase[None, :]

    return amplitude * np.exp(1j * phase)


def safe_lag1_correlation(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    x = values[:-1]
    y = values[1:]
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std == 0 or y_std == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def safe_percentiles(values: np.ndarray, quantiles: list[float]) -> dict[str, float | None]:
    if values.size == 0:
        return {f"p{int(q * 100):02d}": None for q in quantiles}
    return {f"p{int(q * 100):02d}": float(np.quantile(values, q)) for q in quantiles}


def cosine_similarity(a_vec: np.ndarray, b_vec: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
    if denom == 0.0:
        return None
    return float(np.dot(a_vec, b_vec) / denom)


def compute_backbone_stats(h_complex: np.ndarray) -> dict[str, Any]:
    power = np.abs(h_complex) ** 2
    frame_avg_linear = np.mean(power, axis=1)
    frame_avg_db = 10.0 * np.log10(frame_avg_linear + 1e-12)

    phase_diff = np.angle(h_complex[1:] * np.conj(h_complex[:-1]))
    phase_increment = phase_diff.ravel()

    magnitudes = np.abs(h_complex)
    temporal_cos = []
    for prev_mag, curr_mag in zip(magnitudes[:-1], magnitudes[1:]):
        value = cosine_similarity(prev_mag, curr_mag)
        if value is not None:
            temporal_cos.append(value)

    adjacent_cos = []
    for mag in magnitudes:
        value = cosine_similarity(mag[:-1], mag[1:])
        if value is not None:
            adjacent_cos.append(value)

    temporal_cos = np.asarray(temporal_cos, dtype=np.float64)
    adjacent_cos = np.asarray(adjacent_cos, dtype=np.float64)

    return {
        "frame_avg_linear": {
            "mean": float(np.mean(frame_avg_linear)),
            "std": float(np.std(frame_avg_linear)),
            "lag1_corr": safe_lag1_correlation(frame_avg_linear),
            **safe_percentiles(frame_avg_linear, [0.05, 0.5, 0.95]),
        },
        "frame_avg_db": {
            "mean": float(np.mean(frame_avg_db)),
            "std": float(np.std(frame_avg_db)),
            **safe_percentiles(frame_avg_db, [0.05, 0.5, 0.95]),
        },
        "phase_increment_rad": {
            "mean": float(np.mean(phase_increment)),
            "std": float(np.std(phase_increment)),
            "min": float(np.min(phase_increment)),
            "max": float(np.max(phase_increment)),
        },
        "temporal_magnitude_cosine_similarity": {
            "mean": float(np.mean(temporal_cos)) if temporal_cos.size else None,
            "std": float(np.std(temporal_cos)) if temporal_cos.size else None,
        },
        "adjacent_subcarrier_magnitude_cosine_similarity": {
            "mean": float(np.mean(adjacent_cos)) if adjacent_cos.size else None,
            "std": float(np.std(adjacent_cos)) if adjacent_cos.size else None,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a simple synthetic CSI backbone aligned to Route A baseline statistics."
    )
    parser.add_argument("--num-frames", type=int, default=BackboneConfig.num_frames)
    parser.add_argument("--num-subcarriers", type=int, default=BackboneConfig.num_subcarriers)
    parser.add_argument("--mean-frame-power-db", type=float, default=BackboneConfig.mean_frame_power_db)
    parser.add_argument("--std-frame-power-db", type=float, default=BackboneConfig.std_frame_power_db)
    parser.add_argument("--frame-power-lag1-corr", type=float, default=BackboneConfig.frame_power_lag1_corr)
    parser.add_argument("--phase-increment-std", type=float, default=BackboneConfig.phase_increment_std)
    parser.add_argument("--frequency-smoothing-span", type=int, default=BackboneConfig.frequency_smoothing_span)
    parser.add_argument(
        "--residual-freq-smoothing-span",
        type=int,
        default=BackboneConfig.residual_freq_smoothing_span,
    )
    parser.add_argument("--residual-std", type=float, default=BackboneConfig.residual_std)
    parser.add_argument("--residual-time-corr", type=float, default=BackboneConfig.residual_time_corr)
    parser.add_argument("--shape-variation-std", type=float, default=BackboneConfig.shape_variation_std)
    parser.add_argument(
        "--shape-variation-lag1-corr",
        type=float,
        default=BackboneConfig.shape_variation_lag1_corr,
    )
    parser.add_argument("--innovation-burst-std", type=float, default=BackboneConfig.innovation_burst_std)
    parser.add_argument(
        "--innovation-burst-lag1-corr",
        type=float,
        default=BackboneConfig.innovation_burst_lag1_corr,
    )
    parser.add_argument(
        "--event-rate-per-1k-frames",
        type=float,
        default=BackboneConfig.event_rate_per_1k_frames,
    )
    parser.add_argument(
        "--event-duration-frames",
        type=int,
        default=BackboneConfig.event_duration_frames,
    )
    parser.add_argument(
        "--event-strength",
        type=float,
        default=BackboneConfig.event_strength,
    )
    parser.add_argument("--random-seed", type=int, default=BackboneConfig.random_seed)
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=None,
        help="Optional NPZ output path for the generated complex backbone.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON summary path for generated statistics.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = BackboneConfig(
        num_frames=args.num_frames,
        num_subcarriers=args.num_subcarriers,
        mean_frame_power_db=args.mean_frame_power_db,
        std_frame_power_db=args.std_frame_power_db,
        frame_power_lag1_corr=args.frame_power_lag1_corr,
        phase_increment_std=args.phase_increment_std,
        frequency_smoothing_span=args.frequency_smoothing_span,
        residual_freq_smoothing_span=args.residual_freq_smoothing_span,
        residual_std=args.residual_std,
        residual_time_corr=args.residual_time_corr,
        shape_variation_std=args.shape_variation_std,
        shape_variation_lag1_corr=args.shape_variation_lag1_corr,
        innovation_burst_std=args.innovation_burst_std,
        innovation_burst_lag1_corr=args.innovation_burst_lag1_corr,
        event_rate_per_1k_frames=args.event_rate_per_1k_frames,
        event_duration_frames=args.event_duration_frames,
        event_strength=args.event_strength,
        random_seed=args.random_seed,
    )

    h_complex = generate_synthetic_backbone(config)
    stats = compute_backbone_stats(h_complex)
    output = {
        "config": asdict(config),
        "stats": stats,
    }

    if args.output_npz is not None:
        npz_path = args.output_npz.expanduser().resolve()
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, H=h_complex)
        print(f"[saved] {npz_path}")

    if args.output_json is not None:
        json_path = args.output_json.expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"[saved] {json_path}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
