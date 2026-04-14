import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from synthetic_backbone_generator import compute_backbone_stats, smooth_vector
from synthetic_calibration_evaluator import evaluate, load_measurement_target


@dataclass
class ResidualCalibratorConfig:
    residual_freq_smoothing_span: int = 15
    residual_time_corr: float = 0.93
    event_rate_per_1k_frames: float = 10.0
    event_duration_frames: int = 12
    event_strength: float = 1.2
    gain_min: float = 0.0
    gain_max: float = 1.2
    gain_steps: int = 13
    random_seed: int = 123


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

    return np.clip(envelope, 0.5, 3.0)


def build_residual_template(
    num_frames: int,
    num_subcarriers: int,
    config: ResidualCalibratorConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    residual = np.zeros((num_frames, num_subcarriers), dtype=np.float64)
    rho = float(np.clip(config.residual_time_corr, 0.0, 0.9999))
    scale = math.sqrt(max(1.0 - rho * rho, 1e-12))
    event_envelope = build_event_envelope(
        num_frames=num_frames,
        rate_per_1k_frames=config.event_rate_per_1k_frames,
        duration_frames=config.event_duration_frames,
        strength=config.event_strength,
        rng=rng,
    )

    residual[0] = smooth_vector(
        rng.standard_normal(num_subcarriers),
        config.residual_freq_smoothing_span,
    )
    for idx in range(1, num_frames):
        innovation = smooth_vector(
            rng.standard_normal(num_subcarriers),
            config.residual_freq_smoothing_span,
        )
        innovation *= event_envelope[idx]
        residual[idx] = rho * residual[idx - 1] + scale * innovation

    residual -= np.mean(residual, axis=1, keepdims=True)
    std = float(np.std(residual))
    if std > 0:
        residual /= std
    return residual


def relock_frame_power(amplitude: np.ndarray, target_frame_power: np.ndarray) -> np.ndarray:
    current_frame_power = np.mean(amplitude ** 2, axis=1)
    scale = np.sqrt(target_frame_power / np.maximum(current_frame_power, 1e-12))
    return amplitude * scale[:, None]


def apply_residual_calibration(
    h_complex: np.ndarray,
    residual_template: np.ndarray,
    gain: float,
) -> np.ndarray:
    magnitude = np.abs(h_complex)
    phase = np.angle(h_complex)
    target_frame_power = np.mean(magnitude ** 2, axis=1)

    calibrated_magnitude = magnitude * np.exp(gain * residual_template)
    calibrated_magnitude = relock_frame_power(calibrated_magnitude, target_frame_power)
    return calibrated_magnitude * np.exp(1j * phase)


def grid_search_gain(
    h_complex: np.ndarray,
    measurement_target: dict[str, Any],
    config: ResidualCalibratorConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(config.random_seed)
    residual_template = build_residual_template(
        num_frames=h_complex.shape[0],
        num_subcarriers=h_complex.shape[1],
        config=config,
        rng=rng,
    )

    baseline_stats = compute_backbone_stats(h_complex)
    baseline_eval = evaluate(measurement_target, {"stats": baseline_stats})

    best_h = h_complex
    best_stats = baseline_stats
    best_eval = baseline_eval
    best_gain = 0.0

    gains = np.linspace(config.gain_min, config.gain_max, config.gain_steps)
    sweep_rows: list[dict[str, Any]] = []

    for gain in gains:
        h_candidate = apply_residual_calibration(h_complex, residual_template, float(gain))
        stats = compute_backbone_stats(h_candidate)
        evaluation = evaluate(measurement_target, {"stats": stats})
        score = evaluation["weighted_mean_relative_error"]
        sweep_rows.append(
            {
                "gain": float(gain),
                "score": score,
                "temporal_std": stats["temporal_magnitude_cosine_similarity"]["std"],
                "adjacent_std": stats["adjacent_subcarrier_magnitude_cosine_similarity"]["std"],
            }
        )
        if score is not None and (
            best_eval["weighted_mean_relative_error"] is None
            or score < best_eval["weighted_mean_relative_error"]
        ):
            best_h = h_candidate
            best_stats = stats
            best_eval = evaluation
            best_gain = float(gain)

    result = {
        "config": asdict(config),
        "selected_gain": best_gain,
        "baseline_score": baseline_eval["weighted_mean_relative_error"],
        "best_score": best_eval["weighted_mean_relative_error"],
        "score_improvement": (
            None
            if baseline_eval["weighted_mean_relative_error"] is None
            or best_eval["weighted_mean_relative_error"] is None
            else baseline_eval["weighted_mean_relative_error"] - best_eval["weighted_mean_relative_error"]
        ),
        "baseline_stats": baseline_stats,
        "best_stats": best_stats,
        "evaluation": best_eval,
        "gain_sweep": sweep_rows,
    }
    return best_h, result


def load_backbone_npz(path: Path) -> np.ndarray:
    obj = np.load(path)
    if "H" not in obj:
        raise ValueError(f"NPZ file does not contain 'H': {path}")
    return obj["H"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply a first residual-style calibration layer to a synthetic backbone."
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        required=True,
        help="Path to the synthetic backbone NPZ containing array 'H'.",
    )
    parser.add_argument(
        "--measurement-target",
        type=Path,
        required=True,
        help="Path to a measurement_stats_extractor JSON output.",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=None,
        help="Optional output NPZ for the calibrated complex sequence.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON summary of calibration results.",
    )
    parser.add_argument(
        "--residual-freq-smoothing-span",
        type=int,
        default=ResidualCalibratorConfig.residual_freq_smoothing_span,
    )
    parser.add_argument(
        "--residual-time-corr",
        type=float,
        default=ResidualCalibratorConfig.residual_time_corr,
    )
    parser.add_argument(
        "--event-rate-per-1k-frames",
        type=float,
        default=ResidualCalibratorConfig.event_rate_per_1k_frames,
    )
    parser.add_argument(
        "--event-duration-frames",
        type=int,
        default=ResidualCalibratorConfig.event_duration_frames,
    )
    parser.add_argument(
        "--event-strength",
        type=float,
        default=ResidualCalibratorConfig.event_strength,
    )
    parser.add_argument(
        "--gain-min",
        type=float,
        default=ResidualCalibratorConfig.gain_min,
    )
    parser.add_argument(
        "--gain-max",
        type=float,
        default=ResidualCalibratorConfig.gain_max,
    )
    parser.add_argument(
        "--gain-steps",
        type=int,
        default=ResidualCalibratorConfig.gain_steps,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=ResidualCalibratorConfig.random_seed,
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = ResidualCalibratorConfig(
        residual_freq_smoothing_span=args.residual_freq_smoothing_span,
        residual_time_corr=args.residual_time_corr,
        event_rate_per_1k_frames=args.event_rate_per_1k_frames,
        event_duration_frames=args.event_duration_frames,
        event_strength=args.event_strength,
        gain_min=args.gain_min,
        gain_max=args.gain_max,
        gain_steps=args.gain_steps,
        random_seed=args.random_seed,
    )

    h_complex = load_backbone_npz(args.input_npz.expanduser().resolve())
    measurement_target = load_measurement_target(args.measurement_target.expanduser().resolve())
    calibrated_h, result = grid_search_gain(h_complex, measurement_target, config)

    if args.output_npz is not None:
        npz_path = args.output_npz.expanduser().resolve()
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, H=calibrated_h)
        print(f"[saved] {npz_path}")

    if args.output_json is not None:
        json_path = args.output_json.expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[saved] {json_path}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
