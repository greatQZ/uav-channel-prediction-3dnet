import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
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
    burst_freq_smoothing_span: int = 5
    burst_time_corr: float = 0.55
    burst_event_rate_per_1k_frames: float = 22.0
    burst_event_duration_frames: int = 4
    burst_event_strength: float = 1.8
    burst_mix: float = 0.15
    gain_min: float = 0.0
    gain_max: float = 1.2
    gain_steps: int = 13
    random_seed: int = 123


def parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


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


def build_residual_component(
    num_frames: int,
    num_subcarriers: int,
    freq_smoothing_span: int,
    time_corr: float,
    event_rate_per_1k_frames: float,
    event_duration_frames: int,
    event_strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    residual = np.zeros((num_frames, num_subcarriers), dtype=np.float64)
    rho = float(np.clip(time_corr, 0.0, 0.9999))
    scale = math.sqrt(max(1.0 - rho * rho, 1e-12))
    event_envelope = build_event_envelope(
        num_frames=num_frames,
        rate_per_1k_frames=event_rate_per_1k_frames,
        duration_frames=event_duration_frames,
        strength=event_strength,
        rng=rng,
    )

    residual[0] = smooth_vector(
        rng.standard_normal(num_subcarriers),
        freq_smoothing_span,
    )
    for idx in range(1, num_frames):
        innovation = smooth_vector(
            rng.standard_normal(num_subcarriers),
            freq_smoothing_span,
        )
        innovation *= event_envelope[idx]
        residual[idx] = rho * residual[idx - 1] + scale * innovation

    residual -= np.mean(residual, axis=1, keepdims=True)
    std = float(np.std(residual))
    if std > 0:
        residual /= std
    return residual


def build_residual_template(
    num_frames: int,
    num_subcarriers: int,
    config: ResidualCalibratorConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    smooth_component = build_residual_component(
        num_frames=num_frames,
        num_subcarriers=num_subcarriers,
        freq_smoothing_span=config.residual_freq_smoothing_span,
        time_corr=config.residual_time_corr,
        event_rate_per_1k_frames=config.event_rate_per_1k_frames,
        event_duration_frames=config.event_duration_frames,
        event_strength=config.event_strength,
        rng=rng,
    )
    burst_component = build_residual_component(
        num_frames=num_frames,
        num_subcarriers=num_subcarriers,
        freq_smoothing_span=config.burst_freq_smoothing_span,
        time_corr=config.burst_time_corr,
        event_rate_per_1k_frames=config.burst_event_rate_per_1k_frames,
        event_duration_frames=config.burst_event_duration_frames,
        event_strength=config.burst_event_strength,
        rng=rng,
    )
    combined = smooth_component + config.burst_mix * burst_component
    combined -= np.mean(combined, axis=1, keepdims=True)
    std = float(np.std(combined))
    if std > 0:
        combined /= std
    return combined


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


def search_templates_and_gain(
    h_complex: np.ndarray,
    measurement_target: dict[str, Any],
    base_config: ResidualCalibratorConfig,
    residual_freq_smoothing_spans: list[int],
    residual_time_corrs: list[float],
    event_rates_per_1k_frames: list[float],
    event_duration_frames_list: list[int],
    event_strengths: list[float],
    burst_mixs: list[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    baseline_stats = compute_backbone_stats(h_complex)
    baseline_eval = evaluate(measurement_target, {"stats": baseline_stats})

    best_h = h_complex
    best_result: dict[str, Any] | None = None
    search_rows: list[dict[str, Any]] = []

    for residual_freq_smoothing_span in residual_freq_smoothing_spans:
        for residual_time_corr in residual_time_corrs:
            for event_rate_per_1k_frames in event_rates_per_1k_frames:
                    for event_duration_frames in event_duration_frames_list:
                        for event_strength in event_strengths:
                            for burst_mix in burst_mixs:
                                config = replace(
                                    base_config,
                                    residual_freq_smoothing_span=residual_freq_smoothing_span,
                                    residual_time_corr=residual_time_corr,
                                    event_rate_per_1k_frames=event_rate_per_1k_frames,
                                    event_duration_frames=event_duration_frames,
                                    event_strength=event_strength,
                                    burst_mix=burst_mix,
                                )
                                h_candidate, result = grid_search_gain(
                                    h_complex=h_complex,
                                    measurement_target=measurement_target,
                                    config=config,
                                )
                                score = result["best_score"]
                                search_rows.append(
                                    {
                                        "residual_freq_smoothing_span": residual_freq_smoothing_span,
                                        "residual_time_corr": residual_time_corr,
                                        "event_rate_per_1k_frames": event_rate_per_1k_frames,
                                        "event_duration_frames": event_duration_frames,
                                        "event_strength": event_strength,
                                        "burst_mix": burst_mix,
                                        "selected_gain": result["selected_gain"],
                                        "best_score": score,
                                    }
                                )
                                if score is not None and (
                                    best_result is None
                                    or best_result["best_score"] is None
                                    or score < best_result["best_score"]
                                ):
                                    best_h = h_candidate
                                    best_result = result

    if best_result is None:
        best_result = {
            "config": asdict(base_config),
            "selected_gain": 0.0,
            "baseline_score": baseline_eval["weighted_mean_relative_error"],
            "best_score": baseline_eval["weighted_mean_relative_error"],
            "score_improvement": 0.0,
            "baseline_stats": baseline_stats,
            "best_stats": baseline_stats,
            "evaluation": baseline_eval,
            "gain_sweep": [],
        }

    best_result = {
        **best_result,
        "template_search": sorted(
            search_rows,
            key=lambda row: float("inf") if row["best_score"] is None else row["best_score"],
        ),
    }
    return best_h, best_result


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
        "--burst-freq-smoothing-span",
        type=int,
        default=ResidualCalibratorConfig.burst_freq_smoothing_span,
    )
    parser.add_argument(
        "--burst-time-corr",
        type=float,
        default=ResidualCalibratorConfig.burst_time_corr,
    )
    parser.add_argument(
        "--burst-event-rate-per-1k-frames",
        type=float,
        default=ResidualCalibratorConfig.burst_event_rate_per_1k_frames,
    )
    parser.add_argument(
        "--burst-event-duration-frames",
        type=int,
        default=ResidualCalibratorConfig.burst_event_duration_frames,
    )
    parser.add_argument(
        "--burst-event-strength",
        type=float,
        default=ResidualCalibratorConfig.burst_event_strength,
    )
    parser.add_argument(
        "--burst-mix",
        type=float,
        default=ResidualCalibratorConfig.burst_mix,
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
    parser.add_argument(
        "--residual-freq-smoothing-span-list",
        type=str,
        default=None,
        help="Optional comma-separated candidate list for residual frequency smoothing spans.",
    )
    parser.add_argument(
        "--residual-time-corr-list",
        type=str,
        default=None,
        help="Optional comma-separated candidate list for residual time correlations.",
    )
    parser.add_argument(
        "--event-rate-per-1k-frames-list",
        type=str,
        default=None,
        help="Optional comma-separated candidate list for event rates per 1000 frames.",
    )
    parser.add_argument(
        "--event-duration-frames-list",
        type=str,
        default=None,
        help="Optional comma-separated candidate list for event durations.",
    )
    parser.add_argument(
        "--event-strength-list",
        type=str,
        default=None,
        help="Optional comma-separated candidate list for event strengths.",
    )
    parser.add_argument(
        "--burst-mix-list",
        type=str,
        default=None,
        help="Optional comma-separated candidate list for burst-mix weights.",
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
        burst_freq_smoothing_span=args.burst_freq_smoothing_span,
        burst_time_corr=args.burst_time_corr,
        burst_event_rate_per_1k_frames=args.burst_event_rate_per_1k_frames,
        burst_event_duration_frames=args.burst_event_duration_frames,
        burst_event_strength=args.burst_event_strength,
        burst_mix=args.burst_mix,
        gain_min=args.gain_min,
        gain_max=args.gain_max,
        gain_steps=args.gain_steps,
        random_seed=args.random_seed,
    )

    h_complex = load_backbone_npz(args.input_npz.expanduser().resolve())
    measurement_target = load_measurement_target(args.measurement_target.expanduser().resolve())

    residual_freq_smoothing_spans = (
        parse_int_list(args.residual_freq_smoothing_span_list)
        if args.residual_freq_smoothing_span_list
        else [config.residual_freq_smoothing_span]
    )
    residual_time_corrs = (
        parse_float_list(args.residual_time_corr_list)
        if args.residual_time_corr_list
        else [config.residual_time_corr]
    )
    event_rates_per_1k_frames = (
        parse_float_list(args.event_rate_per_1k_frames_list)
        if args.event_rate_per_1k_frames_list
        else [config.event_rate_per_1k_frames]
    )
    event_duration_frames_list = (
        parse_int_list(args.event_duration_frames_list)
        if args.event_duration_frames_list
        else [config.event_duration_frames]
    )
    event_strengths = (
        parse_float_list(args.event_strength_list)
        if args.event_strength_list
        else [config.event_strength]
    )
    burst_mixs = (
        parse_float_list(args.burst_mix_list)
        if args.burst_mix_list
        else [config.burst_mix]
    )

    calibrated_h, result = search_templates_and_gain(
        h_complex=h_complex,
        measurement_target=measurement_target,
        base_config=config,
        residual_freq_smoothing_spans=residual_freq_smoothing_spans,
        residual_time_corrs=residual_time_corrs,
        event_rates_per_1k_frames=event_rates_per_1k_frames,
        event_duration_frames_list=event_duration_frames_list,
        event_strengths=event_strengths,
        burst_mixs=burst_mixs,
    )

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
