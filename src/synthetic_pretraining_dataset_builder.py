import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from synthetic_backbone_generator import BackboneConfig, compute_backbone_stats, generate_synthetic_backbone
from synthetic_calibration_evaluator import load_measurement_target
from synthetic_residual_calibrator import ResidualCalibratorConfig, search_templates_and_gain


def complex_to_representation(h_complex: np.ndarray, representation: str) -> np.ndarray:
    if representation == "magnitude":
        return np.abs(h_complex).astype(np.float32)
    if representation == "complex_ri":
        stacked = np.stack([h_complex.real, h_complex.imag], axis=-1)
        return stacked.astype(np.float32)
    raise ValueError(f"Unsupported representation: {representation}")


def build_windows(
    h_complex: np.ndarray,
    lookback: int,
    horizon: int,
    representation: str,
) -> tuple[np.ndarray, np.ndarray]:
    if lookback <= 0 or horizon <= 0:
        raise ValueError("lookback and horizon must be positive integers")

    max_start = h_complex.shape[0] - lookback - horizon + 1
    if max_start <= 0:
        raise ValueError("Sequence too short for the requested lookback/horizon")

    x_rows = []
    y_rows = []
    rep = complex_to_representation(h_complex, representation)
    target_mag = np.abs(h_complex).astype(np.float32)

    for start in range(max_start):
        x_rows.append(rep[start : start + lookback])
        target_index = start + lookback + horizon - 1
        y_rows.append(target_mag[target_index])

    return np.asarray(x_rows), np.asarray(y_rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a first synthetic pretraining dataset from the backbone + residual pipeline."
    )
    parser.add_argument(
        "--measurement-target",
        type=Path,
        required=True,
        help="Path to a measurement_stats_extractor JSON target file.",
    )
    parser.add_argument("--lookback", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument(
        "--representation",
        type=str,
        default="magnitude",
        choices=["magnitude", "complex_ri"],
        help="Feature representation stored in X.",
    )
    parser.add_argument("--num-frames", type=int, default=256)
    parser.add_argument("--num-subcarriers", type=int, default=128)
    parser.add_argument("--backbone-seed", type=int, default=42)
    parser.add_argument("--residual-seed", type=int, default=123)
    parser.add_argument(
        "--output-npz",
        type=Path,
        required=True,
        help="Output NPZ path containing windowed samples.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output JSON manifest path.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    measurement_target = load_measurement_target(args.measurement_target.expanduser().resolve())

    backbone_config = BackboneConfig(
        num_frames=args.num_frames,
        num_subcarriers=args.num_subcarriers,
        random_seed=args.backbone_seed,
    )
    h_backbone = generate_synthetic_backbone(backbone_config)

    residual_config = ResidualCalibratorConfig(
        random_seed=args.residual_seed,
    )
    h_calibrated, calibration_result = search_templates_and_gain(
        h_complex=h_backbone,
        measurement_target=measurement_target,
        base_config=residual_config,
        residual_freq_smoothing_spans=[residual_config.residual_freq_smoothing_span],
        residual_time_corrs=[residual_config.residual_time_corr],
        event_rates_per_1k_frames=[residual_config.event_rate_per_1k_frames],
        event_duration_frames_list=[residual_config.event_duration_frames],
        event_strengths=[residual_config.event_strength],
        burst_mixs=[residual_config.burst_mix],
    )

    x_data, y_data = build_windows(
        h_complex=h_calibrated,
        lookback=args.lookback,
        horizon=args.horizon,
        representation=args.representation,
    )

    output_npz = args.output_npz.expanduser().resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        X=x_data,
        y=y_data,
        H_calibrated=h_calibrated,
    )

    manifest = {
        "dataset_role": "synthetic_pretraining",
        "representation": args.representation,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "sample_count": int(x_data.shape[0]),
        "num_frames": int(args.num_frames),
        "num_subcarriers": int(args.num_subcarriers),
        "measurement_target_path": str(args.measurement_target.expanduser().resolve()),
        "backbone_config": asdict(backbone_config),
        "residual_config": asdict(residual_config),
        "backbone_stats": compute_backbone_stats(h_backbone),
        "calibration_result": calibration_result,
        "window_contract": {
            "input": f"{args.lookback} historical steps",
            "target": f"magnitude at +{args.horizon} step",
            "target_style": "many-to-one",
        },
    }

    output_json = args.output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[saved] {output_npz}")
    print(f"[saved] {output_json}")


if __name__ == "__main__":
    main()
