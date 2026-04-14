import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_TARGET_PATHS = {
    "frame_avg_db.mean": ("power_stats", "frame_avg_db", "mean"),
    "frame_avg_db.std": ("power_stats", "frame_avg_db", "std"),
    "frame_avg_linear.lag1_corr": ("power_stats", "frame_avg_linear", "lag1_corr"),
    "phase_increment_rad.std": ("calibration_targets", "phase_increment_rad", "std"),
    "temporal_magnitude_cosine_similarity.mean": (
        "calibration_targets",
        "temporal_magnitude_cosine_similarity",
        "mean",
    ),
    "temporal_magnitude_cosine_similarity.std": (
        "calibration_targets",
        "temporal_magnitude_cosine_similarity",
        "std",
    ),
    "adjacent_subcarrier_magnitude_cosine_similarity.mean": (
        "calibration_targets",
        "adjacent_subcarrier_magnitude_cosine_similarity",
        "mean",
    ),
    "adjacent_subcarrier_magnitude_cosine_similarity.std": (
        "calibration_targets",
        "adjacent_subcarrier_magnitude_cosine_similarity",
        "std",
    ),
}


DEFAULT_SYNTH_PATHS = {
    "frame_avg_db.mean": ("stats", "frame_avg_db", "mean"),
    "frame_avg_db.std": ("stats", "frame_avg_db", "std"),
    "frame_avg_linear.lag1_corr": ("stats", "frame_avg_linear", "lag1_corr"),
    "phase_increment_rad.std": ("stats", "phase_increment_rad", "std"),
    "temporal_magnitude_cosine_similarity.mean": (
        "stats",
        "temporal_magnitude_cosine_similarity",
        "mean",
    ),
    "temporal_magnitude_cosine_similarity.std": (
        "stats",
        "temporal_magnitude_cosine_similarity",
        "std",
    ),
    "adjacent_subcarrier_magnitude_cosine_similarity.mean": (
        "stats",
        "adjacent_subcarrier_magnitude_cosine_similarity",
        "mean",
    ),
    "adjacent_subcarrier_magnitude_cosine_similarity.std": (
        "stats",
        "adjacent_subcarrier_magnitude_cosine_similarity",
        "std",
    ),
}


DEFAULT_WEIGHTS = {
    "frame_avg_db.mean": 1.0,
    "frame_avg_db.std": 1.0,
    "frame_avg_linear.lag1_corr": 1.2,
    "phase_increment_rad.std": 1.0,
    "temporal_magnitude_cosine_similarity.mean": 1.2,
    "temporal_magnitude_cosine_similarity.std": 0.8,
    "adjacent_subcarrier_magnitude_cosine_similarity.mean": 1.2,
    "adjacent_subcarrier_magnitude_cosine_similarity.std": 0.8,
}


def get_nested(data: dict[str, Any], path: tuple[str, ...]) -> float | None:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def relative_error(target: float, predicted: float, floor: float = 1e-6) -> float:
    scale = max(abs(target), floor)
    return abs(predicted - target) / scale


def load_measurement_target(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "files" not in obj or not obj["files"]:
        raise ValueError(f"Measurement target file has no files payload: {path}")
    return obj["files"][0]


def load_synthetic_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(
    measurement_target: dict[str, Any],
    synthetic_summary: dict[str, Any],
) -> dict[str, Any]:
    metrics = []
    weighted_sum = 0.0
    weight_total = 0.0

    for metric_name, target_path in DEFAULT_TARGET_PATHS.items():
        synth_path = DEFAULT_SYNTH_PATHS[metric_name]
        target_value = get_nested(measurement_target, target_path)
        predicted_value = get_nested(synthetic_summary, synth_path)
        if target_value is None or predicted_value is None:
            continue

        weight = DEFAULT_WEIGHTS.get(metric_name, 1.0)
        rel_err = relative_error(float(target_value), float(predicted_value))
        weighted_sum += weight * rel_err
        weight_total += weight
        metrics.append(
            {
                "metric": metric_name,
                "target": float(target_value),
                "predicted": float(predicted_value),
                "absolute_error": abs(float(predicted_value) - float(target_value)),
                "relative_error": rel_err,
                "weight": weight,
            }
        )

    weighted_mean_relative_error = weighted_sum / weight_total if weight_total > 0 else None
    return {
        "weighted_mean_relative_error": weighted_mean_relative_error,
        "metric_count": len(metrics),
        "metrics": metrics,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare synthetic backbone summary stats against measurement-derived targets."
    )
    parser.add_argument(
        "--measurement-target",
        type=Path,
        required=True,
        help="Path to a measurement_stats_extractor JSON output.",
    )
    parser.add_argument(
        "--synthetic-summary",
        type=Path,
        required=True,
        help="Path to a synthetic_backbone_generator JSON output.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save evaluation results.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    measurement_target = load_measurement_target(args.measurement_target.expanduser().resolve())
    synthetic_summary = load_synthetic_summary(args.synthetic_summary.expanduser().resolve())
    result = evaluate(measurement_target, synthetic_summary)

    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[saved] {output_path}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
