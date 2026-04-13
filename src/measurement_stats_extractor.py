import argparse
import json
import math
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


HEADER_PATTERN = re.compile(r"SRS Frame (\d+),.* Real: ([\d\.eE+-]+)")
SC_PATTERN = re.compile(r"Sc (\d+): Re = (-?\d+), Im = (-?\d+)")
DEFAULT_NUM_SUBCARRIERS = 1024


def infer_unix_unit(ts_value: float) -> str:
    abs_ts = abs(float(ts_value))
    if abs_ts > 1e17:
        return "ns"
    if abs_ts > 1e14:
        return "us"
    if abs_ts > 1e11:
        return "ms"
    return "s"


def to_utc_iso(ts_value: float) -> str:
    unit = infer_unix_unit(ts_value)
    scale_map = {
        "s": 1.0,
        "ms": 1e-3,
        "us": 1e-6,
        "ns": 1e-9,
    }
    ts_seconds = float(ts_value) * scale_map[unit]
    return datetime.fromtimestamp(ts_seconds, tz=timezone.utc).isoformat()


@dataclass
class RunningMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_value: float = math.inf
    max_value: float = -math.inf

    def update(self, value: float) -> None:
        value = float(value)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

    def update_batch(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        for value in np.asarray(values).ravel():
            self.update(float(value))

    def as_dict(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        variance = self.m2 / self.count
        return {
            "count": self.count,
            "mean": self.mean,
            "std": float(math.sqrt(max(variance, 0.0))),
            "min": self.min_value,
            "max": self.max_value,
        }


def safe_percentiles(values: np.ndarray, quantiles: list[float]) -> dict[str, float | None]:
    if values.size == 0:
        return {f"p{int(q * 100):02d}": None for q in quantiles}
    return {
        f"p{int(q * 100):02d}": float(np.quantile(values, q))
        for q in quantiles
    }


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


def summarize_top_subcarriers(mean_power: np.ndarray, seen_counts: np.ndarray, top_k: int) -> list[dict[str, Any]]:
    active_idx = np.where(seen_counts > 0)[0]
    if active_idx.size == 0:
        return []
    order = active_idx[np.argsort(mean_power[active_idx])[::-1]]
    top_entries = []
    for idx in order[:top_k]:
        top_entries.append(
            {
                "subcarrier_index": int(idx),
                "mean_power_linear": float(mean_power[idx]),
                "mean_power_db": float(10.0 * np.log10(mean_power[idx] + 1e-12)),
                "seen_in_frames": int(seen_counts[idx]),
            }
        )
    return top_entries


def process_channel_file(
    file_path: Path,
    num_subcarriers: int,
    max_frames: int | None,
    progress_every: int,
    top_k: int,
) -> dict[str, Any]:
    power_sum = np.zeros(num_subcarriers, dtype=np.float64)
    power_sq_sum = np.zeros(num_subcarriers, dtype=np.float64)
    seen_counts = np.zeros(num_subcarriers, dtype=np.int64)

    frame_avg_power_series: list[float] = []
    phase_increment_stats = RunningMoments()
    temporal_cosine_stats = RunningMoments()
    adjacent_cosine_stats = RunningMoments()

    first_timestamp_raw = None
    last_timestamp_raw = None
    accepted_frames = 0
    dropped_incomplete_frames = 0
    total_headers_seen = 0

    prev_frame = None
    current_frame = np.zeros(num_subcarriers, dtype=np.complex128)
    current_timestamp_raw = None
    current_frame_index = None
    sc_count = 0

    def finalize_frame() -> bool:
        nonlocal accepted_frames, dropped_incomplete_frames
        nonlocal first_timestamp_raw, last_timestamp_raw, prev_frame
        nonlocal current_frame, current_timestamp_raw, sc_count

        if current_timestamp_raw is None:
            return False
        if sc_count != num_subcarriers:
            dropped_incomplete_frames += 1
            return False

        frame_power = np.abs(current_frame) ** 2
        frame_avg_power = float(np.mean(frame_power))
        frame_avg_power_series.append(frame_avg_power)

        power_sum[:] += frame_power
        power_sq_sum[:] += frame_power ** 2
        seen_counts[:] += 1

        if first_timestamp_raw is None:
            first_timestamp_raw = current_timestamp_raw
        last_timestamp_raw = current_timestamp_raw

        if prev_frame is not None:
            valid = (np.abs(prev_frame) > 0) & (np.abs(current_frame) > 0)
            if np.any(valid):
                phase_diff = np.angle(current_frame[valid] * np.conj(prev_frame[valid]))
                phase_increment_stats.update_batch(phase_diff)

                prev_mag = np.abs(prev_frame[valid])
                curr_mag = np.abs(current_frame[valid])
                denom = np.linalg.norm(prev_mag) * np.linalg.norm(curr_mag)
                if denom > 0:
                    temporal_cosine_stats.update(float(np.dot(prev_mag, curr_mag) / denom))

            curr_mag_all = np.abs(current_frame)
            left = curr_mag_all[:-1]
            right = curr_mag_all[1:]
            denom_adj = np.linalg.norm(left) * np.linalg.norm(right)
            if denom_adj > 0:
                adjacent_cosine_stats.update(float(np.dot(left, right) / denom_adj))

        prev_frame = current_frame.copy()
        accepted_frames += 1
        if progress_every > 0 and accepted_frames % progress_every == 0:
            print(f"[progress] {file_path.name}: accepted_frames={accepted_frames}")
        return True

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            header_match = HEADER_PATTERN.search(line)
            if header_match:
                if finalize_frame() and max_frames is not None and accepted_frames >= max_frames:
                    break

                total_headers_seen += 1
                current_frame.fill(0.0)
                current_frame_index = int(header_match.group(1))
                current_timestamp_raw = float(header_match.group(2))
                sc_count = 0
                continue

            if current_timestamp_raw is None:
                continue

            sc_match = SC_PATTERN.search(line)
            if not sc_match:
                continue

            sc_idx = int(sc_match.group(1))
            if sc_idx < 0 or sc_idx >= num_subcarriers:
                continue

            re_val = int(sc_match.group(2))
            im_val = int(sc_match.group(3))
            current_frame[sc_idx] = complex(re_val, im_val)
            sc_count += 1

        if max_frames is None or accepted_frames < max_frames:
            finalize_frame()

    frame_avg_power_array = np.asarray(frame_avg_power_series, dtype=np.float64)
    active_mask = seen_counts > 0
    mean_power_per_sc = np.divide(
        power_sum,
        seen_counts,
        out=np.zeros_like(power_sum),
        where=active_mask,
    )

    summary = {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "frames": {
            "accepted": accepted_frames,
            "dropped_incomplete": dropped_incomplete_frames,
            "headers_seen": total_headers_seen,
            "max_frames_limit": max_frames,
        },
        "time_range_utc": {
            "first": to_utc_iso(first_timestamp_raw) if first_timestamp_raw is not None else None,
            "last": to_utc_iso(last_timestamp_raw) if last_timestamp_raw is not None else None,
        },
        "subcarriers": {
            "configured_total": num_subcarriers,
            "active_count": int(np.sum(active_mask)),
            "top_mean_power": summarize_top_subcarriers(mean_power_per_sc, seen_counts, top_k=top_k),
        },
        "power_stats": {
            "frame_avg_linear": {
                "mean": float(np.mean(frame_avg_power_array)) if frame_avg_power_array.size else None,
                "std": float(np.std(frame_avg_power_array)) if frame_avg_power_array.size else None,
                "min": float(np.min(frame_avg_power_array)) if frame_avg_power_array.size else None,
                "max": float(np.max(frame_avg_power_array)) if frame_avg_power_array.size else None,
                "lag1_corr": safe_lag1_correlation(frame_avg_power_array),
                **safe_percentiles(frame_avg_power_array, [0.05, 0.5, 0.95]),
            },
            "frame_avg_db": (
                {
                    "mean": float(np.mean(10.0 * np.log10(frame_avg_power_array + 1e-12))),
                    "std": float(np.std(10.0 * np.log10(frame_avg_power_array + 1e-12))),
                    **safe_percentiles(10.0 * np.log10(frame_avg_power_array + 1e-12), [0.05, 0.5, 0.95]),
                }
                if frame_avg_power_array.size
                else {
                    "mean": None,
                    "std": None,
                    "p05": None,
                    "p50": None,
                    "p95": None,
                }
            ),
        },
        "calibration_targets": {
            "phase_increment_rad": phase_increment_stats.as_dict(),
            "temporal_magnitude_cosine_similarity": temporal_cosine_stats.as_dict(),
            "adjacent_subcarrier_magnitude_cosine_similarity": adjacent_cosine_stats.as_dict(),
        },
    }
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract streaming measurement statistics from large channel_estimates logs."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more channel_estimates text files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--num-subcarriers",
        type=int,
        default=DEFAULT_NUM_SUBCARRIERS,
        help="Expected number of subcarriers per frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for smoke tests on very large files.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print a progress message every N accepted frames.",
    )
    parser.add_argument(
        "--top-k-subcarriers",
        type=int,
        default=10,
        help="How many strongest subcarriers to include in the summary.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    summaries = []
    for input_path in args.inputs:
        file_path = Path(input_path).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        print(f"[start] extracting measurement stats from {file_path}")
        summary = process_channel_file(
            file_path=file_path,
            num_subcarriers=args.num_subcarriers,
            max_frames=args.max_frames,
            progress_every=args.progress_every,
            top_k=args.top_k_subcarriers,
        )
        summaries.append(summary)
        print(
            f"[done] {file_path.name}: accepted_frames={summary['frames']['accepted']}, "
            f"active_subcarriers={summary['subcarriers']['active_count']}"
        )

    output = {"files": summaries}
    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"[saved] {output_path}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
