# Current `src/` Keep-Set

This directory is intentionally reduced to the scripts that support the current
Route A measurement-first workflow.

## Kept Scripts

- `measurement_stats_extractor.py`
  - Streaming extractor for calibration-target statistics from large
    `channel_estimates_*.txt` logs.
- `CPE_analyse.py`
  - Existing mission-alignment and phase/CPE analysis utility.
- `raw_vs_compensated_experiment.py`
  - Experiment entry point for comparing raw and compensated variants.
- `data_analyse_radialvelocity_3D_bolek_new.py`
  - BOLEK-focused geometry / radial-velocity analysis.
- `data_analyse_time_speed_power_heatmap_speed_power_bolek.py`
  - BOLEK time-speed-power analysis for the standard-rate slices.
- `data_analyse_time_speed_power_heatmap_speed_power_bolek_5ms.py`
  - BOLEK time-speed-power analysis for the 5 ms slices.

## Archived Local Files

Legacy training, replay, demo, conversion, and one-off analysis scripts were
moved to:

- `archive/local_src_cleanup_20260413/src_legacy/`

That archive is local-only and intentionally excluded from GitHub tracking.
