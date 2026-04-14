# Synthetic Calibration Targets (v0.1)

## Purpose

This note defines the first fitting targets for the initial
measurement-calibrated synthetic prototype.

The goal is not to build a fully general channel generator yet.
The goal is to create a small, controllable synthetic backbone that is
statistically aligned with the cleanest BOLEK `flight-only` measurements and
can be used for the first pretraining experiments.

## Primary Calibration Source

- Mission family: `EDAZ_0917`
- Main baseline:
  - `channel_estimates_20250917_140636_bolek_40m_5mps.txt`
  - interpretation: peak-speed around `5 m/s`
  - window type: `flight-only`
- Secondary comparison baseline:
  - `channel_estimates_20250917_143517_bolek_50m_5mps.txt`
  - interpretation: peak-speed around `5 m/s`
  - window type: `flight-only`

## Why The 40m Flight-Only Slice Is The First Calibration Baseline

Compared with the `50m` flight-only slice, the `40m` flight-only slice is:

- higher-power on average
- less bursty
- more temporally stable
- more frequency-consistent

This makes it the best first target for fitting a residual-style synthetic
prototype.

## Flight-Only Reference Targets

### Baseline A: BOLEK 40m peak-5mps

- `accepted_frames`: `47959`
- `frame_avg_db.mean`: `44.38`
- `frame_avg_db.std`: `3.75`
- `frame_avg_linear.lag1_corr`: `0.97515`
- `phase_increment.std`: `1.81323`
- `temporal_magnitude_cosine_similarity.mean`: `0.99718`
- `temporal_magnitude_cosine_similarity.std`: `0.00610`
- `adjacent_subcarrier_magnitude_cosine_similarity.mean`: `0.99805`
- `adjacent_subcarrier_magnitude_cosine_similarity.std`: `0.00064`

### Baseline B: BOLEK 50m peak-5mps

- `accepted_frames`: `40648`
- `frame_avg_db.mean`: `42.69`
- `frame_avg_db.std`: `6.64`
- `frame_avg_linear.lag1_corr`: `0.97537`
- `phase_increment.std`: `1.81267`
- `temporal_magnitude_cosine_similarity.mean`: `0.99142`
- `temporal_magnitude_cosine_similarity.std`: `0.02943`
- `adjacent_subcarrier_magnitude_cosine_similarity.mean`: `0.99556`
- `adjacent_subcarrier_magnitude_cosine_similarity.std`: `0.01424`

## What The First Synthetic Prototype Must Match

### P0: Must Match

These are the minimum fitting targets for the first prototype:

1. Frame-average power level
2. Frame-average power variability
3. Frame-average power temporal smoothness
4. Phase-increment spread
5. Magnitude temporal consistency
6. Adjacent-subcarrier magnitude consistency

In practice, the first synthetic prototype should reproduce:

- `frame_avg_db.mean`
- `frame_avg_db.std`
- `frame_avg_linear.lag1_corr`
- `phase_increment_rad.std`
- `temporal_magnitude_cosine_similarity.mean/std`
- `adjacent_subcarrier_magnitude_cosine_similarity.mean/std`

### P1: Good To Add Next

- subcarrier-wise power profile shape
- stronger peak-subcarrier structure
- mission-duration statistics
- regime-conditioned slices inside one mission

### P2: Defer For Later

- full multi-domain extrapolation behavior
- explicit NLOS/LOS state labeling
- multi-mission joint fitting
- mission transfer to `EDAZ_0710`

## Recommended Fitting Strategy

### Stage 1: Fit To Baseline A Only

Fit the first prototype to:

- `40m` peak-`5m/s`
- `flight-only`

This stage should optimize stability and interpretability, not complexity.

### Stage 2: Validate Against Baseline B

Without changing the full design, test whether the fitted prototype still
looks reasonable against:

- `50m` peak-`5m/s`
- `flight-only`

This is the first check against overfitting to one clean mission.

### Stage 3: Move To Higher-Mobility Validation

Only after Stage 1 and Stage 2 are stable:

- `50m` peak-`10m/s` inconstant
- `40m` peak-`10m/s` with pause
- mixed peak-`10m/s` stress mission

## First Prototype Design Constraint

The first prototype should be:

- residual-style
- controllable
- streamable
- easy to compare against real targets

It should **not** start as:

- a pure GAN-first generator
- a full end-to-end black-box model
- a large multi-mission framework

## Practical Interpretation For Implementation

The first generator only needs two layers:

1. A simple synthetic backbone that produces a smooth time-varying CSI-like
   sequence
2. A calibration layer that adjusts the backbone so the resulting sequence
   matches the `P0` targets above

## Immediate Next Step

The next implementation step should define:

- backbone input parameters
- generated sequence length
- residual calibration parameters
- fitting loss over the `P0` target set
