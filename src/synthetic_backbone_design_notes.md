# Synthetic Backbone Design Notes

## Purpose

This note records why the first `synthetic_backbone_generator.py` was designed
the way it is, what it is intended to do, and what it is not intended to do.

The current generator is **not** meant to be a final synthetic CSI generator.
It is a **measurement-calibrated synthetic backbone prototype** whose job is to
produce a controllable complex sequence aligned with the cleanest BOLEK
`flight-only` measurement slice.

## What Was Implemented

The current backbone is organized into four layers.

### 1. Mission-level power envelope

The generator first builds a frame-level power envelope with:

- target frame-average power mean
- target frame-average power standard deviation
- target frame-average power lag-1 correlation

This is implemented as an AR(1)-style process in dB-domain with an explicit
numeric calibration step so that the resulting **linear-domain** lag-1 power
correlation better matches the measurement target.

### 2. Smooth frequency profile

A smooth frequency profile is generated and normalized so that neighboring
subcarriers remain correlated instead of behaving like fully independent noise.

This gives the synthetic sequence a stable subcarrier structure before any
mission-conditioned variation is applied.

### 3. Residual field

A time-correlated and frequency-smoothed residual field is then applied on top
of the backbone.

The residual field is used to introduce:

- local amplitude irregularity
- frame-to-frame variation
- subcarrier-wise shape perturbation

This is the first step toward measurement-calibrated residual behavior.

### 4. Phase random walk

The phase process is built from cumulative random increments.

Instead of directly assigning a Gaussian phase sigma, the implementation
numerically calibrates the wrapped phase-increment standard deviation so that
the generated sequence more faithfully matches the measured
`phase_increment_rad.std`.

## Additional Variability Layers

Two extra layers were introduced to make the synthetic sequence less idealized.

### Shape variation

The residual amplitude strength is allowed to vary slowly across time. This
approximates the fact that real missions do not maintain a perfectly stationary
perturbation level.

### Event / regime-switch envelope

A lightweight event envelope was added to briefly increase the residual burst
strength in a small number of short time windows.

This was introduced because earlier backbone versions were too smooth and
systematically underestimated:

- `temporal_magnitude_cosine_similarity.std`

The current event envelope is intentionally lightweight. Its purpose is not to
simulate a full physical state machine, but to introduce short periods of more
realistic local irregularity.

## Why This Design Was Chosen

The current design prioritizes:

- controllability
- interpretability
- alignment with measured statistics

At this stage, this is more important than building the most expressive
generator possible.

The present goal is to answer a workshop-style question:

> Can measurement-calibrated synthetic pretraining help realistic UAV channel
> prediction?

For that purpose, a simple generator whose components can be inspected and tuned
is more useful than a fully black-box approach.

## Why Not Start With GAN-First

The current prototype intentionally does **not** start with a pure GAN-first
generator.

The reasons are:

1. A black-box generator would make it harder to explain which measured
   properties are being preserved.
2. The current project stage is still about proving that
   measurement-calibrated synthetic pretraining has signal at all.
3. The workshop framing benefits from a clear and controllable synthetic
   baseline instead of a large generative framework.

## Literature Influence

The current implementation is **not** a direct reproduction of a specific
open-source project.

Instead, it is a project-specific research prototype guided by the following
literature directions.

### Directly influential papers

1. **GAN-Based Massive MIMO Channel Model Trained on Measured Data**
   - motivated the use of measurement-derived realism targets
   - also reinforced the need to avoid overclaiming generalization

2. **Enabling 6G Through Multi-Domain Channel Extrapolation: Opportunities and
   Challenges of Generative Artificial Intelligence**
   - motivated designing synthetic data as something that must serve a
     downstream prediction task, not only look statistically plausible

3. **Accurate Channel Prediction Based on Transformer: Making Mobility
   Negligible**
   - reinforced the importance of preserving time-structure rather than only
     static summary statistics

4. **Enhanced 6G Non-Terrestrial Network Link Performance using Deep Learning-Based
   Channel Estimation and Doppler Compensation Techniques**
   and
   **5G-NR Physical Layer-Based Solutions to Support High Mobility in 6G
   Non-Terrestrial Networks**
   - reinforced that high-mobility behavior cannot be reduced to average power
     alone

## What The Backbone Already Does Well

The current backbone can already fit several `P0` targets reasonably well:

- `frame_avg_db.mean`
- `frame_avg_db.std`
- `phase_increment_rad.std`
- `adjacent_subcarrier_magnitude_cosine_similarity.mean`
- `adjacent_subcarrier_magnitude_cosine_similarity.std`

## What It Still Does Poorly

The largest remaining mismatch is:

- `temporal_magnitude_cosine_similarity.std`

This indicates that the current backbone is still smoother than the real
mission in the time domain, even after adding lightweight event bursts.

## Interpretation

The current backbone should therefore be treated as:

- a usable first synthetic backbone
- not a finished synthetic generator
- the base layer for the next step: a residual calibrator

## Next Step

The next implementation step should build a residual calibration stage on top
of this backbone so that mission-local irregularity can be added without losing
the already-fitted mission-level statistics.
