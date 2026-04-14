# Synthetic Residual Calibrator Design Notes

## Purpose

This note explains why the first `synthetic_residual_calibrator.py` was added on
top of the synthetic backbone and what role it plays in the current Route A
prototype.

The backbone already matches several mission-level statistics reasonably well,
but it remains too smooth in the time domain. In particular, it still
underestimates the variability represented by:

- `temporal_magnitude_cosine_similarity.std`

The residual calibrator exists to reduce that gap without destroying the
backbone's already-fitted mission-level power and phase statistics.

## Design Goal

The calibrator is designed to do three things:

1. Preserve frame-level power statistics already fitted by the backbone
2. Avoid disturbing the current phase process
3. Add local amplitude irregularity that better resembles realistic mission
   dynamics

This is why the first version calibrates **magnitude only** and then re-locks
each frame to the original frame-power target.

## First-Version Mechanism

The first version of the calibrator works as follows:

1. Load a complex synthetic backbone sequence
2. Build a residual template that is:
   - time-correlated
   - frequency-smoothed
   - lightly modulated by short event bursts
3. Apply the residual only in the amplitude domain with:
   - `magnitude * exp(gain * residual_template)`
4. Re-lock the frame power so the mission-level envelope is preserved
5. Keep the phase unchanged
6. Search for the best residual gain against the measurement targets

## Why The Calibration Acts In The Amplitude Domain First

At this stage, the backbone already matches:

- `frame_avg_db.mean`
- `frame_avg_db.std`
- `phase_increment_rad.std`

So the main deficiency is not mission-level power or wrapped phase spread, but
the lack of enough local temporal irregularity in the magnitude field.

That is why the first calibrator does not yet modify phase.

## Why Re-lock Frame Power After Calibration

If residual perturbations are applied directly to the magnitude field without
re-normalization, the calibrator can easily ruin the backbone's already-fitted
power statistics.

The re-lock step ensures that:

- mission-level power targets remain stable
- local residual structure is added on top of them

This makes the calibrator safer to iterate on.

## Why Search Over Gain Instead Of Hard-Coding One Value

The proper calibration strength is not obvious in advance.

If the gain is too low:

- the residual layer has almost no effect

If the gain is too high:

- the synthetic sequence becomes too irregular
- adjacent-subcarrier consistency can degrade too much

A small gain search makes the first prototype much more informative, because it
lets us observe where the trade-off starts to become harmful.

## Why The Calibrator Needed A Multi-Template Search

The first single-template version showed that:

- a small residual gain helps
- but the outcome still depends strongly on the template shape itself

That is why the next extension adds a limited search over:

- residual frequency smoothing
- residual time correlation
- event rate
- event duration
- event strength

This is still intentionally small-scale and interpretable. It is not meant to
be a large hyperparameter optimization system.

## Why A Two-Stage Residual Is The Next Logical Step

The current evidence suggests that one residual process is not enough to model
both:

1. slow, smooth mission-local fluctuation
2. sharper local bursts or regime changes

So the next design step is a two-stage residual:

- a smooth residual layer
- a sparse burst residual layer

The smooth layer should preserve continuity, while the burst layer should add
rare but meaningful local irregularity.

## Current Working Default

After the first focused smoke tests against the `BOLEK 40m peak-5mps
flight-only` target, the most stable two-stage default so far is:

- smooth residual:
  - `residual_freq_smoothing_span = 15`
  - `residual_time_corr = 0.93`
  - `event_rate_per_1k_frames = 10`
  - `event_duration_frames = 12`
  - `event_strength = 1.2`
- burst residual:
  - `burst_freq_smoothing_span = 5`
  - `burst_time_corr = 0.55`
  - `burst_event_rate_per_1k_frames = 22`
  - `burst_event_duration_frames = 4`
  - `burst_event_strength = 1.8`
  - `burst_mix = 0.15`

The important practical conclusion is that the burst layer helps only when it
remains lightweight. Larger burst mixing quickly degrades adjacent-subcarrier
consistency and worsens the overall calibration score.

## Literature Influence

This calibrator is not copied from a specific open-source project.

Its logic is guided by the same measurement-calibrated philosophy used for the
backbone:

1. **GAN-Based Massive MIMO Channel Model Trained on Measured Data**
   - motivates grounding synthetic generation in measured statistics rather than
     arbitrary simulation assumptions

2. **Enabling 6G Through Multi-Domain Channel Extrapolation: Opportunities and
   Challenges of Generative Artificial Intelligence**
   - reinforces that synthetic data should be judged by downstream usefulness
     rather than by appearance alone

3. **Accurate Channel Prediction Based on Transformer: Making Mobility
   Negligible**
   - reinforces the importance of preserving the time structure relevant for
     prediction tasks

## Current Interpretation

The residual calibrator should currently be understood as:

- a local correction layer
- not a final synthetic generator
- not yet a learned residual model

It is still a controllable research prototype whose purpose is to improve the
synthetic backbone in a measurable and explainable way.

## Next Step

The next implementation step is to upgrade the calibrator from:

- single residual template

to:

- smooth residual template
- sparse burst residual template

and test whether this reduces the remaining temporal-variability mismatch
without breaking the already-fitted adjacent-subcarrier statistics.
