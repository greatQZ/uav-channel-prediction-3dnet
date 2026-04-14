# Mission Tiering Notes

## Interpretation Rule

- The speed token embedded in a mission filename is treated as the **maximum speed
  reached in that mission**, not as a guaranteed constant-speed trajectory.
- Example:
  - `40m_5mps` means `peak-speed ~= 5 m/s`
  - `50m_10mps` means `peak-speed ~= 10 m/s`
- This avoids over-interpreting filename labels as strict motion profiles.

## Dataset Priority

### Primary Source: `EDAZ_0917`

Use `EDAZ_0917` as the main source for Route A analysis because:

- the measurement family is richer and more diverse
- the channel recordings are more complete
- both lower- and higher-peak-speed BOLEK missions are available
- the BOLEK set already supports:
  - baseline calibration
  - higher-mobility validation
  - mixed-regime stress cases

### Secondary Source: `EDAZ_0710`

Treat `EDAZ_0710` as a secondary source because:

- it includes missions with peak speed up to `15 m/s`
- but the recording process suffered deletion / corruption / recovery
- the dataset is less complete than `0917`
- the channel record diversity is lower than `0917`

Use `0710` only as:

- a supplementary generalization check
- a high-speed reference sample
- an auxiliary robustness source

Do **not** use `0710` as the main calibration source for the first synthetic
prototype.

## BOLEK Mission Tiers

### Tier 1: Baseline Calibration

- `channel_estimates_20250917_140636_bolek_40m_5mps.txt`
  - interpreted as: peak-speed around `5 m/s`
  - current best primary calibration baseline
- `channel_estimates_20250917_143517_bolek_50m_5mps.txt`
  - interpreted as: peak-speed around `5 m/s`
  - secondary clean baseline / validation pair

### Tier 2: Higher-Mobility Validation

- `channel_estimates_20250917_164901_5ms_bolek_50m_10mps_inconstant.txt`
  - interpreted as: peak-speed around `10 m/s`
  - non-constant mission, more suitable for validation than calibration
- `channel_estimates_20250917_172122_5ms_bolek_40m_10mps_2roundsandmore.txt`
  - interpreted as: peak-speed around `10 m/s`
  - contains pause / multi-round structure

### Tier 3: Stress / Transition Case

- `channel_estimates_20250917_150803_5ms_bolek_50m_5mps_10mps_withsuddenstop.txt`
  - interpreted as: mixed-regime mission with peak-speed around `10 m/s`
  - includes regime transition and sudden-stop behavior
  - use as a stress case, not as a clean calibration source

## Practical Use For The Current Paper

For the first Globecom workshop paper:

1. Use `0917` as the authoritative mission family.
2. Use the `flight-only` BOLEK segments as the main analysis units.
3. Start synthetic calibration from Tier 1.
4. Use Tier 2 and Tier 3 only after the Tier 1 synthetic prototype is stable.
