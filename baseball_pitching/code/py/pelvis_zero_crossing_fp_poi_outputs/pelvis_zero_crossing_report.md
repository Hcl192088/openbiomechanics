# Pelvis 0-degree crossing analysis

FP definition: `fp_poi_time` only.

Stable crossing definition: the last negative-to-nonnegative crossing of `pelvis_angle_z = 0` between `pkh_time` and `fp_poi_time`, linearly interpolated between adjacent 360 Hz samples. This is an orientation threshold, not the physical onset of rotation.

## Coverage and validation

- Analyzed pitches: 411
- Unique pitchers inferred from the `session_pitch` prefix: 100
- PKH angle below 0 degrees: 411 / 411
- FP angle at or above 0 degrees: 411 / 411
- Pitches with exactly one upward crossing: 406
- Pitches with repeated upward crossings: 5
- Missing crossings: 0

## Stable 0-degree crossing

- All-pitch median relative to FP: -73.6 ms
- Low quartile (<= 81.4 mph): n = 104, mean = -87.3 ms
- High quartile (>= 87.9 mph): n = 104, mean = -87.2 ms
- High-minus-low mean difference: 0.1 ms; Welch p = 0.990760
- Pitch-level regression: n = 411, r = 0.0381, R2 = 0.0014, p = 0.441623
- Pitcher-mean sensitivity: n = 100, r = 0.0413, R2 = 0.0017, p = 0.683540

## First-crossing sensitivity

- After excluding the 5 repeated-crossing pitches, low-quartile mean = -90.4 ms and high-quartile mean = -87.2 ms.
- High-minus-low difference = 3.2 ms; Welch p = 0.676723.
- Therefore the apparent first-crossing association is driven by the repeated-crossing cases and is not retained as the main result.

## PKH-to-FP duration recheck

- Definition: `fp_poi_time - pkh_time`.
- n = 411, r = -0.0104, R2 = 0.000109, slope = -0.5427 mph/s, p = 0.833073.
- This recheck does not support a relationship between PKH-to-FP duration and pitch speed.
