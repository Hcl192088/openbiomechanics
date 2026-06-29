# Qualitative Mechanics Experiment Handoff

## Current State

- Experiment UI lives in `qualitative_mechanics_experiment.py`.
- Manifest lives in `manifest.csv`.
- Raw labels live in `labels.csv`.
- Current labeled sample: `n = 20`.
- Current labeled pitches are unique by `session_pitch`; no duplicate
  `session_pitch` rows were found in the latest check.
- `pitch_speed_mph` joined successfully for all 20 labeled rows.

## Current Labels

- `hip_shoulder_separation`: good, average, bad, unclear
- `lower_body_dominance`: glute, quad, mixed, unclear
- `direction`: good, bad, unclear
- `shoulder_horizontal_abduction`: good, average, bad, unclear
- `torso_velo_z`: fast, slow, unclear
- `hip_extension`: good, bad, unclear
- `heel_connection`: connected, early_extension, unclear
- `drift`: good, average, bad, unclear

## Latest Pilot Read

This is exploratory only. Do not write mechanism conclusions from this sample
without a reliability check and a larger sample.

- `hip_extension` currently has some positive signal:
  - good `n = 13`, bad `n = 7`
  - pitch speed: good 86.77 mph vs bad 84.76 mph
  - pelvis rotation at FP: good 39.01 vs bad 31.11
  - hip-shoulder separation at FP: good 33.05 vs bad 25.76
  - max hip-shoulder separation: good 35.22 vs bad 27.68
  - max torso rotational velocity: good 1085.36 vs bad 1016.79
- POI currently does not expose direct `rear_hip_angle_x_fp` or
  `lead_hip_angle_x_fp`, so `hip_extension` cannot yet be validated as a true
  hip extension angle label from POI alone.
- `heel_connection` stays in the protocol. The definition may still need a
  tighter written rubric before it is treated as a primary analysis variable.

## Pilot Statistics Method

Use this only as a pilot screen. The current analysis is descriptive, not a
defensible inference test.

1. Load `labels.csv` with `session_pitch` as a string.
2. Load `baseball_pitching/data/poi/poi_metrics.csv` with `session_pitch` as a
   string.
3. Left-join labels to POI on `session_pitch`.
4. Before interpreting anything, verify:
   - labeled row count
   - unique `session_pitch` count
   - duplicate `session_pitch` count
   - missing `pitch_speed_mph` count
   - label distribution for every active qualitative field
5. For each qualitative field, exclude blank and `unclear` rows from that
   field's group summary.
6. Report group `n` and group means for pilot metrics.
7. Treat all results as exploratory until sample size and reliability checks are
   adequate.

Default pilot metrics:

- `pitch_speed_mph`
- `max_cog_velo_x`
- `pelvis_rotation_fp`
- `rotation_hip_shoulder_separation_fp`
- `max_rotation_hip_shoulder_separation`
- `shoulder_horizontal_abduction_fp`
- `max_shoulder_horizontal_abduction`
- `max_torso_rotational_velo`
- `torso_rotation_fp`

Extra pilot metrics for `hip_extension`:

- `cog_velo_pkh`
- `stride_length`
- `stride_angle`
- `max_rear_hip_flexion`
- `max_rear_hip_internal_rotation_velo`
- `rear_hip_transfer_pkh_fp`
- `rear_hip_generation_pkh_fp`
- `rear_hip_absorption_pkh_fp`
- `lead_hip_transfer_fp_br`
- `lead_hip_generation_fp_br`
- `lead_hip_absorption_fp_br`
- `lead_knee_extension_from_fp_to_br`
- `lead_knee_extension_angular_velo_fp`
- `lead_grf_x_max`
- `lead_grf_y_max`
- `lead_grf_z_max`
- `rear_grf_x_max`
- `rear_grf_y_max`
- `rear_grf_z_max`

Important limitation: POI currently has hip energy, hip transfer, lead-knee,
GRF, stride, and rear-hip flexion/internal-rotation velocity metrics, but not
direct FP hip extension angle fields such as `rear_hip_angle_x_fp` or
`lead_hip_angle_x_fp`. Do not claim that the `hip_extension` label has been
validated as a direct hip extension angle label from POI alone.

## To Do

1. Add a `lead_knee_extension` label.
   - Candidate visual target: lead knee extension angle pattern around FP to
     BR.
   - Candidate quantitative checks: `lead_knee_extension_from_fp_to_br`,
     `lead_knee_extension_angular_velo_fp`, `lead_knee_extension_angular_velo_br`,
     and `lead_knee_extension_angular_velo_max`.
   - Decide options before coding. Likely candidates are `good`, `bad`,
     `unclear`, or a three-level scale if the visual distinction is stable.

2. Add a `lead_knee_extension_velo` or related velocity label only if it can be
   judged visually with acceptable consistency.
   - If visual judgment is mostly guessing from playback speed, do not add it
     as a separate label.
   - Prefer one lead-knee label first, then split angle vs velocity only if
     pilot reliability supports it.

3. Add a `shin_angle` label.
   - This is the visual category often called hip external rotation in coaching
     language, but that is not the correct biomechanical definition.
   - Keep the field name closer to the observable visual feature, such as
     `shin_angle`, instead of naming it `hip_external_rotation`.
   - If the UI text needs the coaching alias, write it as a note like
     `shin_angle (coaching alias: hip ext rot)` so later analysis does not
     confuse it with true hip external rotation.

4. Improve the `heel_connection` rubric without removing the field.
   - Keep `connected`, `early_extension`, and `unclear` for now.
   - Define the exact observable cue before the next large labeling pass.
   - Avoid mixing heel lift, lead knee extension, pelvis rise, and trunk rise
     into one implicit judgment unless the rubric explicitly says so.

5. Increase sample size before stronger statistics.
   - `n = 40` is the next useful checkpoint.
   - Keep repeated pitchers or repeated pitches in the manifest for reliability
     checks, but analyze reliability separately from the pooled label effects.
