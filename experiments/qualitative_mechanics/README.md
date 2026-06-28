# Qualitative Mechanics Experiment

This experiment tests whether visual mechanical labels from 3D pitching
skeleton playback agree with later quantitative biomechanical measurements.
It is inspired by Driveline's qualitative-vs-quantitative pitching mechanics
experiment, but uses this repo's C3D skeleton data instead of video.

## First-Pass Protocol

1. Start the local experiment server.
2. Enter a rater id.
3. Label pitches in manifest order.
4. Choose the view you actually used: home, open side, or free.
5. Skip clips that are not displayable or not judgeable, and record why.
6. Do not inspect quantitative metric values while labeling.

## Labels

- `hip_shoulder_separation`: present, absent, unclear
- `lower_body_dominance`: glute, quad, mixed, unclear
- `direction`: stride, hip_extension, unclear
- `shoulder_horizontal_abduction`: early, neutral, excessive, unclear
- `heel_connection`: connected, early_extension, unclear
- `drift`: present, absent, unclear

## Sampling

The default manifest keeps repeated pitches from the same pitcher so that
pitch-to-pitch consistency can be checked. Later analysis should be run both
pooled and grouped by pitcher.

## Files

- `qualitative_mechanics_experiment.py`: local server and browser UI.
- `manifest.csv`: sampled pitches and randomized presentation order.
- `labels.csv`: raw label records.

## Run

```powershell
$env:PYTHONIOENCODING='utf-8'
D:\baseball\pitching\obp\baseball_pitching_env\Scripts\python.exe D:\baseball\pitching\obp\experiments\qualitative_mechanics\qualitative_mechanics_experiment.py
```

The script reads source data from `baseball_pitching/data/` and writes only
inside this experiment folder.
