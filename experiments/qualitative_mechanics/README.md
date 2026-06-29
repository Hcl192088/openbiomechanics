# Qualitative Mechanics Experiment

This experiment tests whether visual mechanical labels from 3D pitching
skeleton playback agree with later quantitative biomechanical measurements.
It is inspired by Driveline's qualitative-vs-quantitative pitching mechanics
experiment, but uses this repo's C3D skeleton data instead of video.

## First-Pass Protocol

1. Start the local experiment server.
2. Enter a rater id.
3. Label pitches in manifest order.
4. Choose the view you actually used: home, open side, second base, or free.
5. Skip clips that are not displayable or not judgeable, and record why.
6. Do not inspect quantitative metric values while labeling.

Pitch and pitcher identifiers are hidden by default for blind labeling. Use
`Reveal IDs` only for debugging or audit checks.

## Labels

- `hip_shoulder_separation`: good, average, bad, unclear
- `lower_body_dominance`: glute, quad, mixed, unclear
- `direction`: good, bad, unclear
- `shoulder_horizontal_abduction`: good, average, bad, unclear
- `torso_velo_z`: fast, slow, unclear
- `hip_extension`: good, bad, unclear
- `heel_connection`: connected, early_extension, unclear
- `drift`: good, average, bad, unclear

## Sampling

The default manifest is randomized with a fixed seed and keeps repeated pitches
from the same pitcher so that pitch-to-pitch consistency can be checked. Later
analysis should be run both pooled and grouped by pitcher.

Open-side view is handedness-aware: left-handed pitchers use the opposite
lateral camera offset from right-handed pitchers.

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

Open the pilot dashboard from the server URL plus `/dashboard`. It recomputes
the exploratory label-vs-POI screen from `labels.csv` and
`baseball_pitching/data/poi/poi_metrics.csv`.

Label groups are displayed in stable reader-facing order, for example
`good`, `average`, `bad`, then `unclear` when present.

The dashboard is a pilot screen only. Treat p-values as triage signals for the
next labeling or rubric pass, not as mechanism conclusions.

To emit the same statistics as JSON without opening the browser:

```powershell
$env:PYTHONIOENCODING='utf-8'
D:\baseball\pitching\obp\baseball_pitching_env\Scripts\python.exe D:\baseball\pitching\obp\experiments\qualitative_mechanics\qualitative_mechanics_experiment.py --pilot-stats --no-browser
```

## Patch Existing Labels

To relabel only existing rows that need one updated field, run patch mode. It
loads only rows whose current field value is outside the current allowed values
and merges the new value back into the matching `labels.csv` row.

```powershell
$env:PYTHONIOENCODING='utf-8'
D:\baseball\pitching\obp\baseball_pitching_env\Scripts\python.exe D:\baseball\pitching\obp\experiments\qualitative_mechanics\qualitative_mechanics_experiment.py --patch-field hip_shoulder_separation
```
