# AGENTS.md - Qualitative Mechanics Experiment

## Purpose

This folder is for qualitative pitching-mechanics labeling experiments.
It is not a conclusion-note area and it is not a place to patch final
biomechanics claims by hand.

## Experiment Rules

- Keep qualitative labels separate from quantitative biomechanical metrics.
- Do not turn labels into mechanism conclusions until reliability and metric
  checks have been run.
- Do not silently use fallback data, fallback camera views, fallback C3D files,
  or fallback marker definitions.
- If a pitch cannot be displayed or judged, record it as skipped with a clear
  skip reason.
- Preserve the view condition used for every label because view choice is part
  of the experiment.
- Preserve repeated pitches from the same pitcher. Later analysis must support
  pooled, by-pitcher, and within-pitcher summaries.
- Do not modify source CSVs under `baseball_pitching/data/`.
- Do not write final conclusions under `baseball_pitching/結論/` from this
  folder unless the finding is defensible and the supporting analysis path is
  explicit.

## Command Policy

- Use `PYTHONIOENCODING=utf-8` when running scripts.
- Run scripts with the project virtual environment:

```powershell
D:\baseball\pitching\obp\baseball_pitching_env\Scripts\python.exe
```

## Output Policy

- `manifest.csv` defines the sampled and randomized pitch order.
- `labels.csv` stores raw labeling output and should be append-only during a
  labeling session.
- Skipped items are data, not errors to hide.
