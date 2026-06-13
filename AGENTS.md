# AGENTS.md - Baseball Pitching OBP Project

## Project Goal

This project is a baseball pitching biomechanics analysis and mechanism-note system.
The priority is reproducible analysis workflow and defensible mechanism notes, not one-off final artifacts.

## Working Assumptions

- The active project root is `D:\baseball\pitching\obp`.
- `baseball_pitching/` contains the current data, scripts, figures, and Obsidian-style mechanism notes.
- The repository may contain many exploratory scripts and generated outputs. Do not treat all untracked files as current task scope.
- `rg` is unavailable in this environment.
- Use `PYTHONIOENCODING=utf-8` for commands that read, write, or print project text/data.
- Run Python analysis through the project virtual environment when executing project scripts.

## Shell Policy

Use this PowerShell executable for project commands:

```powershell
C:\Users\User\AppData\Local\Microsoft\WindowsApps\pwsh.exe -Command "<command>"
```

For Python analysis, set UTF-8 and activate the environment first:

```powershell
[Environment]::SetEnvironmentVariable('PYTHONIOENCODING','utf-8','Process')
. 'D:\baseball\pitching\obp\baseball_pitching_env\Scripts\Activate.ps1'
```

## Data Policy

- Do not silently use fallback definitions.
- If an expected column, file, or event definition is missing or inconsistent, stop and surface the issue.
- Preserve original source columns unless the user explicitly approves overwriting them.
- Prefer adding clearly named derived columns over replacing original source columns.
- When updating large CSV files, verify the updated columns directly after the write.
- Do not update zip copies of CSV files unless the task explicitly requires zip synchronization.

## Foot Plant Policy

Official FP event columns are not trusted as analysis time points:

```text
fp_10_time
fp_100_time
```

They may be logged or compared for debugging only.

The project analysis FP is:

```text
fp_poi_time
```

`fp_poi_time` is reconstructed from POI FP angle values by matching these POI fields back to full-signal joint-angle columns within the PKH-to-BR window:

```text
pelvis_anterior_tilt_fp              -> pelvis_angle_x
pelvis_lateral_tilt_fp               -> pelvis_angle_y
pelvis_rotation_fp                   -> pelvis_angle_z
torso_anterior_tilt_fp               -> torso_angle_x
torso_lateral_tilt_fp                -> torso_angle_y
torso_rotation_fp                    -> torso_angle_z
rotation_hip_shoulder_separation_fp  -> torso_pelvis_angle_z
```

Current validation result:

```text
411 valid pitches
fp_poi_n = 7
fp_poi_range_ms = 0.0 for all reconstructed pitches
```

Do not use `lead_knee_extension_angular_velo_fp` to reconstruct FP unless its axis/sign/definition is separately validated. Prior checks showed it can disagree with angle-derived FP.

For FP-to-BR analyses, use:

```text
start = fp_poi_time
end = BR_time
```

and explicitly state this in any result summary or mechanism note.

## Analysis Policy

- Diagnose the data path before editing scripts or notes.
- Prefer deterministic code-side calculations over prompt or narrative fixes.
- For time-series analyses, normalize phases only after event definitions are fixed.
- For exploratory findings, report effect size and uncertainty before mechanism claims.
- For SPM or time-series correlation, first inspect pointwise relationships, then apply appropriate multiple-comparison or cluster correction.
- Do not treat weak correlations as primary mechanisms.
- If a final figure, note, or table is wrong, fix the upstream analysis workflow rather than patching only the final artifact.

## Mechanism Notes Policy

- `baseball_pitching/機制.md` is a Map of Content, not a raw analysis log.
- Mechanism notes should contain concise conclusions only. Put algorithms, detailed calculations, validation tables, and long evidence trails in separate linked notes or scripts.
- It is acceptable and preferred to connect concise mechanism conclusions to supporting material with Obsidian links, for example `[[fp_poi_time reconstruction]]` or `[[pelvis vertical displacement analysis]]`.
- Put phase-specific findings in the relevant phase note:
  - `抬腳.md`
  - `下沉.md`
  - `旋轉.md`
  - `前腳煞車.md`
  - `軀幹.md`
  - `投球手.md`
  - `手套手.md`
  - `體能.md`
- Keep numeric findings traceable to a script, data source, and date.
- Use mechanism-status language such as `假說`, `初步支持`, `弱支持`, `不支持`, or `需重算`.
- Do not manually rewrite final note conclusions as a substitute for updating the calculation path.

## Git And Verification Policy

- After edits, run `git status --short`.
- Run a task-scoped `git diff` or equivalent verification.
- Do not stage or commit unrelated dirty files.
- Do not commit when task files have mixed ownership, large generated rewrites, or ambiguous provenance.
- If the worktree is already dirty, identify the task-owned files clearly and leave unrelated changes alone.
