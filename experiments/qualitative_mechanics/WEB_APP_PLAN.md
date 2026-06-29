# Qualitative Mechanics Web App Plan

## Current Decision

Build a small labeling and analysis tool, not a full coaching SaaS.

The current goal is:

1. Let each coach log in.
2. Show only items that coach has not completed.
3. Save one completed pitch-level task per coach and pitch.
4. Compare coach agreement at the qualitative-label level.
5. Show pooled analysis only for items with acceptable coach agreement.

Do not expand this into athlete management, full video hosting, billing,
coach-to-coach sharing, or public dashboards until the labeling workflow is
useful with real coaches.

## What To Keep

Keep the existing local experiment as the prototype and source of truth for the
first web version.

- `qualitative_mechanics_experiment.py`: current local labeling UI and pilot
  dashboard.
- `manifest.csv`: current sampled pitch order and pitch identifiers.
- `labels.csv`: current raw pilot labels.
- `README.md`: current local run protocol.
- `HANDOFF.md`: current experiment status and pilot findings.

Do not rewrite these files just to make the web app look cleaner. The first
web step should preserve the same label meanings and analysis assumptions.

## Current Scope

Use the user's simplified rule:

- One coach labels each assigned pitch-level task once.
- Each pitch-level task contains the active qualitative fields.
- Completed items are hidden or marked complete for that coach.
- Re-labeling is not part of the default workflow.
- Repeated reliability rounds are out of scope for the first version.

Current verification: `manifest.csv` has 58 rows, so the first web version
should treat the current workload as 58 pitch-level tasks per coach. This is
not 58 x 8 separate UI tasks. Internally, the web database may still store one
row per qualitative field for cleaner analysis, but the user-facing workflow is
one pitch-level task at a time.

## Minimal Data Model

Use a long-table label model for the web app.

```text
coaches
- id
- name
- password_hash
- is_admin
- created_at

label_tasks
- id
- session_pitch
- display_order
- active_label_fields
- active

labels
- coach_id
- session_pitch
- item_name
- label_value
- view_used
- playback_speed
- skipped
- skip_reason
- notes
- created_at
```

Rules:

- `label_tasks` should have one active row per current `manifest.csv` row.
- `labels` should have a unique key on `coach_id, session_pitch, item_name`.
- Do not trust `coach_id` from the browser request body; derive it from the
  login session.
- `item_name` must be one of the allowed qualitative fields.
- `label_value` must be one of the allowed values for that item.
- Skips are data and should be saved, not hidden as errors.

## Migration From Current Files

The current `labels.csv` is a wide table: one row contains all qualitative
fields for one pitch. The web UI should continue to feel like one pitch-level
task, but the database can store one row per qualitative field.

Migration step:

```text
labels.csv wide row
-> labels long rows
   coach_id
   session_pitch
   item_name
   label_value
   view_used
   playback_speed
   skipped
   skip_reason
   notes
   created_at
```

The first migration script should be read-only by default and print:

- source row count
- unique `session_pitch` count
- generated long-label row count
- skipped row count
- missing or invalid label values

Only write the converted output after those checks are clean.

## Analysis Scope

The first analysis page only needs:

1. Agreement summary across coaches.
2. Qualitative label groups vs pitch speed.
3. Qualitative label groups vs selected quantitative metrics.
4. Pooled analysis for items that pass the agreement threshold.

Do not write mechanism conclusions from this web app output unless the result
later becomes defensible under the project conclusion-note rules.

## Agreement Rule

Start simple. For each `item_name`, compare coaches on the same tasks and
report:

- number of shared tasks
- percent exact agreement
- unclear/skipped rate
- label distribution by coach

Only show pooled metric analysis for items above a chosen agreement threshold.
The threshold should be configurable later, but the first version can use a
constant in code.

Do not overbuild per-item reliability statistics until there are enough real
coach labels to justify it.

## Privacy And Permissions

Coach view:

- Can log in.
- Can see own remaining tasks.
- Can save own labels.
- Can see own progress.
- Can see anonymous pooled analysis only if enabled.

Admin view:

- Can create coaches.
- Can inspect all labels.
- Can export labels.
- Can view agreement and pooled analysis.

Default rule: coaches should not see another named coach's raw labels.

## Security Rules

Do these from the first deployed version:

- Store password hashes, never plain passwords.
- Store secrets in Cloudflare secrets, not GitHub.
- Use prepared statements and bind parameters for every database query.
- Validate `session_pitch`, `item_name`, and `label_value` against allowlists.
- Check login session on every write and analysis API.
- Do not commit real coach passwords, production labels, or secret keys.
- Do not use front-end-only permissions.

## Deployment Direction

Preferred deployment for the first online version:

```text
Cloudflare Pages
+ Pages Functions or Workers
+ Cloudflare D1
```

GitHub should host code only. It should not be the production data store.

GitHub Pages alone is not enough for this app because the app needs login,
server-side permission checks, and label writes.

Motion payload decision:

- Coach labels, task metadata, account data, and progress belong in SQL/D1.
- Skeleton motion frames should stay as static JSON assets or object storage
  objects, not SQL rows.
- Current local export uses `export_motion_json.py --write`.
- Current verified export: 58 motion JSON files under `web_motion/`, total
  payload bytes = 32,953,502, plus `web_motion_manifest.csv` mapping
  `session_pitch` to JSON path, frame count, and byte size.

## Implementation Steps

### Step 1 - Freeze Current Local Prototype

Verify current files and document current label fields.

Success check:

- `manifest.csv` can be read.
- `labels.csv` can be read.
- current label fields match `qualitative_mechanics_experiment.py`.

### Step 2 - Define Web Task List

Create a deterministic `label_tasks` source from the current planned tasks.

Success check:

- task count matches the expected count after verification.
- every task has `session_pitch`, `item_name`, and `display_order`.
- every `item_name` has allowed label values.

### Step 3 - Convert Existing Labels

Write a conversion script from current wide `labels.csv` to long labels.

Success check:

- generated row count equals valid source labels after excluding blanks as
  explicitly defined.
- invalid values fail visibly.
- no fallback label values are created.

Current implementation:

- `convert_labels_long.py` validates `labels.csv` against `label_tasks.csv`.
- Default mode is read-only and prints row counts.
- `--write` emits `labels_long.csv`.
- Current verified output: 30 source pitch rows x 8 label fields = 240 long
  label rows, with 0 skipped source rows.

### Step 4 - Build Local Database Prototype

Use SQLite locally to mirror the future Cloudflare D1 schema.

Success check:

- login works locally.
- saving a label creates exactly one row.
- submitting the same task twice updates or blocks according to one explicit
  rule chosen before coding.
- completed tasks no longer appear as pending for that coach.

Current implementation:

- `local_label_db.py --init` rebuilds local SQLite from `label_tasks.csv` and
  `labels_long.csv`.
- `local_label_db.py --smoke-test` checks login, pending task count, one-row
  label insert, duplicate insert blocking, and pitch-level completion.
- Current rule: one pitch-level task is complete only after all 8 active label
  fields exist for that coach and `session_pitch`.
- Current verified seed state: coach `1` has 30 completed pitch-level tasks and
  28 pending tasks; adding one item row keeps the task pending; completing all
  8 fields removes that task from pending.

### Step 5 - Add Minimal Analysis Page

Compute agreement and pooled metric summaries from the local database.

Success check:

- analysis reports shared-task counts before agreement percentages.
- unclear/skipped labels are reported separately.
- pooled metric analysis is hidden for low-agreement items.

Current implementation:

- `analyze_label_db.py` computes item-level coach agreement from local SQLite.
- It validates required POI metric columns before any metric summary.
- Default mode is read-only; `--write` emits `label_analysis_summary.json`.
- Pooled metric summaries are gated by `min_coaches`, `min_shared_tasks`, and
  exact-agreement threshold.
- Current verified state has only coach `1`, so all 8 items report
  `gate=fewer_than_two_coaches` and pooled analysis is hidden.

### Step 6 - Deploy Small Cloudflare Version

Move the working local prototype to Cloudflare Pages, Workers/Functions, and
D1.

Success check:

- production database is initialized from approved seed data.
- test coach can log in.
- labels write to D1.
- API rejects invalid item names and label values.
- GitHub repository contains code only, not production secrets or raw private
  labels.

Local API checkpoint before deployment:

- `local_label_api.py --smoke-test` starts a local HTTP server thread and tests
  login, pending tasks, full pitch-level label save, duplicate save blocking,
  and analysis summary retrieval.
- Current API save rule: `/api/labels` accepts one complete pitch-level task at
  a time and requires all 8 active label fields.
- Current verified API state: login returns 200, pending starts at 28, one
  complete save inserts 8 label rows and reduces pending to 27, duplicate save
  returns 409, analysis returns 8 item summaries.
- `GET /` now serves a minimal local prototype UI for login, pending task
  display, 8-field pitch-level submission, and agreement-gate summary.
- Current smoke test verifies the UI page loads before exercising the API.
- The local UI now reuses the original Three.js skeleton motion player path:
  pending tasks call `/api/motion?session_pitch=...`, which loads C3D motion
  data through the existing `load_motion_data` workflow.
- `/api/motion` now reads the exported static JSON path from
  `web_motion_manifest.csv`; it no longer falls back to reading local C3D files.

## Explicit Non-Goals For Now

- Full athlete-management system.
- Public coach dashboards.
- Paid accounts.
- In-app video storage for every pitch.
- Coach-to-coach named comparison.
- Complex reliability modeling.
- Automatic mechanism conclusions.
- Editing source data under `baseball_pitching/data/`.
