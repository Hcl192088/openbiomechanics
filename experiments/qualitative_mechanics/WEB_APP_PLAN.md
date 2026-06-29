# Qualitative Mechanics Web App Plan

## Current Decision

Build a small labeling and analysis tool, not a full coaching SaaS.

The current goal is:

1. Let each coach log in.
2. Show only items that coach has not completed.
3. Save one label per coach, pitch, and qualitative item.
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

- One coach labels each assigned qualitative item once.
- Completed items are hidden or marked complete for that coach.
- Re-labeling is not part of the default workflow.
- Repeated reliability rounds are out of scope for the first version.

The current user estimate is 58 labeling tasks per coach. Before coding the
web app, verify exactly how those 58 tasks are defined from the current item
and pitch selection, because the existing `labels.csv` stores multiple label
fields in one pitch row.

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
- item_name
- display_order
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

- `labels` should have a unique key on `coach_id, session_pitch, item_name`.
- Do not trust `coach_id` from the browser request body; derive it from the
  login session.
- `item_name` must be one of the allowed qualitative fields.
- `label_value` must be one of the allowed values for that item.
- Skips are data and should be saved, not hidden as errors.

## Migration From Current Files

The current `labels.csv` is a wide table: one row contains all qualitative
fields for one pitch. The web app should store one row per item.

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

### Step 4 - Build Local Database Prototype

Use SQLite locally to mirror the future Cloudflare D1 schema.

Success check:

- login works locally.
- saving a label creates exactly one row.
- submitting the same task twice updates or blocks according to one explicit
  rule chosen before coding.
- completed tasks no longer appear as pending for that coach.

### Step 5 - Add Minimal Analysis Page

Compute agreement and pooled metric summaries from the local database.

Success check:

- analysis reports shared-task counts before agreement percentages.
- unclear/skipped labels are reported separately.
- pooled metric analysis is hidden for low-agreement items.

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

## Explicit Non-Goals For Now

- Full athlete-management system.
- Public coach dashboards.
- Paid accounts.
- In-app video storage for every pitch.
- Coach-to-coach named comparison.
- Complex reliability modeling.
- Automatic mechanism conclusions.
- Editing source data under `baseball_pitching/data/`.

