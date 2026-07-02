# Qualitative Mechanics Label Definitions

This file is a working draft for the Cloudflare labeling UI rubric.
It records current label definitions and open decisions before implementation.

Core rule: qualitative labels are visual judgments. Quantitative values may be
shown only after submission as reference values, and only when the mapping is
explicitly defined.

## hip_shoulder_separation

Current options:

- `good`
- `average`
- `bad`
- `unclear`

Draft definition:

Judge the visible amount of pelvis-to-torso separation during the delivery.
Use the current view only. Do not infer from pitch speed or other metrics.

Post-submit reference metric candidates:

- `rotation_hip_shoulder_separation_fp`
- `max_rotation_hip_shoulder_separation`

Open decision:

- Decide whether the visual label targets separation around foot plant or the
  maximum visible separation during the throwing sequence.

## lower_body_dominance

Current options:

- `glute`
- `quad`
- `mixed`
- `unclear`

Draft definition:

Judge whether the lower-body move visually appears posterior-chain dominant or
front-knee/quadriceps dominant.

Operational cue:

- `glute`: movement appears driven more by posterior-chain loading and hip
  action.
- `quad`: movement appears driven more by front-knee/quadriceps action.
- `mixed`: both patterns are visible or neither clearly dominates.
- `unclear`: the view is not sufficient for the judgment.

Post-submit reference metric candidates:

- To be discussed. Do not silently map this label to a metric.

Open decision:

- Decide whether to keep the coaching-style option names `glute` and `quad`, or
  rename them to more directly observable visual terms.

## direction

Current options:

- `good`
- `bad`
- `unclear`

Problem:

The current `good` / `bad` split likely mixes different stride-direction
patterns. A single `bad` category may collapse open stride and cross-fire even
though they are different visual and biomechanical patterns.

Draft replacement options:

- `straight`
- `open`
- `crossfire`
- `unclear`

Draft definition:

Judge the visible stride direction relative to the intended line toward home
plate.

Post-submit reference metric candidates:

- `stride_angle`
- `stride_length`

Open decision:

- Decide the exact option names before changing the Cloudflare schema or UI.

## shoulder_horizontal_abduction

Current options:

- `good`
- `average`
- `bad`
- `unclear`

Draft definition:

Judge the visible amount of throwing-arm horizontal abduction during the arm
cocking phase. Use the current view only.

Post-submit reference metric candidates:

- `shoulder_horizontal_abduction_fp`
- `max_shoulder_horizontal_abduction`

Open decision:

- Decide whether the visual target is foot plant, maximum layback/arm-cocking
  phase, or another explicit timing window.

## torso_velo_z

Current options:

- `fast`
- `slow`
- `unclear`

Problem:

The field name is metric-like. The UI label should describe the visual task,
not the underlying metric.

Draft UI label:

- `Torso rotation speed`

Draft definition:

Judge whether the torso visually rotates quickly or slowly during the main
rotation phase.

Post-submit reference metric candidate:

- `max_torso_rotational_velo`

Open decision:

- Keep the internal field name for compatibility, but display a clearer UI
  label.

## hip_extension

Current options:

- `good`
- `bad`
- `unclear`

Problem:

This label is hard to judge directly and may be closer to a hip-version of
early extension than a clean hip extension angle judgment.

Draft direction:

- Consider renaming or replacing this with a more visible label before another
  large labeling pass.
- Do not claim direct validation from POI hip extension angle fields unless
  those fields are explicitly available and mapped.

Post-submit reference metric candidates:

- To be discussed. Any metric shown here should be labeled as indirect unless a
  direct hip extension angle mapping exists.

Open decision:

- Decide whether this label should become `hip_extension_direction`,
  `hip_early_extension`, or another observable cue.

## heel_connection

Current options:

- `connected`
- `early_extension`
- `unclear`

Definition boundary:

This label refers to the rear heel.

Draft definition:

Judge whether the rear heel appears to stay connected long enough during the
early drive/stride phase, or whether it appears to extend/lift early.

Operational cue:

- `connected`: rear heel appears to maintain connection through the intended
  early move.
- `early_extension`: rear heel appears to lift or extend early.
- `unclear`: the rear heel cannot be judged from the current view.

Post-submit reference metric candidates:

- To be discussed. Do not silently map this label to lead knee, pelvis rise, or
  trunk rise metrics.

Open decision:

- Define the exact timing window for "long enough" before UI implementation.

## drift

Current options:

- `good`
- `average`
- `bad`
- `unclear`

Draft definition:

Judge the visible forward drift of the body toward home plate during the early
move.

Post-submit reference metric candidates:

- `cog_velo_pkh`
- `max_cog_velo_x`
- `stride_angle`

Open decision:

- Decide whether this label targets early COM movement speed, smoothness of
  forward move, or total forward displacement.
