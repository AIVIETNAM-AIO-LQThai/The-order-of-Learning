# Stage 1 audit fixes — corrected re-evaluation, no model retraining

This package implements Claude’s M1/M2 corrections and the relevant reporting changes.
The canonical Medium World training set is unchanged. All completed model and readout
weights stay under `outputs/medium_world/experiments/architecture_protocol/`.

**From repository root, extract this ZIP and open notebooks from `notebooks/medium_world/`.**

1. Run `00_architecture_evaluation_sets.ipynb` to generate corrected S and new sealed C;
   all outputs go under `architecture_protocol_corrected/`.
2. Run `01_architecture_frozen_incentives.ipynb`. With `REUSE_SAVED_READOUTS=True`,
   it reloads the saved 60 frozen readouts, evaluates them on corrected S, and recomputes G1.
3. Run `02_architecture_stem_training.ipynb`. With `REUSE_SAVED_STEM_WEIGHTS=True`,
   it reloads the original saved end-to-end checkpoints, evaluates corrected S, and recomputes D1.
4. Run `03_architecture_selection_analysis.ipynb` to inspect the corrected paired results.

**No 9,000-update retraining is required.** The evaluation must still run the frozen
ViT forward pass over the new questions and is not instantaneous. The saved original
models/checkpoints must remain in their original paths. Do not delete or overwrite them.

## Changes and decisions

- **M1:** Conjunction and operator examples now contain fixed queried attribute counts;
  new audits admit those counts only for bound-vs-separate comparisons, where they do not reveal the target.
- **M2:** ID-safe and held-out-creating swap metrics are reported separately; ID-safe
  pairs define the primary joint-consistency metric.
- The old C was accessed during the audit, so corrected C uses a new generation seed.
  It remains sealed: notebooks 01–03 use selection only. No confirmation claims are made.
- Frozen-readout and end-to-end weights are loaded from the old output directory;
  corrected metrics, predictions, weight copies and manifests go to the new directory.
- Model-run provenance includes the original checkpoint hashes, evaluated manifest hash,
  environment and a partial FLOP estimate. The D1 analysis includes pointwise and Shape-only contrasts.
- Record the original D1 +5 pp decision threshold; **do not** change it to +3 pp
  after inspecting prior results. No fallback architecture will be trained in this stage.
- Confirmation planning uses at least five and at most eight new seeds, with a
  conservative 4 pp SD planning floor. Insufficient precision is reported transparently.
- **Recoloring:** add a separate single-instance Shape-present diagnostic instead of
  silently changing the D1 set. Recoloring both-correct across all pairs remains the
  preregistered selection check.
- **Readout controls:** G1 uses the five seed-level averages of the two schedules;
  per-schedule checks are now recorded as secondary.
- **Not used:** optional internal-map study and optional operator one-hot remain deferred;
  neither is treated as a completed experiment.

## Additional caution

Old results from the originally generated S are exploratory and **must not be mixed**
with corrected results. The corrected S remains an architecture-selection split;
training happened before this correction, but the same old S was repeatedly inspected,
so this is not a new independent confirmation study. The fresh C must be accessed once
only after confirming the final architecture and reporting plan.

The original protocol v1 is retained in `docs/architecture_protocol_v1_source.md`;
this README serves as a documented audit amendment, not a silent revision of the
pre-existing protocol.
