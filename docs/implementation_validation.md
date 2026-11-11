# Representation Diagnostics — Verification and Fixes

This report applies to the rebuilt **full five-seed package**. The experiment remains configured for seeds **42–46**, `prereq_mixed_pair_focus` and `mixed`, steps **5,400 / 7,200 / 9,000** (10 training runs, 30 checkpoints).

## Corrections

1. Reindented all notebook code to **four spaces per block**; expanded nested loops, multi-operation lines, dictionaries, and long expressions. Retained the original full-run settings and training/evaluation order.
2. Validated notebook metadata and syntax, checked all `INDENT` tokens for four-space multiples, and checked physical line lengths.
3. Added **checkpoint SHA-256 matching** in the analysis loader so every loaded per-run CSV must match the 30-entry checkpoint manifest. The probing notebook also verifies stored dataset and training-source hashes before inference.
4. Made the statistical helper reject missing/repeated seeds and handle exact zero between-seed variance without producing undefined t-intervals.
5. Changed frozen-feature probe checks from requiring the *same number* of observed classes in train and evaluation to requiring that all evaluation classes be **contained in** training classes. Macro-F1 and confusion matrices retain the complete training-class label set.
6. Corrected explanation-question selection: ID-dev uses balanced present-pair positives versus **hard negatives** with both independent attributes present, excluding held-out targets; OOD-dev uses the original held-out Pair questions with balanced labels. This avoids mistaking easy attribute-absence cues for object-level Binding evidence.
7. Improved Network Dissection visualization: display a channel's actual dev-scene activation and mask, selected from the same deterministic dev scenes used for IoU. This deliberately **best-case illustration** is never used as population-level evidence.
8. Added checkpoint provenance to the spectral eigenvalue CSV; implemented a corresponding strict check in analysis.
9. Extended the preflight audit to validate both ID and OOD explanation-question sampling and the oracle.

## Tests run by the assistant

- **PASS** — All three notebook structures validate using `nbformat`, all code cells parse as Python, and all block-indentation tokens are multiples of four spaces (no tabs).
- **PASS** — CPU integration of same-scene reasoning, repeated-question deduplication, invariant/label-changing counterfactuals, object features, linear probing, concept mask overlap, and spectral clustering using oracle-valid synthetic scenes.
- **PASS** — OOD explanation-query selection preserves held-out targets and balanced original labels on synthetic scenes.
- **PASS** — The checkpoint notebook's training and saving flow executed **10 shortened synthetic run slots**, producing 30 checkpoint files and 60 comparison entries. This is not 9,000-step training.
- **PASS** — Every probing notebook stage executed on a synthetic checkpoint, including random encoders, shuffled-label probes, train/dev dissection, attribution maps, spectral graphs, and file-inventory verification.
- **PASS** — The entire analysis notebook executed on a synthetic **30-checkpoint** evidence inventory, including checks against manifest hashes, conditional subgroups, uncertainty calculations, concept visualizations and graph stability.

## What has not been validated

Full training on the user's exact frozen 10,000-question dataset and NVIDIA GPU, matching the archived LR-reset predictions, the *scientific values* of probes/IoU/clusters, and the wall-time/memory demands of 30 real checkpoint evaluations. Those require execution in the actual repository with the genuine data. None of the synthetic outputs are presented as experimental results.

If replacing an earlier ZIP, rerun the diagnostics and analysis notebooks to regenerate all explanation/evaluation outputs under the revised protocol. A matching complete checkpoint group can be reused.
