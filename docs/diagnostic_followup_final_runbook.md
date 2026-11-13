# Final five-seed diagnostics — before changing the ViT

This package replaces the pilot versions of **Notebooks 02, 03 and 04**. The already completed `01_count_free_curves.ipynb` is unchanged and does not need to run again. Nothing in this package updates `learning.py`, the dataset, or original training experiments. The ViT weights stay **frozen**. Only supervised **diagnostic readouts** are fitted in 03 and 04.

## Setup (simple and deterministic)

Extract the ZIP **at the repository root**, preserving its `notebooks/`, `src/`, `scripts/`, and `docs/` paths. The package replaces `src/medium_visual_world/diagnostic_followup.py` with the same v1.1 module used in the pilot; it adds `final_diagnostic_followup.py`. Keep local backups if you made independent changes to the pilot module.

In PowerShell from the repository root, run:

```powershell
python scripts/medium_world/audit_final_diagnostic_followup.py
```

Then open Jupyter using `notebooks/medium_world/` as its **current working directory**. Each notebook uses exactly:

```python
from pathlib import Path
PROJECT_ROOT = Path().cwd().resolve().parent.parent
```

The two `.parent` calls are intentional: `notebooks/medium_world` → `notebooks` → repository root. Do **not** open Jupyter with the repository root as CWD and expect this expression to work.

Run in order: `02_binding_swaps.ipynb`, `03_shape_and_readout_probes.ipynb`, `04_task_head_interference.ipynb`. They are independent and can be run separately. The audit refuses to proceed if the frozen split hashes do not match `phase_aware_lr/schedule_comparison.json`, or if any SHA-256 verification of the original checkpoints fails. If any of the 30 checkpoints is missing, recover **the exact historical checkpoint file**, not a retrained approximation.

## Exact sample of trained checkpoints

- Curricula: `mixed` and `prereq_mixed_pair_focus`; both trained at **Constant LR**, not the decayed-LR models.
- Seeds: 42, 43, 44, 45, 46.
- Steps: 5400, 7200, 9000.
- Checkpoints: 2 × 5 × 3 = **30**. Exactly one original saved checkpoint per cell.
- Data: fixed train, ID-dev, and OOD-dev JSONL. The *new* swap/probe/head evaluation is on **ID-dev**. Notebook 01 already covers the LR policies and count-controlled OOD.
- No model checkpoint selection based on dev results, no training of the ViT and no additional training questions.

All outputs are isolated under:

`outputs/medium_world/experiments/diagnostic_followup/final_five_seeds/`

Do not delete or replace the original `diagnostic_followup/{count_free,binding_swaps,readout_probes,head_competition}` pilot outputs.

### Notebook 02 — Binding swaps

For each checkpoint, evaluate all **109** eligible fixed-count ID Binding questions using the same deterministic, label-flipping Color swap. Save per-question Yes probabilities before/after, true labels, predictions, checkpoint SHA, schedule, seed and step. Report both-correct and directional response separately, balanced by original Yes versus No label (34 versus 75 questions). The balanced aggregate prevents the dominant No→Yes direction from controlling the score. Require all 30 run files; compute paired **seed-level** changes at 5400→7200, 7200→9000, 5400→9000 and prereq-minus-mixed at each step. Inspect both correctness and probability-direction metrics. Do not treat question edits as independently randomized training trials.

Output files: 30 `swap_*.csv`, `swap_summary_30_runs.csv`, `swap_five_seed_contrasts.csv`, `swap_trajectories.png`.

### Notebook 03 — Shape probes

For each trained checkpoint, fit the original pilot's four oracle-centered, four-class Shape probes using **the same 35 epochs**, including 48-D mean, 1200-D spatial, 1200-D shuffled, and a shared patch-score approximation of the readout. Separately fit scene-level Shape Yes/No readouts: the pretrained original head, a logistic classifier on the **exact original eight scalars**, and a spatial-convolutional readout on the frozen 16×16×48 image feature map. These use the original train/ID-dev Shape questions and **no oracle object center**. A random encoder is trained for each seed **exactly once**, then reused as the matched reference for 6 trained checkpoints (2 curricula × 3 stages). Do **not** treat this as six independent random baselines.

35 fixed head epochs, Ridge alpha 10 as in the pilot. The scene Conv has more capacity than the original head: gains are *diagnostic opportunities*, not controlled architectural effects. The 1200-D versus 48-D Shape difference is also confounded by decoder dimensionality/capacity. Spatial shuffled input is a geometry perturbation control, not a proof of learned representations. Prior conclusions about trained ≈ random must be tested **across the five independent seeds**.

Outputs: 30 `object_metrics_*.csv`, 30 `scene_metrics_*.csv`, per-example prediction files, 5 `random_object_probe_seed*.csv`, a 30-row per-decoder aggregate, `trained_vs_random_paired_by_seed.csv`, stage-wise paired contrasts and curve figure.

### Notebook 04 — Task-head competition with better controls

Freeze the original model's 8-dimensional pre-head values. Compare the original pretrained head to four newly trained variants (40 fixed epochs, same question order):

| Variant | Task routing | Approx. parameters |
|---|---|---:|
| `original_pretrained_head` | Implicit in joint-query values | 354 |
| `shared_unaware_32` | Implicit | 354 |
| `task_aware_32` | Explicit 4-hot task identity | 482 |
| `task_aware_capacity_matched_94` | Explicit 4-hot task identity | **1412** |
| `separate_heads_32` | Four independent task heads | **1416** |

There are 4 main tasks plus a separate negative-only Conjunction subgroup, giving 25 summary rows per checkpoint. Evaluate Color-present/Shape-missing Conjunction negatives (specificity) in addition to overall Conjunction, Shape and Binding. The two ~1414-parameter variants improve *parameter matching* but **not** routing function-class or optimizer matching. A benefit for independent heads still does not prove shared-head *gradient interference*.

Outputs: 30 `head_metrics_*.csv`, 30 `head_predictions_*.csv`, `head_metrics_all_30_checkpoints.csv`, stage-wise paired contrasts, `paired_head_design_comparisons.csv`, and a final-head figure.

## Execution protection

- The notebooks **require all 30 checkpoint paths and verify their SHA-256** against the archived manifest.
- They verify original dataset hashes against the phase-aware LR study before running.
- Results are saved **individually per checkpoint**. On reruns, verified completed CSVs are reused; incomplete runs resume instead of recomputing every checkpoint.
- Existing files with wrong seed/schedule/step/checkpoint hash/ID alignment are treated as errors, **not silently overwritten**.
- Aggregate five-seed comparisons are produced only after all 30 conditions succeed; no partial experiment is labeled a final study.
- Question IDs, seed, schedule and checkpoint provenance are saved in raw output files.
- All inferences are **exploratory development results**; 95% seed intervals are unadjusted for multiple comparisons.
- A seed is a paired replication unit. The same 109 Binding question IDs and the same original Shape questions recur across all checkpoints. This is not 30 independent samples.
- The original checkpoints were reconstructed with small mismatches compared with historical archived predictions in earlier work. SHA validates the **saved reconstructed checkpoint**, not exact bitwise identity with past model states.

## What to share for interpretation

After running all three notebooks, ZIP and upload **only**:

`outputs/medium_world/experiments/diagnostic_followup/final_five_seeds/`

There is no need to upload the 30 `.pt` files or the entire dataset unless a provenance discrepancy appears. Share the executed three `.ipynb` files if an exception, warning or unexpected incomplete row count occurs. Do not pick only the best seed or best checkpoint.

## What an architectural conclusion requires

- Consistent improvement in count-controlled swaps from 5400→7200 and deterioration after 7200 suggests specialization-sensitive behavior and potential interference, **not** loss of stored representations.
- Reliable improvements from stronger scene-level Shape readouts on frozen features support *readout accessibility*, not whether spatial geometry, parameter count, or additional training caused the benefit.
- Random and trained encoders tested with the same external decoder can inform the marginal value of VQA training for that **specific probe**; the object-centric probe receives privileged ground-truth centers and labels.
- A task-aware shared head matching separate-head parameter count provides a better comparator than the original pilot, but interference needs further controlled ablations before a causal claim.

The model remains unchanged until we have analyzed the complete 30-checkpoint dataset.
