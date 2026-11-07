# Medium World — isolating task-mixture smoothness and Binding recency

**Frozen design · November 7, 2026**  
**Status:** protocol and code prepared; no GPU results yet

## Question

After learning Binding, do smaller changes in the task mixture preserve Binding better than larger stepwise changes **when total training exposure and the mean timing of Binding practice are matched**? Does the effect differ when Binding practice is relatively early or late?

This is a controlled follow-up to `transition_rehearsal`. Preserve the existing `AttributeBindingViT(dim=48, patch_size=4, depth=1)`, frozen Medium World dataset, and step-6,300 `prereq_mixed_pair_focus` parent. Train five paired seeds, 42–46. Keep AdamW, learning rate `1e-4` (constant), weight decay 0, batch size 50, augmentation, clipping, and evaluation every 100 updates unchanged. **No LR scheduler is introduced in this experiment.**

## Shared parent

For each seed, the first 5,400 steps mix Color, Shape, and Conjunction; steps 5,400–6,300 train Binding. Continue from **identical weights, full AdamW state, and Python/NumPy/PyTorch CPU/CUDA RNG state** for all four strategies. Import the previously saved parent checkpoint and prefix trace from `transition_rehearsal/five_seeds` if available and verified; otherwise train and save the same parent again.

## Four suffix strategies

Each entry is the **percentage of Binding examples** in consecutive **300-update windows** after step 6,300. The other three tasks divide the remainder equally. Task types are randomly interleaved within each window using the same symmetric-shuffle construction.

| Window ending at step | Smooth / early | Stepwise / early | Smooth / late | Stepwise / late |
|---|---:|---:|---:|---:|
| 6,600 | 80% | 80% | 20% | 20% |
| 6,900 | 70% | 74% | 25% | 32% |
| 7,200 | 65% | 74% | 35% | 32% |
| 7,500 | 60% | 46% | 45% | 46% |
| 7,800 | 50% | 46% | 50% | 46% |
| 8,100 | 45% | 46% | 60% | 46% |
| 8,400 | 35% | 32% | 65% | 74% |
| 8,700 | 25% | 32% | 70% | 74% |
| 9,000 | 20% | 20% | 80% | 80% |

**All four:** 2,700 suffix updates; 67,500 Binding presentations and 22,500 of each earlier task, and exactly **45 presentations per individual training question** over the entire 9,000 updates. Every strategy uses the same order of questions *within* each task. No ID/OOD test split is used.

### Matched timing and smoothness

Let `p[w]` be the Binding percentage in window `w` (1–9). The two smoothness strategies within each recency level have:

- The same `sum(p[w]) = 450` and equal example counts for all tasks.
- The same **first temporal moment**, `sum(w * p[w])`: `1,800` for early and `2,700` for late, giving a mean Binding presentation at step **7,350** or **7,950**, respectively.
- The same Binding percentage in the **first and last window** (80%/20% for early; 20%/80% for late).
- An exactly matched *within-window* temporal centroid for every task, enforced by mirroring the shuffled list of task labels around the center of each 300-update window. This makes mean sample-position timing equal, not just equal in expectation.

**Smoothness manipulation:** the largest adjacent 300-window Binding-percentage jump is **10 pp** for smooth and **28 pp** for stepwise. The sum of squared adjacent percentage-point changes is **500** for smooth and **1,160** for stepwise (different patterns, same total variation). The percentages define smoothness *at a 300-step resolution*, not gradient-level smoothness.

**Recency manipulation:** reversing the profile moves mean Binding exposure **600 optimizer steps later** without changing totals or the profile's maximal jump. Recency also changes the mixture immediately after step 6,300 (80% vs 20% Binding), so interpret it as a *time-allocation intervention*, not a single perfect scalar control.

### Scope of the causal comparison

1. **Smoothness at fixed recency** is the main comparison: `smooth_early − step_early` and `smooth_late − step_late`. Number of presentations, per-task question order, mean timing, and first/last-window exposure match. Differences concern the intermediate profile's sharpness and higher-order timing; we cannot claim every temporal property is matched.
2. **Recency at fixed profile class:** `smooth_late − smooth_early` and `step_late − step_early`. Counts remain equal while practice is shifted later; initial task-shift magnitude also changes.
3. **Interaction:** `(smooth_late − step_late) − (smooth_early − step_early)`.

No strategy is the exact prior `abrupt` baseline. The earlier three-strategy experiment remains a separate reference, not a member of this factorial control.

## Outcomes and analysis

**Primary outcome:** final step-9,000 ID-dev Binding accuracy, paired by seed, for the two smoothness comparisons. Report all five differences, mean, standard deviation, and paired 95% Student's t-interval. The recency and interaction contrasts are secondary.

**Additional outcomes:** OOD-dev Binding, Binding accuracy by global update, Color/Shape/Conjunction retention from the common checkpoint, positive recall/negative specificity, correctness churn and recovery, and final errors by target pair and number of objects. Evaluate step 6,300, 7,200, 7,300, and 9,000 as predefined landmarks. Avoid treating a chosen maximum as the primary endpoint.

**Interpretation:** evidence for an effect of the measured **size of changes in task mixture** requires matched-recency smoothness comparisons to differ consistently; any late-practice benefit is not evidence for smoothness. If neither comparison is convincing, document the null/mixed result without adding unplanned curriculum variants.

## Files

- `src/medium_visual_world/smoothness_recency.py`: frozen schedule generator.
- `valid_tests/test_smoothness_recency.py`: exposure and exact timing checks.
- `notebooks/medium_world/smoothness_recency.ipynb`: all 20 continuations and checkpoint reuse.
- `notebooks/medium_world/smoothness_recency_analysis.ipynb`: paired comparisons, curves, retention and errors.
- Output directory: `outputs/medium_world/experiments/smoothness_recency/five_seeds/`.

Keep checkpoints locally; the repository `.gitignore` already excludes `.../experiments/.../checkpoints/`. Save the output traces and summary separately from prior experiments.

## Deferred research directions

**After analysis is complete:** (1) examine learning-rate scheduler resets at task boundaries; (2) discuss gradient-informed task allocation with Claude **before** designing an implementation, considering gradient alignment/conflict, norms and learning progress, and safeguards against allocating more questions to already easy/noisy tasks. Decide whether that line warrants a targeted small-world pilot or whether Large World is more informative. Neither direction changes this frozen experiment.
