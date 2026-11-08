# Scaffolded Spiral — experimental protocol (planned)

**Status: planned, not executed.** This is a prespecified protocol, **not findings**. Do not create a findings document until the completed training outputs and analysis have been reviewed.

## Research question

Does learning Color and Shape together, introducing Conjunction with rehearsal, then concentrating Binding training before a rehearsal-based rebalancing phase yield better fixed-final Binding than established curricula under the same total presentations?

The combined curriculum is a *complete policy*. A difference from a reference curriculum cannot by itself identify the benefit of each ingredient. The `spiral_late_focus` control reverses the two suffix blocks after an identical 4,450-update prefix, making the effect of ordering more identifiable; it still changes intermediate exposure timing.

## Fixed setting

Medium World frozen `train.jsonl`, `id_dev.jsonl`, and `ood_dev.jsonl`. `AttributeBindingViT(dim=48, patch_size=4, depth=1)` unchanged. AdamW, constant `lr=1e-4`, weight decay 0, batch 50, gradient clipping 1.0, existing image augmentation. Paired seeds 42–46. Nine thousand updates per schedule; every exact question receives 36 main-phase passes plus nine common-Mixed passes (45 total). The same per-task question streams and final common-Mixed example sequence are used for each seed. The audit also verifies that the three reference schedules reproduce the corresponding existing builders exactly, and that both Spiral variants have identical first 4,450 updates.

No scheduler resets, mastery decisions, new architectures, datasets, or test-set evaluations occur in this study.

## Training schedules

Each number means complete passes over that task's 2,500 questions: C = Color, S = Shape, J = Conjunction, P = Binding. Each curriculum has 36 main passes per task plus C9/S9/J9/P9 common consolidation.

| Curriculum | Main sequence | Purpose |
|---|---|---|
| `scaffolded_spiral` | C28 S28 (2,800 updates) → C3 S3 J27 (1,650) → C2 S2 J3 P26 (1,650) → C3 S3 J6 P10 (1,100) | Integrated scaffold, Binding focus then rebalancing |
| `spiral_late_focus` | Same first 4,450 updates → C3 S3 J6 P10 (1,100) → C2 S2 J3 P26 (1,650) | Reverse last two blocks, same prefix and final totals |
| `staged_rehearsal` | C30 S30 (3,000) → C3 S3 J33 (1,950) → C3 S3 J3 P36 (2,250) | Existing staged-rehearsal reference |
| `prereq_mixed_pair_focus` | C36 S36 J36 (5,400) → P18 (900) → P18 (900) | Prerequisite-mixed reference, late Binding specialization |
| `mixed` | C36 S36 J36 P36 (7,200) | Fully mixed reference |

Every schedule ends with C9/S9/J9/P9 shared Mixed consolidation (1,800 updates, 7,200–9,000).

For `scaffolded_spiral`, stage boundaries are **2,800, 4,450, 6,100, 7,200, 9,000**; initial Binding exposure is at step 4,450. Binding shares in successive non-common stages are 0%, 0%, 78.8%, 45.5%, followed by 25% in consolidation.

## Hypotheses and comparisons

**Primary:** fixed-final step-9,000 ID-dev Binding accuracy difference `scaffolded_spiral − staged_rehearsal`, paired by seed.

**Secondary:** `scaffolded_spiral − spiral_late_focus` to compare the direction of the Binding-focus/rebalancing sequence; `scaffolded_spiral − prereq_mixed_pair_focus` to compare against mixed prerequisites plus late Pair focus; and `scaffolded_spiral − mixed` as conventional baseline. These contrasts compare the *full observed differences*, not individually isolated properties.

Report all five seed differences, mean, sample SD, 95% paired t interval, and count of positive differences. With n=5, intervals are descriptive, not a guarantee of reproducibility. No cherry-picking checkpoints or seeds.

## Evaluation

- Step 0, every 100 updates, exact stage boundaries, and 100 updates after each boundary. Compare **fixed-final step 9,000** as primary outcome.
- ID-dev 256 per task; OOD-dev 400 withheld Binding targets. Do not evaluate ID-test or OOD-test.
- Final all-task accuracy, ID/OOD Binding, cross-entropy, Binding AULC against global steps, and Binding accuracy aligned by cumulative Pair presentations.
- Positive recall, negative specificity and predicted-positive rate separately, to detect a change in Yes/No prediction bias.
- Shape recognition by queried shape; Conjunction negative cases by which attribute is missing; Binding positives/hard negatives; OOD per held-out pair and object count.
- Per-question prediction traces allow correct→wrong, wrong→correct, recovered at final step, and schedule-paired disagreements.
- Analyze **genuine task-distribution transitions**, not phase-label boundaries where the proportions are unchanged (e.g. the two pure Binding blocks).

Metrics from all evaluation questions are repeated within each seed, so **seed is the unit of replication**. Note that the historical 600-update pilot already touched test splits and development OOD has been examined repeatedly; neither creates a fresh independent final test for the present study.

## Interpretive caution

AULC against global updates rewards early Binding introduction. Exposure-aligned curves are useful but cannot isolate the effect of prerequisite training. Matching the seed and per-task presentation order does not match stochastic augmentation on an example-by-example basis. A higher OOD total accuracy may arise from positive-recall bias rather than better hard-negative rejection. Existing Conjunction data have a color-only shortcut. Any positive result supports this complete curriculum at the tested settings, not a universal principle or independent benefit from every included property.

## Source files and outputs

- `src/medium_visual_world/scaffolded_spiral.py`: schedule generation only.
- `scripts/medium_world/audit_scaffolded_spiral.py`: exact-exposure scientific audit.
- `notebooks/medium_world/scaffolded_spiral.ipynb`: train and record.
- `notebooks/medium_world/scaffolded_spiral_analysis.ipynb`: analyze saved outputs, no retraining.
- `outputs/medium_world/experiments/scaffolded_spiral/`: result JSON, compressed prediction traces, CSVs, figures.

Write `docs/scaffolded_spiral_findings.md` **only after** evaluating the completed outputs. Do not pre-create it.
