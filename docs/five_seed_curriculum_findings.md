# Early Medium World curricula: what the five-seed comparison established

*November 13, 2026*

## What was tested

The first Medium World curriculum comparison asked whether presenting the tasks in a prerequisite order improves final same-object Binding over mixing all tasks from the start. A Reverse order tested whether the opposite sequence was harmful.

All 15 runs used the same 10,000-question training set, `AttributeBindingViT`, five paired seeds (42–46), and **3,000 optimizer updates**. Each task had 2,500 questions. Every question was presented **15 times**: 12 passes in the first 2,400 updates and three more during the final 600-update Mixed consolidation. Therefore, all schedules had 150,000 total presentations and 37,500 per task.

| Schedule | Updates 0–2,400 | Updates 2,400–3,000 |
|---|---|---|
| Mixed | All four tasks interleaved | Common Mixed |
| Staged | Color + Shape (1,200) → Conjunction (600) → Binding (600) | Common Mixed |
| Reverse | Binding (600) → Conjunction (600) → Color + Shape (1,200) | Common Mixed |

This was **not an order-only experiment**. All runs started with AdamW at learning rate `1e-4`, but `ReduceLROnPlateau` monitored *overall* ID-dev cross-entropy every 100 updates and carried its scheduler state across phases. It halved the LR after plateaus (patience three evaluations, absolute improvement threshold 0.001, floor `1e-6`). Each schedule therefore experienced a different effective LR history as its task mix changed. The results compare **complete training procedures: task order together with its induced LR schedule**.

The primary observations were the fixed-final ID-dev Binding score, Binding accuracy over global training steps (AULC), and task behavior around transitions. OOD-dev was inspected as a secondary, exploratory dataset. No held-out test split was used.

## Final performance

Five-seed means ± sample SD at update 3,000:

| Metric | Mixed | Staged | Reverse |
|---|---:|---:|---:|
| Overall ID accuracy | **72.46 ± 2.39%** | 71.68 ± 2.35% | 63.63 ± 6.33% |
| Color | **96.80 ± 4.36%** | 95.16 ± 3.99% | 79.22 ± 12.90% |
| Shape | **61.33 ± 1.33%** | 60.47 ± 1.45% | 56.25 ± 5.12% |
| Conjunction | **78.13 ± 3.75%** | 76.95 ± 4.48% | 66.25 ± 7.70% |
| Binding | 53.59 ± 2.73% | **54.14 ± 3.78%** | 52.81 ± 1.69% |
| OOD-dev Binding | **54.20 ± 1.98%** | 52.30 ± 5.52% | 51.70 ± 2.29% |
| ID Binding AULC | **0.530** | 0.521 | 0.514 |

The **Staged − Mixed** final ID Binding difference was **+0.55 percentage points**, with Staged ahead in only three of five seeds. Its descriptive paired bootstrap 95% interval was **[−1.64, +2.58] pp**. The direction is uncertain: these results do not show that the proposed prerequisite sequence wins reliably.

Reverse finished below Mixed in overall ID accuracy in all five seeds, by **−8.83 pp** on average (descriptive bootstrap interval **[−14.02, −3.63] pp**). This is a consistent disadvantage of the **Reverse procedure as implemented**. Because its LR also decayed differently, the result cannot be assigned uniquely to reversed task order.

## Learning speed and transitions

The global-step Binding AULC values were **0.5299 Mixed**, **0.5208 Staged**, and **0.5136 Reverse**. Staged − Mixed was **−0.0091** (about −0.91 pp of average accuracy), higher for Staged in one of five seeds; descriptive bootstrap interval **[−0.0156, −0.0009]**.

This metric includes the time before Staged receives any Binding examples. It measures performance over the *entire training budget*, not how efficiently Binding is learned after it first appears.

The Staged task transitions caused large short-term changes. Mean ID-dev accuracy changes over the first 100 updates after the **Conjunction → Binding** transition (1,800 → 1,900) were:

| Task | Change |
|---|---:|
| Color | **−14.84 pp** |
| Shape | −2.97 pp |
| Conjunction | **−12.97 pp** |
| Binding | +3.59 pp |
| Overall | −6.80 pp |

After the switch to common Mixed consolidation (2,400 → 2,500), Color increased **+15.70 pp** and Conjunction **+12.34 pp** on average, while Binding decreased **−2.27 pp**. The earlier **Attributes → Conjunction** switch also reduced Color by **−3.67 pp** and improved Conjunction by **+7.42 pp**.

These are changes in *answers*, not proof that features were deleted or restored. They show that specializing on one task changes performance on others, and that a common Mixed suffix can reverse some of those changes.

## Why the LR history changes the interpretation

The same adaptive LR rule had different outcomes:

- **Mixed:** no LR reductions in the five seeds; final LR `1e-4`.
- **Staged:** one to three reductions; final LR about `1.25e-5` to `5e-5`.
- **Reverse:** one to five reductions; in some runs the final LR reached about `3.125e-6`.

The scheduler evaluated all four tasks even while a stage withheld some of them. Worse overall validation loss or plateaus could trigger a reduction before a newly introduced task had much training. Consequently, a task that arrived late did not necessarily receive the same-size optimization updates as a task that arrived early.

The experiment therefore cannot tell us how much of Reverse's weakness, or Staged's limited improvement, came from task order rather than **task-dependent LR decay**. Later constant-LR and phase-aware-LR experiments were needed to examine that ambiguity.

## What the OOD result can and cannot say

Staged's mean OOD-dev Binding score was **1.90 pp below Mixed**; the descriptive paired bootstrap interval **[−6.25, +3.90] pp** spans both directions. Staged was ahead on OOD-dev in only one seed. ID and OOD rankings varied noticeably by seed.

The later dataset audit found a deeper limitation: in the current Medium World OOD generator, a scene contains **any held-out Color–Shape object if and only if the OOD Binding label is Yes**. Scene complexity also differs from ID. The OOD score is therefore **not a clean test of novel-combination compositional transfer**. The historical numbers remain useful for comparing predictions on that fixed evaluation set, but should not be used as proof of successful or failed transfer.

## Interpretation

- **No reliable prerequisite-first advantage:** Staged had a small mean final Binding benefit, but it was inconsistent across seeds and did not improve global-step AULC.
- **The Reverse training procedure was worse overall:** this includes both its learning order and the LR reductions induced by that order.
- **Training-stage transitions matter:** Binding gains were accompanied by losses in earlier-task answers; Mixed consolidation often recovered earlier tasks while reducing some Binding performance.
- **The underlying prerequisite capabilities were not established:** Shape remained weak, and later audits showed that the Conjunction generator permits a strong Color-only shortcut.

The outcome justified more controlled work on exposure timing, rehearsal and learning-rate policy, rather than a conclusion that the intended hierarchy of visual concepts had been acquired.

The executed training and analysis are in [`plateau_five_seeds.ipynb`](../notebooks/medium_world/plateau_five_seeds.ipynb) and [`plateau_five_seeds_analysis.ipynb`](../notebooks/medium_world/plateau_five_seeds_analysis.ipynb). The saved results are under [`outputs/medium_world/experiments/plateau_five_seeds/`](../outputs/medium_world/experiments/plateau_five_seeds/). All uncertainty estimates here are exploratory and use repeatedly inspected development data.
