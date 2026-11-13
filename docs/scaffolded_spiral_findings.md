# Scaffolded Spiral: balancing Binding specialization and earlier-task retention

*November 13, 2026*

## Why this curriculum was tested

Earlier experiments showed that training heavily on Binding can improve Binding accuracy while disrupting Color, Shape and Conjunction answers. Rehearsal reduces some of that disruption, but it does not establish a better final Binding outcome. Scaffolded Spiral combined staged prerequisites, later Binding focus and broader rehearsal to test whether these effects could be balanced within the same fixed budget.

A second schedule, **Spiral Late Focus**, reversed the order of the last two training blocks. This asked a narrower question: after a shared planned prefix, should Binding-heavy practice come **before** or **after** rebalancing?

## The comparison

Five seeds (42–46) were run across five curricula using the frozen Medium World and the same `AttributeBindingViT`. AdamW used a **constant learning rate of `1e-4`**, batch size 50 and unchanged augmentations. Every run had **9,000 updates**, **45 presentations per question**, and an identical planned nine-pass Mixed consolidation block during updates 7,200–9,000. Results below use fixed-final development checkpoints, not the best checkpoint selected afterward.

The two Spiral schedules were:

| Phase | Scaffolded Spiral | Spiral Late Focus |
|---|---|---|
| 0–2,800 | Color and Shape | Same |
| 2,800–4,450 | Conjunction with earlier-task rehearsal | Same |
| 4,450–5,550 | Binding-heavy practice (part of block ending at 6,100) | Broader rebalancing |
| 5,550–6,100 | Binding-heavy practice | Binding-heavy practice (part of block ending at 7,200) |
| 6,100–7,200 | Broader rebalancing | Binding-heavy practice |
| 7,200–9,000 | Common Mixed | Common Mixed |

More precisely, the **Scaffolded** suffix is `C2 S2 J3 P26` passes for 1,650 updates, followed by `C3 S3 J6 P10` for 1,100 updates. **Late Focus** reverses those two blocks. Before this suffix, `C28 S28` passes occupy 2,800 updates and `C3 S3 J27` passes occupy 1,650. These allocations give the two schedules equal total presentations; their timing differs. In Scaffolded Spiral, Binding makes up **78.8%** of the focus block and **45.5%** of rebalancing.

The comparison also included Staged Rehearsal, Prerequisite-Mixed → Pair Focus, and Fully Mixed as references. All five schedules share task-level exposure totals but not the same intermediate state or task sequence.

**Important execution limit:** the two Spiral runs were trained separately rather than forked from the same saved model, AdamW and RNG state at update 4,450. Their recorded prefix predictions match for seeds **43–46**, but **seed 42 differs slightly before the intended divergence**. Therefore the suffix contrast is not a perfect same-state causal intervention. It measures the realized schedules with that qualification.

## Fixed-final results

Five-seed mean at update 9,000:

| Curriculum | ID Binding | OOD-dev Binding | ID Binding AULC |
|---|---:|---:|---:|
| Prerequisite-Mixed → Pair Focus | **62.97%** | **57.50%** | 54.99% |
| Staged Rehearsal | 62.73% | 56.65% | 55.17% |
| Fully Mixed | 61.88% | 56.45% | **56.76%** |
| **Scaffolded Spiral** | **61.25%** | **55.45%** | **54.94%** |
| Spiral Late Focus | 60.78% | 55.90% | 54.73% |

Paired contrasts for Scaffolded Spiral minus its references (percentage points, unadjusted 95% Student-*t* intervals across five seeds):

| Reference | ID Binding | OOD-dev Binding |
|---|---:|---:|
| Staged Rehearsal, primary comparison | **−1.48 [−4.42, +1.45]** | −1.20 [−4.19, +1.79] |
| Spiral Late Focus | +0.47 [−1.51, +2.45] | −0.45 [−2.01, +1.11] |
| Prerequisite-Mixed → Pair Focus | −1.72 [−4.80, +1.36] | −2.05 [−7.88, +3.78] |
| Fully Mixed | −0.63 [−3.63, +2.38] | −1.00 [−8.26, +6.26] |

**No comparison shows a reliable fixed-final Binding advantage for Scaffolded Spiral.** The small mean ID difference between the two Spiral suffix orders is unresolved, and the seed-42 prefix discrepancy further limits its interpretation.

The global-step Binding AULC for Scaffolded Spiral was **1.83 pp below Fully Mixed** (paired interval **[−3.25, −0.41] pp**; lower in all five seeds). Fully Mixed encounters Binding from the beginning, whereas Spiral first encounters it at update 4,450. This AULC difference describes end-to-end performance under equal global budgets; it does **not** isolate learning efficiency after Binding introduction.

## What changed during training

Immediately before Binding was introduced in Scaffolded Spiral (update 4,450), the mean ID-dev accuracies were **99.22% Color**, **63.59% Shape**, **79.92% Conjunction**, and **50.39% Binding**. The curriculum had completed the planned prerequisites, but had **not** demonstrated robust Shape mastery. The apparently strong Conjunction result also needs the shortcut analysis below.

At update 7,200, just before the final Mixed period:

| Task | Scaffolded Spiral | Prerequisite-Mixed → Pair Focus | Staged Rehearsal |
|---|---:|---:|---:|
| Color | 98.83% | 94.30% | 98.36% |
| Shape | 63.83% | 57.34% | 64.61% |
| Conjunction | 79.77% | 70.55% | 78.44% |
| Binding | 61.64% | **67.73%** | 64.06% |

Scaffolded Spiral preserved earlier-task answers better than the Binding-only reference at this point, but learned less Binding by the same global update. This is a **specialization–retention trade-off in predictions**, not evidence that the two schedules differ in one isolated mechanism or that one has stored better object representations.

During final Mixed consolidation (7,200 → 9,000):

| Curriculum | Shape change | Binding change |
|---|---:|---:|
| Scaffolded Spiral | +4.22 pp | −0.39 pp |
| Spiral Late Focus | +1.25 pp | −1.64 pp |
| Staged Rehearsal | +2.42 pp | −1.33 pp |
| Prerequisite-Mixed → Pair Focus | **+9.45 pp** | **−4.77 pp** |
| Fully Mixed | +2.42 pp | +1.41 pp |

The strongest Shape rebound follows the most concentrated Binding phase. Conversely, that schedule loses the largest mean Binding accuracy during consolidation. The short-term question-level forgetting rate over the first 100 mixed updates is **18.57%** for Prerequisite-Mixed → Pair Focus versus **9.39%** for Scaffolded Spiral, among previously correct ID Binding questions. Their correctness churn is **19.92%** versus **11.41%**. Again, differing entry accuracies and task histories prevent a rehearsal-only causal explanation.

The two Spiral suffix orders also differ by **+0.94 pp final ID Shape** in favor of specialization-before-rebalancing (paired interval **[+0.38, +1.49] pp**). This is a small secondary observation with multiple comparisons and an imperfect shared prefix; it should not determine an optimal schedule.

## What the errors reveal

### High Conjunction accuracy does not establish the prerequisite

The ID-dev Conjunction split contains **128 positive cases**, **76 negatives with the queried Color missing**, and **52 negatives with Color present but the queried Shape missing**, per seed. A Color-only oracle would therefore score **79.69%**.

Scaffolded Spiral scored **80.00%** overall. On the critical Color-present/Shape-missing negatives, it correctly rejected only **25.38%**; it rejected **99.21%** when Color was missing. The pattern is **strongly Color-dominant**, though not perfectly Color-only. Thus an approximately 80% Conjunction score is not persuasive evidence that the model reliably checks both attributes.

Shape also remains uneven: at the final checkpoint the accuracies by queried Shape were **80.9% circle**, **67.4% crescent**, **65.0% triangle**, and **58.5% square**.

### Aggregate Binding hides different decisions

On the original OOD-dev questions, Scaffolded Spiral achieved **59.4% positive recall** and **51.5% negative specificity**, versus **77.3% recall** and **35.6% specificity** for Fully Mixed. Spiral predicted `Yes` less often (53.95% versus 70.85%). Better negative rejection came at the cost of more missed positives.

The schedules disagreed in correctness on **292/1,280 (22.8%)** ID Binding seed-question instances and **576/2,000 (28.8%)** OOD instances. Similar overall accuracy therefore conceals substantial differences in which questions were answered correctly. Such differences do not identify whether the model changed its representation, evidence pooling or decision threshold.

The mean OOD cross-entropy was **0.793** for Scaffolded Spiral versus **0.758** for Fully Mixed. An accuracy above chance does not imply reliable probability estimates.

The OOD-dev generator has since been found to contain a scene-level label shortcut based on the presence of any held-out object. Accordingly, these OOD differences **cannot establish compositional transfer**.

## Interpretation

- **No final advantage from the full Spiral schedule:** the proposed combination of scaffolding, focused Binding and rebalancing did not exceed its references reliably.
- **Rehearsal-rich training preserved earlier-task answers better** during Binding practice, while Binding-only training reached higher intermediate Binding accuracy and lost more of it during consolidation. The compared schedules changed multiple ingredients.
- **Suffix order remained unresolved:** the point differences were small, the comparisons exploratory, and one seed did not preserve the intended prefix exactly.
- **The proposed prerequisite hierarchy was not verified:** Shape remained weak and Conjunction success was compatible with a strong Color-based shortcut.

The lasting finding is that curricula change *what the model can answer at each stage*, but a sophisticated schedule does not automatically build a reusable object-level capability or improve the final Binding endpoint.

The training and analysis are in [`scaffolded_spiral.ipynb`](../notebooks/medium_world/scaffolded_spiral.ipynb) and [`scaffolded_spiral_analysis.ipynb`](../notebooks/medium_world/scaffolded_spiral_analysis.ipynb). Saved metrics, prediction traces and paired contrasts are in [`outputs/medium_world/experiments/scaffolded_spiral/`](../outputs/medium_world/experiments/scaffolded_spiral/). The results use repeatedly inspected development sets and unadjusted exploratory contrasts; no untouched final test was conducted.
