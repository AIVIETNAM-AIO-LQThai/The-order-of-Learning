# Mastery Progression — Evaluation and Error Analysis

**Project:** The Order of Learning  
**Experiment:** Trace-enabled Medium World mastery progression  
**Source:** Executed `mastery_evaluation_analysis.ipynb` (6 November 2026), with the training configuration from `mastery_progression.ipynb`  
**Status:** 15 completed runs; 15 matching prediction traces

## 1. Experiment

Medium World contains 64×64 RGB scenes with 3–5 objects, five colors, and four shapes. Four binary tasks test Color, Shape, independent Color-and-Shape presence (**Conjunction**), and whether both attributes belong to the **same object** (**Binding**). Binding negatives contain both requested attributes on different objects. OOD questions query four color–shape pairs excluded from training.

The model is the same randomly initialized `AttributeBindingViT` in every run: 4×4 patches, 48-dimensional features, one Transformer encoder layer (four attention heads), and a query-conditioned head that pools patch-level attribute and joint evidence.

| Setting | Protocol |
|---|---|
| Data | 10,000 training questions from 5,000 scenes; 1,024 ID-dev questions; 400 OOD-dev Binding questions |
| ID-dev use | Scene-disjoint halves: 256 scenes for progression decisions and 256 for held-out analysis |
| Training | Five paired seeds (42–46); AdamW; constant learning rate `1e-4`; batch size 50; 9,000 updates |
| Exposure | 45 presentations per training question; 7,200 main updates followed by 1,800 common mixed updates |
| Schedules | `fixed_early` advances at steps 1,400/2,200; `fixed_late` at 3,000/5,000; `mastery` advances on smoothed progress criteria or stage limits |
| Evaluation | Every 100 updates; per-question labels, predictions and probabilities saved for ID-dev and OOD-dev |

All percentages below are **means across five seeds** from the trace-enabled runs. These results are kept separate from the earlier mastery experiment without prediction traces.

## 2. Final Binding performance

| Schedule | Held-out ID accuracy | OOD accuracy | OOD positive recall | OOD negative specificity |
|---|---:|---:|---:|---:|
| `fixed_early` | 60.8% | 54.5% | 67.4% | 41.6% |
| `fixed_late` | 60.2% | **55.6%** | 60.0% | **51.1%** |
| `mastery` | 60.5% | 54.6% | 64.1% | 45.1% |

The curricula end at similar accuracy. Adaptive progression does not improve held-out ID or OOD Binding over fixed timing. Fixed-late has the highest mean OOD accuracy, but the difference is small.

The error breakdown explains why accuracy alone is insufficient. On OOD, fixed-early recognizes more positive pairs than fixed-late (67.4% versus 60.0%) but rejects fewer negative pairs correctly (41.6% versus 51.1%). The three curricula therefore arrive at similar accuracy through different error balances. Their mean OOD predicted-positive rates are 62.9%, 54.4%, and 59.5%, respectively.

The held-out ID results also favor positive recall over negative specificity: `fixed_early` **71.3% / 52.6%**, `fixed_late` **67.2% / 54.7%**, and `mastery` **69.1% / 53.8%**. This imbalance persists at the final checkpoint.

## 3. Changes when Binding training begins

The next table compares the evaluation just before the Binding stage with the first evaluation afterward. *Forgotten* means correct → incorrect on the **same question**; *churn* counts both correct → incorrect and incorrect → correct. The held-out ID split is used throughout.

| Schedule | Color forgotten | Shape forgotten | Conjunction forgotten | Binding correctness churn |
|---|---:|---:|---:|---:|
| `fixed_early` | 0.2% | 11.4% | 4.3% | 32.6% |
| `fixed_late` | 0.9% | 16.2% | 6.7% | 40.7% |
| `mastery` | 3.1% | 19.6% | 9.4% | 41.3% |

Color is mostly preserved, while Shape and Conjunction lose more previously correct answers. Mastery shows the largest average disruption at this transition. The schedules enter Binding at different steps, so these numbers describe their actual training trajectories rather than a comparison at identical exposure histories.

Mean held-out Binding accuracy **increases** over the same first 100 updates by **1.82**, **6.61**, and **5.95 percentage points**, respectively. Yet roughly one-third to two-fifths of its questions switch correctness. Since Binding has not previously received its dedicated training stage, these switches measure **prediction turnover**, not forgetting of an established Binding skill.

## 4. What changes in the errors?

Immediately after entering Binding, all three schedules produce **fewer false positives but more false negatives**. On held-out ID Binding, mean counts per seed change as follows:

| Schedule | False positives, before → after | False negatives, before → after |
|---|---:|---:|
| `fixed_early` | 60.8 → 43.6 | 4.2 → 19.2 |
| `fixed_late` | 58.8 → 37.2 | 5.8 → 19.4 |
| `mastery` | 59.2 → 35.8 | 5.4 → 21.6 |

Before Binding specialization, the model tends to answer **Yes** to many Binding questions. The first Binding updates make it more selective: negative specificity improves while positive recall falls. This shift explains why aggregate Binding accuracy can improve while many previously correct answers become wrong.

A similar positive-to-negative shift occurs on OOD Binding, where all 400 questions are Binding questions. Later training and common mixed consolidation bring the schedules toward more similar positive-recall and negative-specificity trajectories.

## 5. Recovery and scene characteristics

By the final checkpoint (step 9,000), some questions forgotten immediately after entering Binding are correct again:

| Task | `fixed_early` | `fixed_late` | `mastery` |
|---|---:|---:|---:|
| Shape | 65.6% | 77.5% | 78.8% |
| Conjunction | 63.0% | 55.2% | 62.4% |
| Binding prediction reversals | 58.9% | 46.4% | 47.6% |

These percentages are calculated **among questions that first changed from correct to incorrect**. They track recovery of those specific answers rather than merely a return of aggregate accuracy.

For held-out ID Binding, final accuracy declines from **61–64%** on three-object scenes to about **59%** on five-object scenes. On OOD Binding, accuracy instead ranges from **52–53%** with three objects to **55–57%** with five. Object count alone does not explain the ID–OOD difference.

## 6. Subgroup error analysis

Source: the complete `prediction_error_subgroups.csv`, using the final checkpoint (step 9,000). Each percentage is the **mean across five seeds**, with the same evaluation questions used for each seed. The tables do not treat repeated predictions on those questions as independent samples.

### Unseen color–shape combinations

| OOD Binding target | Questions per seed | `fixed_early` | `fixed_late` | `mastery` |
|---|---:|---:|---:|---:|
| Blue–square | 97 | 55.9% | 60.8% | 58.8% |
| Green–circle | 109 | **60.9%** | **62.0%** | **59.1%** |
| Purple–triangle | 111 | 50.8% | 48.6% | 50.6% |
| Red–crescent | 83 | 49.4% | 50.1% | 49.2% |

Generalization varies by pair. Green–circle and blue–square are recognized more successfully, whereas purple–triangle and red–crescent remain near chance under every schedule. Green–circle's higher accuracy is accompanied by **82–87% positive recall** but only **33–42% negative specificity**: the models frequently answer Yes, including when the target pair is absent. This is not uniformly reliable detection of the unseen combination.

Because each held-out color is paired with one particular shape, these results cannot separate the effects of color, shape, and their combination.

### Shape recognition and Conjunction

The held-out ID **Shape** task exhibits a consistent accuracy difference by queried shape:

| Shape | `fixed_early` | `fixed_late` | `mastery` |
|---|---:|---:|---:|
| Circle | 80.0% | 79.4% | 78.2% |
| Crescent | 62.9% | 61.4% | 64.3% |
| Square | 56.7% | 54.4% | 54.4% |
| Triangle | 60.0% | 57.4% | 61.9% |

Circles are learned most reliably; squares remain especially difficult. This uneven attribute recognition is relevant to later Binding performance, which depends on recognizing both the queried color and shape.

The held-out ID **Conjunction** task exposes a more specific weakness:

| Question case | Questions per seed | `fixed_early` | `fixed_late` | `mastery` |
|---|---:|---:|---:|---:|
| Negative: requested color absent | 36 | 100.0% | 100.0% | 100.0% |
| Negative: color present, requested shape absent | 25 | 28.0% | 27.2% | 27.2% |
| Positive: both attributes present | 67 | 88.7% | 89.0% | 88.7% |

All models reject every Conjunction question where the color is absent, yet often answer Yes when the color is present and the shape is missing. Their behavior closely follows the **color-only shortcut** identified in the dataset audit: checking color without consistently checking shape can achieve **80.5%** on this held-out subset. Aggregate Conjunction accuracy therefore overstates evidence of reliably recognizing both attributes.

### Scene complexity and repeated attributes

The OOD Binding subgroup analysis shows:

| Scene property | `fixed_early` | `fixed_late` | `mastery` |
|---|---:|---:|---:|
| 3 objects | 51.7% | 52.9% | 52.4% |
| 4 objects | 55.9% | 55.9% | 55.0% |
| 5 objects | 54.7% | 56.6% | 55.4% |
| No repeated shapes | 41.8% | 44.9% | 42.1% |
| Repeated shapes present | 56.6% | 57.3% | 56.7% |
| No repeated colors | 49.6% | 51.1% | 50.7% |
| Repeated colors present | 55.9% | 56.8% | 55.7% |

Accuracy is not consistently worse in scenes with more objects. The lower performance on scenes without repeated shapes is notable, but does **not** mean that repetition helps the model: the two groups differ in their positive-label proportions (**38.6% versus 51.9%**) and in other scene characteristics. These are useful error categories for future controlled comparisons, not isolated causal effects.

## 7. Interpretation

1. **Mastery-based timing has not improved final Binding performance.** One of five seeds passes the first mastery gate before its stage limit, and three pass the second gate.
2. **Binding training changes the balance of errors.** Initially, false positives decrease while false negatives increase. Aggregate accuracy alone hides the substantial turnover in individual predictions.
3. **Retention differs by task.** Color stays relatively stable; Shape and Conjunction lose more previously correct answers. Many recover by the final checkpoint.
4. **Generalization differs across unseen pairs.** Green–circle and blue–square perform better than purple–triangle and red–crescent, but positive recall and negative specificity reveal unequal error patterns even for the stronger pairs.
5. **Shape understanding and Conjunction remain bottlenecks.** Squares are difficult to recognize, and Conjunction predictions strongly favor the presence of the queried color over checking the requested shape. These results help explain why aggregate success on simpler tasks does not imply reliable same-object Binding.

## 8. Saved evidence

All files are under `outputs/medium_world/experiments/mastery_progression/`:

- `schedule_comparison.json` and `prediction_traces/*.npz`: training histories and per-question predictions.
- `mastery_transition_accuracy.csv`: accuracy changes around stage boundaries.
- `prediction_change_summary.csv` and `prediction_changes_by_example.csv.gz`: retention, forgetting, new learning, and recovery.
- `prediction_error_checkpoints.csv`: confusion matrices and class-specific metrics over training.
- `prediction_error_subgroups.csv`: complete final and transition-checkpoint results by object count, repeated attributes, queried attributes, color–shape pair, and question case.
- `prediction_error_transitions.csv`: false-positive and false-negative changes at stage transitions.
- `prediction_binding_positive_recall.png`, `prediction_binding_negative_specificity.png`, and `prediction_binding_by_object_count.png`: visual summaries.
