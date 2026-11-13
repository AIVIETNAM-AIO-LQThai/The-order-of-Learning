# Smoothness and Recency of Binding Practice

*13 November 2026*

The earlier transition-rehearsal experiment showed that redistributing practice after a Binding-focused phase could improve the final result. It did not tell us whether the benefit came from **smoother task transitions** or simply **placing Binding practice later**. This experiment compared those two scheduling properties separately.

## Controlled continuation

Five seeds (42–46) each supplied one common parent checkpoint at step **6,300**, after 5,400 updates of mixed prerequisites and 900 Binding-only updates. All four continuations restored the **same weights, full AdamW state and Python/NumPy/PyTorch RNG state** within each seed. The model, constant LR `1e-4`, dataset and final exposure budget remained unchanged.

The final 2,700 updates were divided into nine 300-update windows. Each window allocated a percentage of training to Binding; Color, Shape and Conjunction divided the remainder equally.

| Continuation | Binding share in consecutive 300-update windows (%) |
|---|---|
| Smooth / early | 80, 70, 65, 60, 50, 45, 35, 25, 20 |
| Stepwise / early | 80, 74, 74, 46, 46, 46, 32, 32, 20 |
| Smooth / late | 20, 25, 35, 45, 50, 60, 65, 70, 80 |
| Stepwise / late | 20, 32, 32, 46, 46, 46, 74, 74, 80 |

Every condition received **67,500 Binding presentations and 22,500 presentations of each other task** in the continuation, reaching 45 presentations per question by step 9,000. The within-task question streams were held fixed. Smooth and Stepwise profiles at the same recency level also matched total exposure, first/last window percentages, and the **mean timing of Binding practice**. Their maximum adjacent-window change was **10 versus 28 percentage points**. The code used mirrored within-window task shuffles to keep the timing centroid matched, not just matched in expectation.

The Early and Late versions shifted mean Binding presentation time from step **7,350** to **7,950**. That change necessarily also alters which task dominates immediately after the shared checkpoint. “Smoothness” is defined at the **300-update window resolution**, not as a per-gradient measure.

## Final results

| Continuation | ID Binding | OOD Binding | OOD Yes recall | OOD No specificity |
|---|---:|---:|---:|---:|
| Smooth / early | **65.47%** | 58.70% | 66.9% | 50.5% |
| Stepwise / early | 65.00% | 59.10% | 65.6% | **52.6%** |
| Smooth / late | 65.39% | 59.35% | **71.5%** | 47.2% |
| Stepwise / late | 65.23% | **59.80%** | **71.5%** | 48.1% |

At matched recency, the paired Smooth-minus-Stepwise ID difference was **+0.47 pp** (95% CI [−1.39, +2.33]) for Early and **+0.16 pp** ([−1.68, +1.99]) for Late. Averaged across the two recency settings, the difference was **+0.31 pp** (CI [−0.98, +1.60]). None establishes a repeatable final advantage from the smoother profile.

Moving practice later gave an average **+0.08 pp ID** (CI [−1.50, +1.66]) and **+0.68 pp OOD** (CI [−3.27, +4.62]). The smoothness-by-recency interaction was also unresolved. These intervals are exploratory five-seed paired comparisons, not equivalence tests.

## The learning curves tell a more specific story

The OOD question-level traces show a change in **when** answers are lost and recovered:

| Continuation | Correct at 6,300 → wrong at 9,000 | Wrong at 6,300 → correct at 9,000 | Correctness churn |
|---|---:|---:|---:|
| Smooth / early | 16.9% | 25.6% | 20.6% |
| Stepwise / early | 16.9% | 26.4% | 20.9% |
| Smooth / late | 12.7% | 21.6% | 16.4% |
| Stepwise / late | 12.3% | 22.1% | 16.5% |

Late schedules ended with **fewer formerly correct answers lost**, but also converted **fewer formerly wrong answers into correct ones**. These opposing changes explain why the net final accuracy effect is small.

At step 7,200, Late schedules had temporarily forgotten more previously correct OOD Binding answers (**17.4–19.7%** versus **10.7–11.6%** for Early). They later recovered a greater share of those interim losses (**56.3–58.3%** versus **38.9–41.1%**). The data show an effect on prediction stability and timing, not a measured loss or restoration of stored visual concepts.

Color, Shape and Conjunction accuracy improved across the continuation under all four policies. However, neither final accuracy nor the question-level traces isolate a special advantage of smoother transitions.

## Interpretation

The earlier Abrupt-versus-Gradual improvement cannot be reduced to **window-level smoothness** or **mean Binding recency alone**. In this more tightly matched factorial comparison, neither factor produced a reliable final Binding gain. The previous experiment also used an extreme 100%-Binding-to-25%-Binding switch that these smoother profiles do **not** reproduce, so the null comparison does not cover every transition shape.

Later Binding practice did change the path of forgetting and recovery, and it shifted Yes recall upward at the expense of negative specificity. The OOD evaluation itself has a known scene-level label shortcut and cannot establish true novel-combination transfer. These observations concern the tested training schedules on a fixed synthetic benchmark, not a universal optimal teaching order.

Evidence: [`smoothness_recency.ipynb`](../notebooks/medium_world/smoothness_recency.ipynb), [`smoothness_recency_analysis.ipynb`](../notebooks/medium_world/smoothness_recency_analysis.ipynb), and the saved [five-seed traces and tables](../outputs/medium_world/experiments/smoothness_recency/five_seeds/).
