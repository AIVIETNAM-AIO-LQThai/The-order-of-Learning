# Phase-Aware Learning Rates: Findings and Limits

*13 November 2026*

The earlier curriculum study used an adaptive LR scheduler that reacted differently to different task orders. When Binding arrived late, the optimizer might already have reduced its learning rate substantially. This experiment tested whether **continuing a decayed schedule** or **restarting it at a real task-mixture transition** changes Binding acquisition. Constant LR was the essential reference: a restart is not useful merely because it beats an unnecessarily weak global decay.

## Comparison and controls

Four fixed curricula (`prereq_mixed_pair_focus`, `staged_rehearsal`, `scaffolded_spiral`, and `mixed`) were crossed with five LR policies and five seeds (42–46), giving **100 completed runs**. Each used the same Medium World, the unchanged `AttributeBindingViT`, AdamW, batch size 50, and **9,000 optimizer updates** with **45 presentations of every question**. Model weights and optimizer moments were *not* reset at phase boundaries.

| LR policy | Behavior |
|---|---|
| **Constant** | `1e-4` at every update |
| **Global Cosine** | Cosine from `1e-4` to `1e-6` across all 9,000 updates |
| **Reset Cosine** | New cosine segment when the task mixture actually changes, peaking at `1e-4` (`5e-5` for final Mixed) |
| **Global Plateau** | Overall ID-dev cross-entropy; factor 2/3, patience two evaluation calls, absolute threshold 0.001, minimum `1e-6` |
| **Reset Plateau** | Same monitor, but restore LR and reset the running plateau history at genuine task-mixture changes |

Restarts occurred at steps **5,400 and 7,200** for Prereq-Mixed; **3,000, 4,950 and 7,200** for Staged Rehearsal; and **2,800, 4,450, 6,100 and 7,200** for Scaffolded Spiral. Fully Mixed had **no true restarts**. In particular, the Prereq-Mixed labels `pair_early` and `pair_focus` mark a bookkeeping split at 6,300, **not** a change in training mixture; there was no reset there.

Plateau was checked using the **overall ID-dev CE**, not loss on the currently trained task. Reset-policy checks were phased relative to the latest real boundary, while global policies continued on the global clock. No policy selected a best checkpoint as its final outcome; the main comparison was at **step 9,000**.

## Final results

The table gives five-seed mean **ID-dev / OOD-dev Binding accuracy** at the final checkpoint:

| Curriculum | Constant | Global Cosine | Reset Cosine | Global Plateau | Reset Plateau |
|---|---:|---:|---:|---:|---:|
| Prereq-Mixed → Pair | **62.89 / 57.20** | 54.69 / 55.05 | 57.58 / 56.60 | 51.41 / 52.35 | 58.20 / 57.35 |
| Staged Rehearsal | **63.28 / 56.20** | 57.03 / 53.20 | 59.06 / 56.35 | 53.12 / 50.70 | 59.92 / 56.40 |
| Scaffolded Spiral | **61.17 / 56.15** | 56.80 / 53.55 | 58.28 / 56.80 | 53.12 / 52.35 | 60.55 / 56.80 |
| Fully Mixed | **62.03 / 56.65** | 56.95 / 55.10 | 56.95 / 55.15 | 58.98 / 54.85 | 58.36 / 55.15 |

**Constant LR is best on mean final ID Binding in every curriculum.** Restarts often outperform uninterrupted decay, particularly Global Plateau, but they do not outperform Constant on this primary outcome.

For example, Reset Plateau minus Global Plateau improves final ID Binding by **+6.80 pp** for Prereq-Mixed (paired 95% CI [+4.17, +9.42]), **+6.80 pp** for Staged Rehearsal ([+2.70, +10.89]), and **+7.42 pp** for Scaffolded Spiral ([+1.59, +13.25]). Reset Cosine improves Prereq-Mixed over Global Cosine by **+2.89 pp** ([+0.76, +5.02]). But Prereq-Mixed Reset Cosine is **−5.31 pp** versus Constant (CI [−10.35, −0.28]).

Fully Mixed is a negative control because its schedule has no genuine task-mixture transition. Global and Reset Cosine specify identical LR trajectories there, and their five-seed means are both **56.95%** ID Binding. Small numerical discrepancies in run predictions cannot be interpreted as a restart effect.

## Why decay before Binding matters

Prereq-Mixed does not train Binding directly until update **5,400**. Its mean ID Binding trajectory was:

| Policy | 5,400 | 7,200 | Final 9,000 |
|---|---:|---:|---:|
| Constant | 50.16% | **67.50%** | **62.89%** |
| Global Cosine | 50.62% | 57.89% | 54.69% |
| Global Plateau | 50.62% | 50.94% | 51.41% |
| Reset Cosine | 50.55% | 60.94% | 57.58% |
| Reset Plateau | 50.62% | 62.19% | 58.20% |

Under Global Plateau, the mean LR was already near **4×10⁻⁶** at Binding introduction; some seeds were at the `1e-6` floor. Little Binding learning followed. This is not proof that the model lacked the capacity to learn Binding: the constant-LR counterpart acquired substantially more under the same task schedule.

The logged sum of nominal **LR × Binding-question presentations** was **11.250** (Constant), **2.063** (Global Cosine), **5.119** (Reset Cosine), **0.203** (Global Plateau), and **6.108** (Reset Plateau). Reset policies therefore delivered much more nominal optimization opportunity to late Binding examples than the corresponding decayed-global policies.

This exposure summary is useful, but **not a measurement of parameter change**. It omits gradients, AdamW moments, clipping, and the location of examples in parameter space. Reset Plateau also clears the best-loss and bad-count history, so this study does not isolate a pure LR boost from scheduler-history resetting.

A later analysis of the saved, **count-controlled Binding predictions** found that final discrimination across the 20 curriculum×policy groups correlated with nominal LR-weighted Pair exposure, especially **after step 5,400** (descriptive Spearman ρ around **0.88** for later exposure, versus **0.29** for earlier exposure; total ρ about **0.89**). This is a retrospective association across policies, not an independent causal experiment. Earlier stages, effective LR and optimizer history still covary.

## What happens to earlier tasks?

On switching Prereq-Mixed to Binding (5,400 → 5,500), Constant LR changed mean ID Shape by **−7.19 pp**, Conjunction by **−8.20 pp** and Binding by **+2.27 pp**. Global Plateau changed them by only **−0.78, −0.55 and −0.23 pp**, respectively. The latter's apparent retention is misleading: its LR was so small that it barely improved the new task.

The final Mixed phase can reverse the pattern. In the first 100 updates after 7,200, Prereq-Mixed Constant lost about **5.23 pp Binding** while gaining **7.42 pp Shape**. Reset Cosine lost **2.73 pp Binding** while gaining **5.31 pp Shape**. These are changes in prediction accuracy, not proof of internal feature deletion or recovery.

The optimization intervention also did **not** repair the model's underlying difficulties. Final Shape accuracy averaged **67.32% Constant**, **64.16% Reset Cosine**, and **64.82% Reset Plateau** across curricula. On the 52 Conjunction negatives with Color present but Shape absent, correct rejection remained weak: approximately **26.25% Constant**, **15.19% Reset Cosine**, and **17.69% Reset Plateau**. A Color-only classifier can score **79.69%** on the standard ID Conjunction set without learning to check Shape reliably.

## OOD interpretation

Held-out OOD Binding accuracy is exploratory for two reasons. First, the generator makes the **presence of any held-out Color–Shape object coincide with the Yes label** in this split. Second, the original distributions differ in scene complexity. The OOD score therefore does not isolate generalization to the *queried* unseen composition.

Across the LR policies, higher OOD Yes recall frequently accompanied lower No specificity. For instance, Constant averaged **66.30% recall / 46.80% specificity**, while Reset Cosine averaged **72.35% / 40.10%**. Even where total OOD accuracy improved, the classifier could simply have become more willing to answer Yes. Reset Cosine's lower OOD cross-entropy does not by itself establish calibrated or more compositional predictions.

## Interpretation

1. **Restarts help relative to some poorly timed global decay policies.** The biggest improvements occur when a late task would otherwise receive very little effective learning rate.
2. **There is no demonstrated advantage over Constant LR.** The policies should be compared with a well-performing non-decaying reference, not only with each other.
3. **The mechanism is coupled.** LR magnitude, its timing, optimizer history and (for Plateau) monitor-history resetting are not independently manipulated.
4. **The visual task bottleneck remains.** Shape recognition, missing-Shape Conjunction negatives and count-controlled Binding cannot be solved by an LR policy alone.

The comparisons involve five seeds, repeatedly inspected development splits, many exploratory contrasts and some numerical divergence between nominally identical runs. The later representation and diagnostic reports provide behavioral and probe-level constraints on the explanation; they do not turn this LR-policy comparison into a causal isolation of internal mechanism.

Evidence: [`lr_reset.ipynb`](../notebooks/medium_world/lr_reset.ipynb), [`lr_reset_analysis.ipynb`](../notebooks/medium_world/lr_reset_analysis.ipynb), saved [phase-aware LR results](../outputs/medium_world/experiments/phase_aware_lr/), [count-controlled follow-up](diagnostic_followup_findings.md), and [`src/medium_visual_world/lr_policies.py`](../src/medium_visual_world/lr_policies.py).
