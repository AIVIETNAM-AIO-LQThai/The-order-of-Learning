# Mastery-Based Progression: Learning and Error Analysis

*13 November 2026*

This experiment tested whether moving to the next task when a model appears to have learned its prerequisites is better than advancing at fixed update counts. The question matters because **finishing a planned stage is not the same as mastering its intended capability**.

## What was compared

All 15 runs used the same Medium World, `AttributeBindingViT` (4×4 patches, width 48, one Transformer layer), constant AdamW learning rate `1e-4`, batch size 50 and **9,000 updates**. Each training question was presented 45 times. Five paired seeds (42–46) were evaluated every 100 updates using saved predictions.

The ID-dev scenes were divided into two disjoint halves: one for progression decisions and one for evaluating behavior independently of those decisions. The three schedules were:

| Schedule | When it changed stages |
|---|---|
| `fixed_early` | Predefined steps 1,400 and 2,200 |
| `fixed_late` | Predefined steps 3,000 and 5,000 |
| `mastery` | Smoothed development-progress rules, or the maximum duration of a stage |

All schedules ended with a common Mixed training period. The experiment compared **when** transitions happened; it did not change the model, data or total task exposure.

## Final Binding performance

| Schedule | Held-out ID Binding | OOD-dev Binding | OOD Yes recall | OOD No specificity |
|---|---:|---:|---:|---:|
| Fixed early | **60.8%** | 54.5% | **67.4%** | 41.6% |
| Fixed late | 60.2% | **55.6%** | 60.0% | **51.1%** |
| Mastery | 60.5% | 54.6% | 64.1% | 45.1% |

Mastery-guided progression did **not** show a final Binding advantage over fixed timing. Only **one of five seeds** passed the first mastery gate before its stage limit, and three passed the second. The strategy often remained governed by its time limits rather than by clear attainment of the intended skills.

The OOD totals are not a clean transfer measure: the generator makes the presence of any held-out pair coincide with the Yes label at scene level. These scores are useful historical comparisons on a fixed development set, **not evidence that one schedule generalized compositionally**.

## What changed at the Binding stage?

The first 100 updates after Binding began caused substantial turnover on the same held-out ID questions:

| Schedule | Shape questions previously correct → wrong | Conjunction correct → wrong | Binding correctness churn |
|---|---:|---:|---:|
| Fixed early | 11.4% | 4.3% | 32.6% |
| Fixed late | 16.2% | 6.7% | 40.7% |
| Mastery | **19.6%** | **9.4%** | **41.3%** |

Color was more stable (0.2–3.1% newly wrong among previously correct answers). The changes show **behavioral interference** with Shape and Conjunction, but do not establish that their representations were removed.

The initial Binding updates also shifted predictions away from Yes. On the held-out ID subset, mean false-positive counts fell while false-negative counts rose:

| Schedule | False positives, before → after | False negatives, before → after |
|---|---|---|
| Fixed early | 60.8 → 43.6 | 4.2 → 19.2 |
| Fixed late | 58.8 → 37.2 | 5.8 → 19.4 |
| Mastery | 59.2 → 35.8 | 5.4 → 21.6 |

This is why aggregate Binding accuracy can improve while many individual predictions become incorrect: the model learns to reject more absent pairs but simultaneously misses more present ones.

By the final checkpoint, **65.6–78.8%** of Shape questions that had changed from correct to incorrect were correct again, depending on schedule. This is recovery of *answers on the same examples*, not proof of restored latent features.

## Which errors persisted?

| Held-out OOD target | Questions per seed | Fixed early | Fixed late | Mastery |
|---|---:|---:|---:|---:|
| Blue–Square | 97 | 55.9% | 60.8% | 58.8% |
| Green–Circle | 109 | 60.9% | 62.0% | 59.1% |
| Purple–Triangle | 111 | 50.8% | 48.6% | 50.6% |
| Red–Crescent | 83 | 49.4% | 50.1% | 49.2% |

The more successful combinations do not demonstrate reliably correct binding. Green–Circle, for example, shows very high Yes recall (roughly 82–87%) but low No specificity (roughly 33–42%). Models frequently predict that the pair exists.

On held-out ID Shape questions, **Circle** was the strongest class (around 78–80% across schedules), while **Square** was around 54–57%. Conjunction showed an even more revealing weakness: every tested model rejected all 36 negatives where the requested Color was absent, but rejected only **27–28%** of the 25 negatives where Color was present and Shape absent. A Color-presence-only rule scores roughly **80.5%** on this held-out subset. High Conjunction totals therefore do not establish dependable Shape use.

Object-count and repeated-attribute subgroups were also associated with different Binding accuracies. Those groups have different answer proportions and visual characteristics; their differences do not identify object count or repetition as the causal factor.

## Interpretation

- **Mastery timing did not improve the fixed-final Binding outcome** in this study. A gate triggered at a stage limit should not be interpreted as evidence of mastering the target concept.
- **Transitions change decisions, not merely aggregate scores.** The false-positive/false-negative trade-off and question-level churn are important outcomes even when final accuracy is similar.
- **The intended prerequisites remain weak.** Unreliable Shape recognition and Color-dominant Conjunction weaken the premise that earlier stages had established the capabilities Binding needs.

The evaluation is exploratory: five training seeds, repeated use of development examples and no untouched model-test evaluation. The OOD split has the later-identified label shortcut.

Evidence: [`mastery_progression.ipynb`](../notebooks/medium_world/mastery_progression.ipynb), [`mastery_evaluation_analysis.ipynb`](../notebooks/medium_world/mastery_evaluation_analysis.ipynb) and the saved results under [`outputs/medium_world/experiments/mastery_progression/`](../outputs/medium_world/experiments/mastery_progression/).
