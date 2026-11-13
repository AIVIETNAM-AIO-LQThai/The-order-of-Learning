# Transition rehearsal: what happens after Binding specialization?

*November 13, 2026*

## The controlled question

After a model has practiced Binding alone, should earlier tasks return abruptly, gradually, or in a way that keeps more Binding practice late? The first matched-exposure curricula suggested that the common Mixed phase could reduce newly acquired Binding accuracy. This study changed the **continuation after the same trained prefix**, rather than training each schedule independently from initialization.

For each of five seeds (42–46), one Medium World `AttributeBindingViT` was trained with Color, Shape and Conjunction mixed to update 5,400, followed by Binding alone to update **6,300**. The three suffixes then restored the **same saved model weights, AdamW state and Python/NumPy/PyTorch RNG states** for that seed. Within-task question order and final exposure counts were matched. This makes the branching comparison stronger than independently training similar prefixes.

All runs used constant LR `1e-4`, batch size 50, the same data and augmentations, and a fixed **9,000-update** endpoint. Each question appeared **45 times** by the end; the runs were evaluated on the same ID-dev and OOD-dev questions every 100 updates.

## Three continuations

| Suffix schedule | 6,300–6,600 | 6,600–7,200 | 7,200–9,000 |
|---|---|---|---|
| **Abrupt** | 100% Binding | 100% Binding | 25% Binding |
| **Gradual bridge** | 100% Binding | 75% Binding | 33⅓% Binding |
| **Late Binding** | 100% Binding | 25% Binding | 50% Binding |

The remaining percentage in each block is divided equally among Color, Shape and Conjunction. Every suffix has **27 Binding passes and nine passes per prerequisite task** after the shared branch. Final exposure is equal, but **intermediate exposure at update 7,200 is not**. Therefore, a 7,200-point difference is not a matched-Binding-exposure effect.

Saved run checks reported **five common parents**, **15 completed continuations**, matching parent hashes and prefix predictions within each seed, 91 evaluation checkpoints per run, and the expected dataset and exposure records.

## Final results

Five-seed means at update 9,000:

| Continuation | ID Binding | OOD-dev Binding | OOD positive recall | OOD negative specificity |
|---|---:|---:|---:|---:|
| Abrupt | 62.50% | 57.20% | 67.2% | 47.2% |
| Gradual bridge | **65.86%** | 58.90% | 70.7% | 47.1% |
| Late Binding | **65.86%** | **59.60%** | 72.6% | 46.6% |

Paired differences in percentage points (five seeds; unadjusted 95% Student-*t* intervals):

| Contrast | ID Binding | OOD-dev Binding |
|---|---:|---:|
| Gradual − Abrupt | **+3.36 [+1.17, +5.54]** | +1.70 [−0.86, +4.26] |
| Late − Abrupt | **+3.36 [+1.12, +5.60]** | +2.40 [−1.42, +6.22] |
| Gradual − Late | 0.00 [−2.20, +2.20] | −0.70 [−3.02, +1.62] |

Both alternatives improved final ID Binding relative to Abrupt in **all five seeds**. Their OOD mean differences are uncertain, and **Gradual does not outperform Late Binding** on the primary endpoint. The result supports changing the abrupt continuation, but it does not isolate transition smoothness as the cause. Gradual and Late also change *how recently* Binding questions are seen.

## Immediate prediction disruption

At update 7,200 the suffixes enter different mixtures, starting from different intermediate Binding exposure and performance:

| Continuation | ID Binding 7,200 → 7,300 | OOD Binding 7,200 → 7,300 | Previously correct ID Binding questions that become incorrect |
|---|---|---|---:|
| Abrupt | 67.3% → 61.9% | 59.9% → 57.8% | **18.1%** |
| Gradual bridge | 65.4% → 64.2% | 58.4% → 59.4% | 10.7% |
| Late Binding | 59.8% → 62.2% | 56.6% → 58.4% | 12.0% |

The Abrupt continuation has the largest immediate drop. Its ID correctness churn was **19.1%**, versus **12.7% Gradual** and **16.6% Late**. On OOD, churn was **20.9%, 12.5% and 12.3%**, respectively.

Of ID questions that were correct at 7,200 but wrong at 7,300, the fraction correct again by 9,000 was **50.5% Abrupt**, **64.3% Gradual**, and **38.4% Late**. These are conditional rates over different sets of forgotten questions, not a direct comparison of recovered knowledge in identical populations.

The results show **prediction instability during the task-mixture transition**. They do not prove that visual features were erased or that gradual training mechanically prevents forgetting. The suffixes receive different amounts of Binding at the transition and have different accuracies at its start.

## What the OOD errors mean

Performance on the four held-out combinations varied:

| Held-out pair | Abrupt | Gradual | Late Binding |
|---|---:|---:|---:|
| Blue–square | 60.4% | 61.6% | 62.9% |
| Green–circle | 65.0% | 67.5% | 69.4% |
| Purple–triangle | 52.3% | 52.8% | 51.7% |
| Red–crescent | 49.9% | 52.5% | 53.5% |

The OOD mean improvement mainly came from **higher positive recall**, while negative specificity remained around **47%** in all three continuations. The model therefore still made many false positive Binding predictions.

The Medium World OOD generator also makes held-out-object presence perfectly track the scene's OOD label. These numbers describe this particular confounded benchmark; they do **not** establish genuine transfer to previously unseen combinations.

## Interpretation

- **Continuation strategy matters for final ID Binding.** From identical saved parent states and equal final exposure, Gradual and Late both improved over an abrupt return to Mixed.
- **The first Mixed updates are particularly disruptive** after prolonged Binding-only practice, but intermediate comparisons have unequal Binding exposure.
- **A special benefit from smoothness is not identified.** Gradual and Late are similar at the fixed final endpoint; their differences mix transition shape with recency of Binding practice.
- **Prediction changes are not representation loss.** The evidence concerns correctness, recall and recovery on fixed question IDs.

This experiment narrowed the curriculum question from “does rehearsal help?” to **which properties of the continuation schedule matter when the model starts from exactly the same state**. The later Smoothness × Recency study was designed to separate those factors more carefully.

The completed notebooks are [`transition_rehearsal.ipynb`](../notebooks/medium_world/transition_rehearsal.ipynb) and [`transition_rehearsal_analysis.ipynb`](../notebooks/medium_world/transition_rehearsal_analysis.ipynb). The saved histories and per-question analyses are under [`outputs/medium_world/experiments/transition_rehearsal/five_seeds/`](../outputs/medium_world/experiments/transition_rehearsal/five_seeds/). All intervals and interpretations are exploratory and based on repeatedly inspected development sets.
