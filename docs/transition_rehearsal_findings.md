# Transition Rehearsal — Five-Seed Findings

**Project:** The Order of Learning · Medium World  
**Sources:** Executed `transition_rehearsal(1).ipynb`, `transition_rehearsal_analysis(1).ipynb`, and `paired_differences(1).csv`  
**Scope:** ID-dev and OOD-dev; five paired seeds (42–46); no test-set evaluation.

## Experimental setting

The existing `AttributeBindingViT` (dimension 48, patch size 4, one Transformer layer) is trained on the frozen Medium World dataset: 10,000 training questions across Color, Shape, Conjunction, and same-object Binding (2,500 per task). Training uses AdamW (learning rate `1e-4`, weight decay 0), batch size 50, image augmentation, and 9,000 updates. Each exact training question is presented 45 times. Evaluation occurs every 100 updates, with 256 ID-dev and 400 OOD-dev Binding questions.

For each seed, a *single parent* trains with Color/Shape/Conjunction mixed through step 5,400, then Binding through step 6,300. Three continuations restore **identical model weights, optimizer state, and random-number-generator state** at step 6,300. They also preserve the within-task question order and final per-question exposure.

| Continuation | Steps 6,300–6,600 | 6,600–7,200 | 7,200–9,000 |
|---|---|---|---|
| **Abrupt** | Binding only | Binding only | Uniform four-task mix |
| **Gradual bridge** | Binding only | 75% Binding, 25% earlier tasks | 33⅓% Binding, 66⅔% earlier tasks |
| **Late Binding** | Binding only | 25% Binding, 75% earlier tasks | 50% Binding, 50% earlier tasks |

The earlier-task percentage is divided evenly among Color, Shape, and Conjunction. Each continuation uses 27 Binding passes and nine passes per earlier task after the branch, so **total exposure is identical at step 9,000**. Exposure differs at intermediate checkpoints.

**Integrity:** The executed notebooks report 15 completed runs, five distinct shared parents, matching parent hashes and prefix predictions within each seed, verified dataset hashes, 91 checkpoints per run, and the required final exposure counts.

## Final Binding performance

Mean across the five seeds at step 9,000; the ± value is between-seed standard deviation.

| Continuation | ID accuracy | OOD accuracy | OOD positive recall | OOD negative specificity |
|---|---:|---:|---:|---:|
| Abrupt | 62.50% ± 1.85 pp | 57.20% ± 2.69 pp | 67.2% | 47.2% |
| Gradual bridge | **65.86% ± 2.61 pp** | 58.90% ± 2.68 pp | 70.7% | 47.1% |
| Late Binding | **65.86% ± 2.00 pp** | **59.60% ± 2.39 pp** | **72.6%** | 46.6% |

Paired differences (percentage points; five seeds; 95% *t* intervals):

| Contrast | ID difference [95% CI] | OOD difference [95% CI] |
|---|---:|---:|
| Gradual − Abrupt | **+3.36 [+1.17, +5.54]** | +1.70 [−0.86, +4.26] |
| Late − Abrupt | **+3.36 [+1.12, +5.60]** | +2.40 [−1.42, +6.22] |
| Gradual − Late | +0.00 [−2.20, +2.20] | −0.70 [−3.02, +1.62] |

Both alternatives outperform Abrupt on **final ID Binding in all five seeds**. Late Binding also exceeds Abrupt on OOD in all five, while Gradual does so in four. However, the OOD paired intervals include zero. **There is no resolved difference between Gradual and Late Binding at the final endpoint.**

## Behavior around step 7,200

At step 7,200, the schedules have **different cumulative Binding exposure**; these comparisons characterize their trajectories, not an exposure-controlled treatment effect.

| Continuation | ID: 7,200 → 7,300 | OOD: 7,200 → 7,300 | Previously correct ID questions becoming incorrect | Same measure, OOD |
|---|---|---|---:|---:|
| Abrupt | 67.3% → 61.9% | 59.9% → 57.8% | 18.1% | 19.0% |
| Gradual bridge | 65.4% → 64.2% | 58.4% → 59.4% | 10.7% | 9.9% |
| Late Binding | 59.8% → 62.2% | 56.6% → 58.4% | 12.0% | 9.5% |

The abrupt switch exhibits the largest immediate drop and the highest question-level correctness churn (19.1% ID and 20.9% OOD). Gradual bridge has lower churn (12.7% ID, 12.5% OOD); Late Binding has 16.6% ID and 12.3% OOD churn. Of questions that were correct at 7,200 and incorrect at 7,300, the fractions correct again by 9,000 are 50.5%/33.6% (Abrupt, ID/OOD), 64.3%/45.5% (Gradual), and 38.4%/32.2% (Late). These are descriptive conditional rates, with different initial-error denominators across schedules.

The final OOD gains are mainly associated with **higher positive recall**, not better negative rejection: specificity stays around 47% for all three continuations. The model still struggles to reject many nonexistent color–shape pairs.

## OOD held-out combinations

Pooled final accuracy across the five paired seeds. Subgroup support and labels are identical between continuations.

| Held-out pair | Abrupt | Gradual | Late Binding |
|---|---:|---:|---:|
| Blue–square | 60.4% | 61.6% | 62.9% |
| Green–circle | 65.0% | 67.5% | 69.4% |
| Purple–triangle | 52.3% | 52.8% | 51.7% |
| Red–crescent | 49.9% | 52.5% | 53.5% |

The improvements are uneven. Green–circle contributes clearer gains; purple–triangle and red–crescent remain much harder. This experiment improves the *timing* of Binding practice without establishing robust compositional generalization across held-out combinations.

## Interpretation

1. **Reordering the post-Binding continuation matters.** Under equal final exposure and identical parent states, both Gradual and Late Binding improve final ID Binding over an abrupt return to uniformly mixed tasks.
2. **The abrupt transition produces more immediate prediction disruption.** The lower forgetting and churn under the alternatives are consistent with reduced short-term interference, although the strategies have different Binding exposure at step 7,200.
3. **A uniquely beneficial gradual-transition mechanism is not established.** Gradual and Late Binding have equal mean final ID accuracy, similar OOD accuracy, and different late-stage task proportions. The current comparisons cannot separate smoothness from recency of Binding exposure.
4. **OOD Binding remains limited.** Most OOD improvement reflects positive recall, while hard-negative specificity and performance on difficult held-out combinations remain weak.

**Next methodological priority:** Analyze Color/Shape/Conjunction retention under these continuations, then design a tighter smoothness-versus-recency control that matches task proportions in comparable late-training windows. Preserve the current runs and their fixed-final endpoints as the reference.
