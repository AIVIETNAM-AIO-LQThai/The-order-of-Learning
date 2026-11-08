# Scaffolded Spiral — Five-Seed Findings

## Research question

Can a **scaffolded, rehearsed Binding curriculum** improve final same-object Binding and its generalization without sacrificing Color, Shape, and Conjunction? Does **Binding specialization followed by rebalancing** outperform the same two blocks in the opposite order?

The combined schedule was motivated by earlier curriculum, transition, and rehearsal studies. Its components are **not** separately identified by a comparison of complete schedules.

## Setup and experimental integrity

- **World:** frozen Medium World, 64×64 RGB scenes; 10,000 training questions (2,500 each for Color, Shape, Conjunction, Binding). ID-dev has 256 questions per task; OOD-dev contains 400 Binding questions covering held-out color–shape combinations.
- **Architecture:** unchanged `AttributeBindingViT(dim=48, patch_size=4, depth=1)`. AdamW, constant `lr=1e-4`, weight decay 0, batch 50, gradient clipping 1.0, image augmentation enabled. **No LR reset or architecture modification was tested here.**
- **Design:** five paired seeds, 42–46, across five curricula; **25 completed runs**, all at 9,000 updates. Every exact training question is presented **45 times** (450,000 total presentations per run). Per-task question streams and the final nine-pass Mixed block are matched by seed.
- **Measurements:** step 0, every 100 updates, phase boundaries and 100 updates after boundaries. Main outcome: **ID-dev Binding at the fixed step-9,000 endpoint**; OOD-dev and other diagnostics are secondary. No test split was evaluated for this report.

| Curriculum | Main training before common Mixed (steps 7,200–9,000) |
|---|---|
| `scaffolded_spiral` | Color/Shape jointly to 2,800 → Conjunction + prerequisite rehearsal to 4,450 → Binding-heavy rehearsal to 6,100 → broader rebalancing to 7,200 |
| `spiral_late_focus` | Identical planned first 4,450 steps → rebalancing to 5,550 → Binding-heavy rehearsal to 7,200 |
| `staged_rehearsal` | Attributes → Conjunction with rehearsal → Binding with rehearsal |
| `prereq_mixed_pair_focus` | Color/Shape/Conjunction mixed to 5,400 → Binding-only to 7,200 |
| `mixed` | All four tasks mixed throughout |

For Scaffolded Spiral, Binding is **absent** until step 4,450, then comprises **78.8%** of the Binding-specialization phase and **45.5%** of rebalancing. The last 1,800 updates are uniformly mixed across tasks in every curriculum. These allocations are fixed **time-based stages**, not mastery-triggered transitions.

## Final Binding results

Means over five seeds, at the predefined final checkpoint (percent accuracy; AULC is Binding accuracy integrated over all 9,000 optimizer steps):

| Curriculum | ID-dev Binding | OOD-dev Binding | ID Binding AULC |
|---|---:|---:|---:|
| Prerequisite-Mixed → Pair Focus | **62.97%** | **57.50%** | 54.99% |
| Staged Rehearsal | 62.73% | 56.65% | 55.17% |
| Fully Mixed | 61.88% | 56.45% | **56.76%** |
| **Scaffolded Spiral** | **61.25%** | **55.45%** | **54.94%** |
| Spiral Late Focus | 60.78% | 55.90% | 54.73% |

Paired differences in percentage points, five paired seeds, two-sided 95% *t* intervals (unadjusted for multiple comparisons):

| Contrast (Scaffolded Spiral − reference) | ID Binding [95% CI] | OOD Binding [95% CI] |
|---|---:|---:|
| Staged Rehearsal — **prespecified primary** | −1.48 [−4.42, +1.45] | −1.20 [−4.19, +1.79] |
| Spiral Late Focus | +0.47 [−1.51, +2.45] | −0.45 [−2.01, +1.11] |
| Prerequisite-Mixed → Pair Focus | −1.72 [−4.80, +1.36] | −2.05 [−7.88, +3.78] |
| Fully Mixed | −0.63 [−3.63, +2.38] | −1.00 [−8.26, +6.26] |

**No fixed-final Binding contrast demonstrates a repeatable Scaffolded Spiral advantage.** The primary ID difference favors Staged Rehearsal in mean, but its paired interval spans zero. With only five seeds and several comparisons, intervals should be read descriptively, not as decisive evidence of equality or inferiority.

Global-step Binding AULC is **1.83 pp lower than Fully Mixed** for Scaffolded Spiral (paired interval [−3.25, −0.41]; all five seeds lower). This is an end-to-end efficiency result: Fully Mixed receives Binding questions from step 0, while Spiral does not see them until step 4,450. Therefore AULC **does not isolate post-introduction learning efficiency**.

## What happened at the learning stages?

### The prerequisite stage did not establish Shape mastery

Immediately before Binding introduction in Scaffolded Spiral (step 4,450), five-seed ID-dev mean accuracies were:

| Color | Shape | Conjunction | Untrained Binding |
|---:|---:|---:|---:|
| 99.22% | **63.59%** | 79.92% | 50.39% |

The schedule successfully defers Binding exposure, but **passing a prerequisite stage is not evidence that its intended skill has been mastered**. Shape remains weak and the Conjunction score conceals an important shortcut (below).

### Rehearsal retains earlier-task performance during specialization

At step 7,200, immediately before common consolidation:

| ID-dev task | Scaffolded Spiral | Prerequisite-Mixed → Pair Focus | Staged Rehearsal |
|---|---:|---:|---:|
| Color | 98.83% | 94.30% | 98.36% |
| Shape | **63.83%** | 57.34% | **64.61%** |
| Conjunction | **79.77%** | 70.55% | 78.44% |
| Binding | 61.64% | **67.73%** | 64.06% |

The pure Binding-focus reference reaches higher intermediate Binding accuracy while earlier tasks deteriorate. Scaffolded Spiral preserves substantially more of Color, Shape, and Conjunction, but acquires less Binding by the same global step. **This is a specialization–retention trade-off**, not proof that rehearsal alone causes the difference: these complete curricula also vary in earlier exposure and timing. Staged Rehearsal achieves a somewhat better Binding/Shape combination, indicating that rehearsal quantity and placement matter.

The shared 1,800-step Mixed consolidation changes mean ID-dev scores as follows (step 7,200 → 9,000):

| Curriculum | Shape change | Binding change |
|---|---:|---:|
| Scaffolded Spiral | +4.22 pp | −0.39 pp |
| Spiral Late Focus | +1.25 pp | −1.64 pp |
| Staged Rehearsal | +2.42 pp | −1.33 pp |
| Prerequisite-Mixed → Pair Focus | **+9.45 pp** | **−4.77 pp** |
| Fully Mixed | +2.42 pp | +1.41 pp |

Common consolidation restores earlier-task performance most strongly after pure Binding focus, where those skills had fallen furthest. It may also reduce Binding performance. These are accuracy and question-correctness changes, **not direct measurements of erased or recovered internal representations**.

Immediately after the step-7,200 transition (through step 7,300), **18.57%** of previously correct ID Binding questions become incorrect under Prerequisite-Mixed → Pair Focus, versus **9.39%** under Scaffolded Spiral. The corresponding correctness churn is **19.92% versus 11.41%**. These trace-based transition rates support a difference in short-term prediction disruption, but the groups enter consolidation with different Binding exposure trajectories and accuracies; they do not isolate rehearsal as the cause.

### Does specialization-before-rebalancing outperform the reverse?

The two Spiral schedules have the same intended prefix, totals, and final common-Mixed block. At step 9,000, specialization-before-rebalancing yields **+0.47 pp ID Binding** (CI [−1.51, +2.45]) and **−0.45 pp OOD Binding** (CI [−2.01, +1.11]) relative to the reverse order. Neither establishes a Binding advantage. Final ID Shape is **+0.94 pp** higher (paired CI [+0.38, +1.49], positive in all five seeds), a modest secondary observation that warrants replication.

**Prefix caveat:** the recorded pre-divergence dev trajectories of these schedules match exactly for seeds 43–46, but have small discrepancies for seed 42 despite the intended identical prefix. This is unresolved in the available outputs. The two runs were independently trained rather than forked from a persisted common model/optimizer/RNG checkpoint; thus this is **not** a perfectly controlled common-state continuation experiment. A future causal suffix comparison should save and restore one parent state per seed.

## Error analysis: why high prerequisite accuracy is deceptive

### Conjunction is consistent with a Color-only shortcut

For Scaffolded Spiral at the final checkpoint, pooled over five seeds:

| Conjunction question type | Accuracy |
|---|---:|
| Both requested attributes exist (`Yes`) | 90.78% |
| Requested Color is missing (`No`) | **99.21%** |
| Requested Shape is missing, Color is present (`No`) | **25.38%** |

The split contains 128 positives, 76 missing-Color negatives, and 52 missing-Shape negatives **per seed**. A classifier that simply answers whether the requested Color is present would therefore achieve **(128 + 76)/256 = 79.69% accuracy** on this ID-dev task. Scaffolded Spiral achieves **80.00%** overall Conjunction accuracy, nearly the same.

This strongly suggests that high Conjunction accuracy can be attained without reliably checking both attributes. It is **behavioral evidence consistent with a Color-dominant shortcut**, not proof of the network's internal mechanism. It undermines the assumption that Conjunction performance alone certifies prerequisite understanding for same-object Binding. A shape-controlled counterfactual evaluation is needed.

### Shape and held-out composition remain difficult

Final Scaffolded Spiral ID-dev Shape accuracy by queried shape: **circle 80.9%, crescent 67.4%, triangle 65.0%, square 58.5%** (pooled across seeds). Weak Shape recognition persists despite extensive foundations training.

Final OOD-dev Binding accuracy on held-out combinations (pooled over seeds):

| Combination | Scaffolded Spiral | Prerequisite-Mixed | Staged Rehearsal |
|---|---:|---:|---:|
| Blue–square | 62.9% | 60.6% | 61.6% |
| Green–circle | 59.4% | 65.0% | 59.8% |
| Purple–triangle | **48.5%** | 52.6% | 50.5% |
| Red–crescent | **50.8%** | 50.6% | 54.9% |

Purple–triangle and red–crescent remain near chance. Since held-out pairs also differ in their component attributes and scene composition, these results **cannot isolate** a shape-specific causal explanation; they motivate architecture and counterfactual diagnostics.

### Similar accuracy, different prediction behavior

On OOD Binding, Scaffolded Spiral has **59.4% positive recall / 51.5% negative specificity**, while Fully Mixed has **77.3% recall / 35.6% specificity**. Spiral predicts `Yes` less often (53.95% versus 70.85%). Better negative rejection comes with more missed true pairs; neither pattern alone establishes improved object-level Binding.

`paired_question_disagreement.csv` shows correctness disagreements between Scaffolded Spiral and Fully Mixed on **292/1,280 (22.8%) ID Binding seed-question instances** and **576/2,000 (28.8%) OOD instances**. On OOD instances, Spiral alone is correct on **278**, versus **298** for Mixed alone. Their near-similar aggregate accuracy masks substantially different choices; without feature probes, these cannot be attributed uniquely to representation changes instead of decision-boundary or confidence differences.

Mean final OOD cross-entropy is **0.793 for Scaffolded Spiral**, compared with 0.758 for Fully Mixed; the Spiral value exceeds the approximately 0.693 loss of uninformative 50/50 probabilities. Accuracy above 50% thus does not imply good OOD probability quality or calibration.

## Interpretation and next experiment

1. **The full combined curriculum is not validated as better.** Scaffolded Spiral does not improve the prespecified fixed-final Binding outcome or OOD generalization over the existing controls. The result is informative even without a winning schedule.
2. **Specialization versus retention is the main learning-dynamics finding.** Rehearsal-rich schedules preserve older-task accuracy better during Binding practice, whereas pure Binding focus can acquire higher intermediate Binding accuracy and subsequently lose some of it during consolidation. A schedule-level comparison cannot assign the effect to one ingredient.
3. **There is no established optimal order of the two Spiral suffixes.** Swapping rebalancing and Binding focus yields no resolved final Binding difference; a genuinely shared checkpoint is needed for a cleaner follow-up.
4. **The assumed prerequisite hierarchy is not yet verified.** Color can approach mastery while Shape remains weak, and Conjunction can look successful under a Color-only heuristic. Task accuracy is an imperfect proxy for acquiring the underlying visual relationship.
5. **Next, test phase-aware learning-rate resets while freezing the architecture and data.** Compare constant LR, global decay, and resets at **genuine task-mixture boundaries**, holding task schedules and example exposure fixed. After the optimization experiment, redesign Shape/object-sensitive representations and rerun relevant historical curriculum controls on the *new* architecture rather than comparing architectures indirectly.

## Evidence and limitations

Sources: executed `scaffolded_spiral(1).ipynb`, `scaffolded_spiral_analysis(1).ipynb`, `schedule_comparison(20261108-105603).json`, `paired_differences(3).csv`, `phase_exposure(1).csv`, `prediction_change_summary(1).csv`, `prediction_error_subgroups(2).csv`, and `paired_question_disagreement.csv`. Corresponding project outputs are under `outputs/medium_world/experiments/scaffolded_spiral/`.

Only the **development splits** were analyzed. Both are repeatedly inspected in this research line and should not be presented as a fresh, sealed final test. The present file-level review verifies the supplied aggregate histories and CSVs; the compressed prediction trace files and original frozen scene images were not independently re-executed here. Five seeds, non-independent evaluations of the same questions, unadjusted multiple contrasts, and the seed-42 common-prefix discrepancy limit stronger causal claims. No additional training was performed for this report.
