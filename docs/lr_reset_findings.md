# Phase-Aware Learning-Rate Restarts — Five-Seed Findings

## Research question

When a curriculum introduces a new task or changes its task distribution, does restarting the learning-rate (LR) schedule improve acquisition of same-object Binding, retention of earlier tasks, and generalization to unseen color–shape combinations? In particular, is a phase-aware restart better than an uninterrupted global decay **and** better than a constant LR?

## Experimental setting and controls

- **World and task:** frozen Medium World (64×64 RGB scenes). Training contains 10,000 questions, equally divided among Color, Shape, Conjunction, and same-object Binding (`pair`). ID-dev contains 1,024 questions (256 per task); OOD-dev contains 400 Binding questions on four held-out color–shape combinations.
- **Architecture:** unchanged `AttributeBindingViT`, dimension 48, 4×4 patches, one Transformer encoder layer; no explicitly object-indexed representations.
- **Training:** AdamW, weight decay 0, batch size 50, image augmentation, gradient clipping at norm 1, initial LR `1e-4`. Five paired seeds, **42–46**, and four fixed curricula: `prereq_mixed_pair_focus`, `staged_rehearsal`, `scaffolded_spiral`, and `mixed`.
- **Budget:** **9,000 optimizer updates**, 450,000 question presentations, exactly **45 presentations of every training question** per run. The model and optimizer parameters persist across stages; reset policies change LR scheduling, not the network weights or AdamW moments.
- **Policies:** Constant (`1e-4`); Global Cosine (`1e-4` to `1e-6` over 9,000 updates); Phase-Reset Cosine (cosine cycles within genuinely different task mixtures); Global ReduceLROnPlateau; Phase-Reset ReduceLROnPlateau. Both plateau policies use factor `2/3`, patience 2 **evaluation calls**, absolute threshold `0.001`, minimum LR `1e-6`, and monitor **overall ID-dev cross-entropy**, not active-task loss. Evaluation occurs at least every 100 updates, with extra phase-relative checkpoints. The restart LR is `1e-4`, except the final common-Mixed phase starts at `5e-5`.
- **Actual reset boundaries:** Prerequisite-Mixed at steps **5,400, 7,200**; Staged Rehearsal at **3,000, 4,950, 7,200**; Scaffolded Spiral at **2,800, 4,450, 6,100, 7,200**; Fully Mixed has **none**. The nominal Prerequisite-Mixed boundary at 6,300 is not a reset, since both adjacent blocks contain only Binding.
- **Integrity:** 100/100 completed runs; all reach step 9,000 and the required exposure counts. ID Binding area under the learning curve (AULC) is computed on the common 100-step grid. Dataset hashes are recorded in the executed report; no architecture, data-split, or total-budget change was introduced.

All results below are on repeatedly inspected **development** splits, not an untouched final test. Percentages refer to accuracy unless specified otherwise.

## Final Binding performance

Five-seed means at the **predeclared final step 9,000**; entries are **ID-dev / OOD-dev** accuracy (%).

| Curriculum | Constant | Global Cosine | Reset Cosine | Global Plateau | Reset Plateau |
|---|---:|---:|---:|---:|---:|
| Prereq-Mixed → Pair | **62.89 / 57.20** | 54.69 / 55.05 | 57.58 / 56.60 | 51.41 / 52.35 | 58.20 / 57.35 |
| Staged Rehearsal | **63.28 / 56.20** | 57.03 / 53.20 | 59.06 / 56.35 | 53.12 / 50.70 | 59.92 / 56.40 |
| Scaffolded Spiral | **61.17 / 56.15** | 56.80 / 53.55 | 58.28 / 56.80 | 53.12 / 52.35 | 60.55 / 56.80 |
| Fully Mixed | **62.03 / 56.65** | 56.95 / 55.10 | 56.95 / 55.15 | 58.98 / 54.85 | 58.36 / 55.15 |

The average across the four curricula (descriptive, **not** four independent replications) is:

| Policy | ID Binding | OOD Binding | ID Binding AULC | Final ID Shape |
|---|---:|---:|---:|---:|
| Constant | **62.34%** | **56.55%** | **55.49%** | **67.32%** |
| Global Cosine | 56.37% | 54.23% | 53.53% | 64.55% |
| Reset Cosine | 57.97% | 56.23% | 54.37% | 64.16% |
| Global Plateau | 54.16% | 52.56% | 52.86% | 62.50% |
| Reset Plateau | 59.26% | 56.43% | 54.82% | 64.82% |

**Primary result:** Constant LR has the highest mean final **ID Binding** accuracy in every curriculum. Restarts improve performance against global decay in curricula with actual phase transitions, but **no restart policy establishes a final ID Binding advantage over Constant**. OOD advantages are small and inconsistent.

## Matched-policy effects, with paired uncertainty

Differences are **Reset minus Global** in percentage points (pp). Intervals are two-sided, unadjusted 95% paired *t* intervals over the five seed-wise differences; exploratory contrasts are numerous and not multiplicity-corrected.

| Curriculum | Reset Cosine: ID | Reset Cosine: OOD | Reset Plateau: ID | Reset Plateau: OOD |
|---|---:|---:|---:|---:|
| Prereq-Mixed → Pair | **+2.89** [0.76, 5.02] | +1.55 [−1.60, 4.70] | **+6.80** [4.17, 9.42] | **+5.00** [1.22, 8.78] |
| Staged Rehearsal | +2.03 [−2.17, 6.24] | **+3.15** [0.84, 5.46] | **+6.80** [2.70, 10.89] | **+5.70** [1.57, 9.83] |
| Scaffolded Spiral | +1.48 [−2.65, 5.62] | **+3.25** [0.65, 5.85] | **+7.42** [1.59, 13.25] | +4.45 [−0.69, 9.59] |
| Fully Mixed | +0.00 [−0.34, 0.34] | +0.05 [−0.09, 0.19] | −0.62 [−1.74, 0.49] | +0.30 [−0.72, 1.32] |

Against **Constant**, the contrast is unfavorable or unresolved:

- Prereq-Mixed: Reset Cosine **−5.31 pp ID** [−10.35, −0.28]; Reset Plateau **−4.69 pp ID** [−8.98, −0.39].
- Staged Rehearsal: Reset Cosine −4.22 pp ID [−10.65, 2.21]; Reset Plateau −3.36 pp ID [−10.22, 3.51].
- Scaffolded Spiral: Reset Cosine **−2.89 pp ID** [−5.18, −0.60]; Reset Plateau −0.62 pp ID [−3.40, 2.15].
- Fully Mixed: Reset Cosine **−5.08 pp ID** [−7.19, −2.96]; Reset Plateau −3.67 pp ID [−7.34, approximately 0].

The Fully Mixed negative control has no real reset events. Global/Reset Cosine therefore specify the same LR sequence, with almost identical output; small differences in some seeds indicate residual execution nondeterminism. Global/Reset Plateau likewise have no restarts, but their independent numerical histories can diverge slightly. Do not interpret these as restart effects.

## Mechanism: task introduction after LR decay

Prereq-Mixed trains only prerequisites until step 5,400, then Binding only until 7,200. Mean **ID Binding accuracy** across five seeds:

| Policy | Step 5,400 | Step 7,200 | Final 9,000 |
|---|---:|---:|---:|
| Constant | 50.16% | **67.50%** | **62.89%** |
| Global Cosine | 50.62% | 57.89% | 54.69% |
| Global Plateau | 50.62% | 50.94% | 51.41% |
| Reset Cosine | 50.55% | 60.94% | 57.58% |
| Reset Plateau | 50.62% | 62.19% | 58.20% |

For Global Plateau, the mean LR by Binding introduction is approximately `4e-6` (some seeds are already at the `1e-6` floor). Thus the first difficult-task phase receives little optimization capacity. The reset version raises the LR on task change and subsequently learns more Binding. That **explains a plausible mechanism** for the relative benefit; it does not isolate scheduler-history resetting from the larger LR itself.

Logged cumulative **LR × question presentations** for Binding in Prereq-Mixed:

| Policy | Binding-weighted LR exposure |
|---|---:|
| Constant | 11.250 |
| Global Cosine | 2.063 |
| Reset Cosine | 5.119 |
| Global Plateau | 0.203 |
| Reset Plateau | 6.108 |

Reset Cosine allocates about **2.48×** as much LR-weighted exposure to Binding as Global Cosine; Reset Plateau about **30×** Global Plateau. This is **not** an estimate of parameter displacement: gradient direction, magnitude, clipping, and AdamW state are not represented. The reset comparison conflates LR magnitude, LR timing, and for Plateau the reset of best-loss history.

The Plateau monitor presents an additional design problem: overall ID-dev loss includes tasks that are absent from the active training phase. Thus it may worsen as the model learns the currently trained task, triggering early LR reductions for reasons unrelated to that task's learnability. The current patience is measured in **100-update evaluations**, not whole task epochs. The experiment tests *this policy*, not every possible ReduceLROnPlateau design.

## Interference, retention, and consolidation

When Prereq-Mixed first switches to Binding, changes from step **5,400 → 5,500** are:

| Policy | ID Shape | ID Conjunction | ID Binding |
|---|---:|---:|---:|
| Constant | −7.19 pp | −8.20 pp | +2.27 pp |
| Global Cosine | −5.16 pp | −7.42 pp | +0.00 pp |
| Global Plateau | −0.78 pp | −0.55 pp | −0.23 pp |
| Reset Cosine | −4.14 pp | −9.77 pp | +3.20 pp |
| Reset Plateau | −6.95 pp | −10.62 pp | +1.88 pp |

Almost no change under Global Plateau is **not evidence of superior retention**: this model has already decayed its LR so far that it learns Binding weakly. In the first 100 updates of final consolidation (**7,200 → 7,300**), Constant loses about **5.23 pp ID Binding** while gaining **7.42 pp Shape**; Reset Cosine loses **2.73 pp Binding** while gaining **5.31 pp Shape**. These are prediction-level accuracy changes, not direct measurements of stored-knowledge erasure or representation recovery.

## OOD behavior and remaining visual-reasoning failures

OOD accuracy pooled across the four curricula and five seeds by held-out color–shape pair:

| Held-out pair | Constant | Global Cosine | Reset Cosine | Global Plateau | Reset Plateau |
|---|---:|---:|---:|---:|---:|
| Blue–square | 61.19% | 58.09% | 60.36% | 57.22% | 61.55% |
| Green–circle | 62.20% | 63.76% | **66.38%** | 61.51% | 63.99% |
| Purple–triangle | 50.99% | 49.28% | 50.27% | 49.77% | 50.95% |
| Red–crescent | 51.14% | 43.80% | 46.02% | 39.10% | 47.83% |

The difficult purple–triangle and red–crescent pairs remain near chance or worse under every policy. OOD differences may also reflect differences in scene complexity and individual constituent attributes; they do not isolate pure binding.

Across the four curricula, final OOD Binding decision characteristics are:

| Policy | Positive recall | Negative specificity | Mean OOD CE (nats) |
|---|---:|---:|---:|
| Constant | 66.30% | **46.80%** | 0.774 |
| Global Cosine | 66.45% | 42.00% | 0.729 |
| Reset Cosine | 72.35% | 40.10% | **0.713** |
| Global Plateau | **75.15%** | 29.98% | 0.750 |
| Reset Plateau | 72.03% | 40.83% | 0.745 |

The reset policies increase positive recall relative to Constant but reduce specificity: more true pairs are accepted **and** more absent pairs are falsely accepted. Lower cross-entropy under Reset Cosine is an interesting probability-quality result, but it does **not** alone prove better calibration. Even its mean OOD CE exceeds the `ln(2) ≈ 0.693` loss of a uniform 50/50 predictor.

### Shape remains weak

Across curricula at the final checkpoint, mean ID Shape accuracy is **67.32% Constant**, **64.16% Reset Cosine**, and **64.82% Reset Plateau**. The scheduler intervention did not fix Shape recognition. In this model, patchwise query scores are pooled using max/mean; there is no explicit mechanism assigning all patches of one object the same object identity.

### Conjunction remains Color-dominant

Final ID Conjunction subgroup accuracies pooled across curricula and seeds:

| Subgroup | Constant | Reset Cosine | Reset Plateau |
|---|---:|---:|---:|
| Both requested attributes present (Yes) | 91.25% | 93.48% | 93.28% |
| Requested Color missing (No) | 99.01% | 98.88% | 99.14% |
| Requested Shape missing, Color present (No) | **26.25%** | 15.19% | 17.69% |

The ID split has 128 positives, 76 negatives missing Color, and 52 negatives missing Shape. Therefore the **Color-only shortcut**, answering Yes whenever the requested Color exists, achieves `(128 + 76)/256 = 79.69%` Conjunction accuracy without checking Shape. The learned models score near this level, but the asymmetry is stronger evidence for shortcut behavior than total accuracy alone. It remains a **behavioral hypothesis**, not proof that Shape information is absent internally.

## Interpretation and limitations

1. **Relative benefit supported:** phase-aware restarts generally recover Binding acquisition lost under uninterrupted global decay, especially for Global Plateau. The benefits are specific to the tested stage-changing curricula and optimizer policies.
2. **No general superiority:** Constant LR is stronger on mean final ID Binding in all four curricula; no consistent advantage on OOD development accuracy is established.
3. **Coupled mechanisms:** resetting LR, redistributing LR-weighted task exposure, and (Plateau only) clearing the running loss history occur together. These results cannot identify which component causes the improvement.
4. **Adaptation–retention trade-off:** stronger Binding acquisition can coincide with transient loss of earlier-task accuracy; a nearly frozen model can misleadingly appear to retain skills.
5. **Representation limits persist:** Shape recognition, missing-Shape Conjunction negatives, and several OOD combinations remain hard. LR alone does not demonstrate correct object-level reasoning.
6. **Exploratory reliability:** five seeds, repeated use of the same development splits, many unadjusted comparisons, and slight divergence between nominally identical Fully Mixed schedules limit strong statistical/causal claims. Do not report these as sealed test performance or claim internal feature destruction from prediction churn.

## Next research step

**Preserve this experiment, then investigate the existing architecture before redesigning it.** Begin with behavioral counterfactual probes and frozen-representation diagnostics focused on:

- Attribute detection: Does the image encoder reliably represent each shape, independently of Color?
- Conjunction: Are predictions insensitive to removing or adding the queried Shape while retaining Color?
- Same-object Binding: Does swapping colors between two existing objects change the requested-pair prediction even when the scene's color and shape multisets are unchanged?

Compare frozen features with task-head outputs to distinguish an encoder bottleneck from a head/aggregation bottleneck. Then redesign the architecture and **rerun matched baseline curricula with both old and new models**, initially using Constant LR and keeping Reset Cosine as a secondary optimization control. This investigation is specified separately in `representation_failure_diagnostics_protocol.md` if adopted.

## Evidence and provenance

- Executed `notebooks/medium_world/lr_reset.ipynb` and `lr_reset_analysis.ipynb`.
- `outputs/medium_world/experiments/phase_aware_lr/schedule_comparison.json` (100 completed runs; model, schedule, LR histories, final metrics, dataset hashes), with generated `paired_differences.csv`, `prediction_change_summary.csv`, and `prediction_error_subgroups.csv`.
- Model and scheduler code: `src/medium_visual_world/learning.py`, `src/medium_visual_world/lr_policies.py`; initial setup: `docs/lr_reset_protocol.md`.

No untouched final-test evaluation was conducted for this report. All explanations beyond measured error patterns are explicitly hypotheses.
