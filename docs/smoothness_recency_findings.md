# Smoothness × Recency — Five-Seed Findings

## Research question

Does gradually changing the task mixture preserve visual Binding better than sharper transitions **when Binding exposure timing is matched**? Independently, does shifting Binding exposure later improve retention?

## Setup and controls

- **World:** Medium World, 64×64 RGB scenes; Color, Shape, Conjunction, and same-object Binding (Pair) questions. The OOD-dev split evaluates held-out color–shape combinations.
- **Model:** unchanged `AttributeBindingViT`, dimension 48, 4×4 patches, one Transformer encoder layer. AdamW, constant learning rate `1e-4`, batch size 50, image augmentation enabled.
- **Design:** five paired seeds (42–46). For each seed, all four strategies resume the same **step-6,300 model, optimizer and RNG state** from the Prerequisite-Mixed → Pair prefix. Evaluation every 100 updates.
- **Budget:** 9,000 updates and 45 presentations of **every** training question per strategy. The last 2,700 updates include 67,500 Binding-question presentations and 22,500 from each other task.
- **Factorial intervention:** nine 300-update windows after step 6,300. Within each Early/Late recency level, Smooth and Stepwise have equal overall exposure, equal mean Binding presentation time, and equal first/last window percentages. Their maximum adjacent-window mixture changes are **10 versus 28 percentage points**. Early and Late shift mean Binding presentation time from optimizer step **7,350 to 7,950**.
- **Integrity:** both executed notebooks report **20 completed runs, five shared parents, 91 evaluation checkpoints per run**, matching pre-branch prediction traces, and verified dataset hashes.

| Strategy | Binding share across nine 300-step windows (%) |
|---|---|
| Smooth / Early | 80, 70, 65, 60, 50, 45, 35, 25, 20 |
| Stepwise / Early | 80, 74, 74, 46, 46, 46, 32, 32, 20 |
| Smooth / Late | 20, 25, 35, 45, 50, 60, 65, 70, 80 |
| Stepwise / Late | 20, 32, 32, 46, 46, 46, 74, 74, 80 |

## Final Binding performance

Five-seed mean at the predefined final checkpoint, step 9,000:

| Strategy | ID-dev Binding | OOD-dev Binding | OOD positive recall | OOD negative specificity |
|---|---:|---:|---:|---:|
| Smooth / Early | **65.47%** | 58.70% | 66.9% | 50.5% |
| Stepwise / Early | 65.00% | 59.10% | 65.6% | 52.6% |
| Smooth / Late | 65.39% | 59.35% | 71.5% | 47.2% |
| Stepwise / Late | 65.23% | **59.80%** | 71.5% | 48.1% |

### Does smoother task mixing help?

| Paired contrast (Smooth − Stepwise) | ID difference | OOD difference |
|---|---:|---:|
| Early | +0.47 pp, 95% CI [−1.39, +2.33] | −0.40 pp, CI [−2.45, +1.65] |
| Late | +0.16 pp, CI [−1.68, +1.99] | −0.45 pp, CI [−2.43, +1.53] |

The average smoothness effect across the two recency conditions is **+0.31 pp ID** (paired 95% CI [−0.98, +1.60]) and **−0.43 pp OOD** (CI [−2.24, +1.39]). There is **no consistent final-accuracy benefit** from smaller task-mixture jumps in the tested range.

### Does later Binding practice help?

Later Binding exposure raises mean OOD accuracy by **+0.65 pp** for Smooth and **+0.70 pp** for Stepwise, with seed-to-seed reversals. Averaged across transition styles, the paired recency effect is **+0.68 pp OOD** (95% CI [−3.27, +4.62]); for ID it is **+0.08 pp** (CI [−1.50, +1.66]). Neither supports a consistent final advantage. The Smoothness × Recency interaction also has a confidence interval spanning zero on both splits.

## Prediction dynamics reveal a timing effect

Comparing question correctness at step 6,300 and step 9,000, the OOD Binding traces show:

| Strategy | Forgotten / previously correct | Newly correct / previously wrong | Correctness churn |
|---|---:|---:|---:|
| Smooth / Early | 16.9% | 25.6% | 20.6% |
| Stepwise / Early | 16.9% | 26.4% | 20.9% |
| Smooth / Late | 12.7% | 21.6% | 16.4% |
| Stepwise / Late | 12.3% | 22.1% | 16.5% |

Later Binding practice is associated with **less final question-level forgetting and lower prediction turnover**. However, fewer initially wrong questions become correct, so the net OOD accuracy gain is small. This is a more informative pattern than the final scores alone.

The **intermediate trajectory differs** as well. Between steps 6,300 and 7,200, the Late strategies temporarily forget more previously correct OOD Binding questions (**17.4–19.7%** versus **10.7–11.6%** for Early). By step 9,000, the Late strategies have recovered a larger fraction of those intermediate losses (**56.3–58.3%** versus **38.9–41.1%**). These comparisons describe correctness changes at specific checkpoints, not permanent loss of knowledge.

Earlier-task ID accuracy improves from step 6,300 to step 9,000 under all four strategies: approximately **+3.2–3.8 pp Color**, **+8.7–9.5 pp Shape**, and **+7.1–8.8 pp Conjunction**. Their mean worst observed accuracy drop below the shared starting checkpoint is at most **0.55 pp**. Thus, sustained interleaving restores earlier tasks without a large observed intermediate decline.

## OOD composition errors

The held-out combinations remain uneven. Across the four strategies, aggregated final OOD accuracy is approximately **63–65% for blue–square**, **66–70% for green–circle**, **50–53% for purple–triangle**, and **53–54% for red–crescent**. The latter two remain difficult under every allocation. These pair-specific differences do not separately identify whether the color or shape is responsible.

The Late schedules also trade higher OOD positive recall for lower negative specificity, consistent with a greater tendency to predict that a requested pair exists.

## Interpretation and next decision

1. **Smoothness hypothesis:** Within the tested 10 pp versus 28 pp window-to-window change, and with Binding timing matched, smoother transitions do **not** deliver a repeatable final Binding advantage.
2. **Recency hypothesis:** Shifting the mean Binding presentation time 600 steps later does **not** deliver a repeatable final accuracy advantage, although it changes forgetting, recovery, and positive/negative prediction behavior.
3. **Connection to the prior transition-rehearsal experiment:** That experiment found improvements from redistributing the last training phase relative to an extreme abrupt switch. The present controls show that those gains cannot be attributed straightforwardly to transition smoothness or mean Binding recency **alone**. The current profiles do not reproduce the prior 100%-to-25% switch, so they do not test every possible transition.
4. **Next phase:** Preserve this result and examine **learning-rate scheduler resets at task transitions** as a separate intervention, with the data schedule held fixed. Discuss the proposed gradient-informed allocation framework with Claude **after** that phase and before designing it; only then decide between pursuing it and moving to Large World.

## Evidence

Executed `smoothness_recency(1).ipynb` and `smoothness_recency_analysis(1).ipynb`, plus `paired_differences(2).csv`. The analysis notebook also generates `final_by_seed.csv`, `learning_curves.csv`, `earlier_task_retention.csv`, `prediction_change_summary.csv`, `recovery_summary.csv`, and `prediction_error_subgroups.csv` in `outputs/medium_world/experiments/smoothness_recency/five_seeds/`. No retraining or test-set evaluation was performed for this report.
