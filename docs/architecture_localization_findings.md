# Architecture Localization: Spatial Evidence Helps Binding, but Operator Selectivity Remains Weak

## Research question

The curriculum experiments exposed three recurring behaviors: Shape recognition was unreliable, Conjunction often behaved as though Color mattered more than Shape, and Binding improved under concentrated practice but partly deteriorated during later mixed training. Earlier frozen-feature probes showed that spatially ordered visual features contained recoverable geometric information, whereas the deployed model reduced query-conditioned evidence maps to eight scalar maxima and means. We therefore asked a narrower question:

**Does preserving spatial evidence improve the intended Shape, Conjunction and same-object Binding behaviors—and can the model use that evidence only when the queried operator requires it?**

The key distinction is between independent presence and object-level association. For a scene with objects $o$,

$$
J(c,s)=\mathbb{1}[\exists o:\operatorname{color}(o)=c]\;\land\;\mathbb{1}[\exists o:\operatorname{shape}(o)=s],
$$

$$
B(c,s)=\mathbb{1}[\exists o:\operatorname{color}(o)=c\land\operatorname{shape}(o)=s].
$$

A Color swap between two objects can flip $B(c,s)$ without changing $J(c,s)$. That intervention lets us distinguish useful association sensitivity from inappropriate sensitivity on an independent-Conjunction question.

## Comparison and evidence

We reused **30 frozen `AttributeBindingViT` checkpoints**: two Constant-LR curricula (`mixed` and `prereq_mixed_pair_focus`), paired seeds 42–46, and pretrained training steps 5,400 / 7,200 / 9,000. At each checkpoint, the original pretrained head was evaluated alongside three **newly supervised** readouts:

| Pathway | Inputs and operations | Trainable parameters |
|---|---|---:|
| **Pretrained** | Original query-conditioned max/mean summaries → shared `8 → 32 → 2` MLP; no new fitting | Original frozen head |
| **Scalar A** | The exact eight original pooled values → new `8 → 256 → 2` MLP | 2,818 |
| **Spatial B** | Four spatial `16 × 16` attribute/joint/interaction score maps → two 3×3 convolutions, spatial max/mean, classifier | 2,978 |
| **Tokens C** | Frozen `256 × 48` visual tokens plus trained query representation → learned projection and spatial convolutional readout | 2,892 |

The readouts were fitted **independently at every checkpoint**, using the same 10,000 training questions, 30 epochs, batch size 128 and learning rate `1e-3`. The pretrained ViT was not updated. The new readouts have approximately matched parameter counts, but **different architectures, access paths and inductive biases**; parameter matching does not isolate spatial pooling as the sole mechanism. All results below use the repeatedly inspected ID-development data, not sealed test results.

The evaluation comprises 1,024 ordinary ID-dev questions (256 per task), a 109-question requested-attribute-count-controlled Binding subset (**34 Yes / 75 No**), its **62 swaps that do not create a held-out combination**, and a broader grid of **239 ID-safe Binding swaps**. The last set tests four task operators on the original and edited scene. Shape-insertion and Color-recoloring controls test geometry sensitivity and invariance. Means and contrasts use **five training seeds** as the units of replication, not hundreds of correlated edited questions.

Five random-visual controls also refit all three readouts at the Mixed step-9,000 setting, replacing the visual pathway with a newly randomized one while **retaining the pretrained query embeddings/projector**. Thus this is not a fully random-model baseline.

## 1. Spatial readout makes Binding more discriminative

At the final pretrained checkpoint, mean ordinary Binding accuracy is **73.75%** for Spatial B versus **66.25%** for Scalar A in Mixed; the corresponding Prerequisite-Mixed figures are **71.64%** versus **63.75%**. These aggregate gains are promising but insufficient on their own: the original dataset permits incidental shortcuts.

The stricter evaluations point in the same general direction:

| Final checkpoint, five-seed mean | Mixed A | Mixed B | Prereq-Mixed A | Prereq-Mixed B |
|---|---:|---:|---:|---:|
| Count-controlled Binding **balanced accuracy** | 66.94% | **71.58%** | 63.82% | **70.15%** |
| Count-controlled Binding **ROC-AUC** | 72.35% | **79.00%** | 71.36% | **77.98%** |
| ID-safe strict swap **both correct**, 62 pairs | 34.52% | **44.84%** | 33.87% | **46.45%** |
| ID-safe strict swap **correct probability direction** | 73.87% | **81.29%** | 73.23% | **82.26%** |

The count-controlled AUC is threshold-free: it does not depend on whether the Yes probability crosses 0.5. On the swaps, correctly moving the probability in the direction of the oracle label change is also different from getting **both** edited answers correct. Spatial B improves both kinds of measurements, but its best strict both-correct rate is still below 47%; reliable association-sensitive decisions have **not** been achieved.

Paired **Spatial B minus Scalar A** differences at step 9,000, in percentage points (95% unadjusted Student-*t* interval over five seeds):

| Target | Mixed difference [95% CI] | Prereq-Mixed difference [95% CI] |
|---|---|---|
| Count-controlled Binding ROC-AUC | +6.64 [−1.86, +15.15], 4/5 positive | **+6.61 [+1.01, +12.22]**, 5/5 |
| Strict ID-safe swap both-correct | +10.32 [−1.41, +22.05], 4/5 | +12.58 [−4.32, +29.48], 5/5 |
| Strict ID-safe swap correct direction | **+7.42 [+4.73, +10.11]**, 5/5 | **+9.03 [+0.12, +17.94]**, 4/5 |

There is converging evidence that retaining spatial score-map structure can help a newly fitted readout exploit association-sensitive signals. It does **not** show that the pretrained model already implements a stable object abstraction, or that max/mean pooling alone caused every failure.

## 2. Conjunction improvement hides a positive–negative trade-off

The standard ID Conjunction benchmark rewards a Color-dominant strategy: the rule “Yes whenever the requested Color exists” scores **79.69%** without checking Shape reliably. Its most discriminating subset consists of **52 negatives** where Color is present and Shape is absent.

| Final checkpoint | Mixed Scalar A | Mixed Spatial B | Prereq Scalar A | Prereq Spatial B |
|---|---:|---:|---:|---:|
| Correct rejection: **Shape-missing** Conjunction negatives | 31.54% | **39.23%** | 31.54% | **46.54%** |
| Recall on **positive** Conjunction questions | **88.91%** | 83.44% | **90.78%** | 83.12% |
| Overall Conjunction accuracy | **80.31%** | 79.14% | **81.33%** | 80.47% |
| Ordinary Shape accuracy | 67.66% | 68.28% | **67.58%** | 66.33% |

The Shape-missing gain for Spatial over Scalar is **+7.69 pp** in Mixed [−0.41, +15.79] and **+15.00 pp** in Prerequisite-Mixed [+3.62, +26.38]. But positive Conjunction recall decreases, leaving overall accuracy slightly **lower**, not higher. This is compatible with changed selectivity or threshold behavior; it is **not evidence of a repaired logical AND computation**.

Nor does Spatial B substantially improve ordinary Shape detection. The paired final differences are +0.62 pp [−1.61, +2.86] in Mixed and −1.25 pp [−4.27, +1.77] in Prerequisite-Mixed. On Color-only recoloring, Shape answers for both Shape-present and Shape-absent controls remain correct at both endpoints only around **51–60%**, depending on condition. Roughly one-fifth of hard Shape predictions change despite the true Shape answer being invariant. An invariant prediction is also not necessarily **correct**; always giving the same wrong answer would pass a pure flip-rate test.

## 3. Spatial Binding sensitivity leaks into independent Conjunction

The broader **239 ID-safe Binding swaps** preserve the object count, Color/Shape multisets and geometry, while changing which Color and Shape occur on the same object. They therefore alter the Binding label but preserve the Color, Shape and **independent Conjunction** labels.

At step 9,000, Spatial B flips its **Conjunction** hard prediction between a swap's two endpoints in **27.87%** of Mixed cases and **30.88%** of Prerequisite-Mixed cases. Scalar A flips in only **14.81%** and **16.07%**, respectively. Since the oracle Conjunction label is unchanged, any such flip leaves at least one endpoint wrong.

The paired Spatial-minus-Scalar excess flip rate is **+13.05 pp [5.91, 20.20]** in Mixed and **+14.81 pp [3.63, 25.99]** in Prerequisite-Mixed, **positive in all five seeds** of both comparisons. This is one of the clearest replicated effects in the experiment, though the intervals remain exploratory and unadjusted for many contrasts.

The response matrix shows where the changes occur. For Spatial B at the final checkpoint:

| Intervention | Mixed: hard-prediction flip rate | Prereq-Mixed: flip rate | Correct oracle behavior |
|---|---:|---:|---|
| Binding swap → **Binding** question | 48.1% | 50.0% | Answer should flip |
| Binding swap → **Conjunction** question | **27.9%** | **30.9%** | Answer should remain unchanged |
| Binding swap → **Shape** question | 17.7% | 17.8% | Answer should remain unchanged |
| Binding swap → **Color** question | 1.8% | 2.2% | Answer should remain unchanged |

The extra Binding responsiveness is therefore **not purely operator-specific**. The spatial readout also responds to the association edit in task contexts that require only independent attribute presence. Probability-level responses tell the same story: under Binding swaps, Spatial B's mean absolute change in Conjunction Yes probability is approximately **0.226** in Mixed and **0.238** in Prerequisite-Mixed, despite no Conjunction label change.

The notebook's `binding_minus_conjunction_delta` is a descriptive **signed-response contrast aligned with the Binding label change**, not a validated disentanglement or causal selectivity score. Spatial B raises this contrast somewhat, but it **also increases unwanted Conjunction flips**. The latter prevents interpreting the signed gap alone as successful operator separation.

## 4. Correctness across all related questions is still low

We next require **all eight answers** to be correct: four task operators on both endpoints of each of the 239 swaps. This is more demanding than one Binding score because it jointly tests attribute availability, independent Conjunction and same-object association.

| Final checkpoint, jointly correct on all 8 answers | Pretrained | Scalar A | Spatial B | Tokens C |
|---|---:|---:|---:|---:|
| Mixed | 11.55% | 14.56% | **17.91%** | 11.88% |
| Prerequisite-Mixed | 13.72% | **16.65%** | 16.49% | 11.55% |

Spatial-minus-Scalar is **+3.35 pp [−1.91, +8.61]** for Mixed and **−0.17 pp [−6.77, +6.43]** for Prerequisite-Mixed. Neither comparison resolves a reliable gain. Joint correctness is low under every readout, even though ordinary Binding accuracy can exceed 70%.

This highlights a critical distinction: improving an association-sensitive score is **not** equivalent to producing a coherent set of predictions obeying all four task definitions.

## 5. Direct visual-token readout did not establish an additional benefit

Tokens C receives the full frozen visual-token field plus query conditioning, but under the current parameter budget and fitting procedure it does **not** outperform Spatial B consistently.

For Mixed at step 9,000, C achieves **69.23%** count-controlled AUC versus B's **79.00%**; its ID-safe strict-swap both-correct rate is **32.58%** versus **44.84%**. The paired Tokens-minus-Spatial effects are **−9.76 pp [−17.70, −1.83]** for AUC and **−12.26 pp [−19.96, −4.55]** for both-correct, with no positive seed on either contrast. Other contrasts, including Prerequisite-Mixed AUC, are uncertain.

This is a comparison of **these particular compact readouts**. Tokens C must learn a new mapping from high-dimensional visual tokens with limited parameters and fixed training. Its weaker results do **not** establish that raw visual tokens lack the information, or that the trained attribute-score maps are sufficient in principle.

## 6. Random-visual controls show useful learned pathway contributions, with limits

At the Mixed step-9,000 checkpoint, replacing the visual pathway with a random one while retaining the trained query pathway and refitting each readout lowers Spatial B's Shape accuracy by **9.69 pp [3.33, 16.04]**, count-controlled Binding balanced accuracy by **18.45 pp [9.01, 27.89]**, and strict ID-safe swap correct-direction rate by **19.68 pp [16.09, 23.26]**. All five seeds favor the pretrained visual pathway on these three contrasts.

For Tokens C, trained-minus-random differences are small and uncertain on Shape (+2.11 pp [−0.38, +4.60]) and count-controlled Binding balanced accuracy (−1.58 pp [−10.99, +7.83]). These differences describe the chosen readout and control, not an absolute absence of learned information.

The controls support a contribution of visual-pathway training to Scalar/Spatial task-effective information, while leaving open exactly what visual features were learned. The randomly initialized convolutional projection can itself preserve pixel geometry, and the random condition retains trained query machinery.

## 7. Checkpoint comparisons are about accessibility, not an unbroken new-readout learning curve

The Prerequisite-Mixed source model has **not** received direct Binding training by checkpoint 5,400. Its original count-controlled Binding balanced accuracy is **48.22%**, while a *newly supervised* Spatial B readout fitted at that frozen checkpoint achieves **67.15%** (and count-controlled AUC **74.79%**). This indicates that the available frozen representation plus newly trained decoder can support some discrimination **before** the original model learns to use it on Binding. It does **not** show that the original model already mastered Binding at step 5,400; the new decoder receives Binding labels during its own fitting.

The source model's Binding discrimination strengthens by 7,200, as observed in the prior diagnostic investigation. The newly fitted Spatial B readouts score **71.14%** count-controlled balanced accuracy at 7,200 and **70.15%** at 9,000. But the B readouts at those checkpoints are **different trained models**, fitted afresh on different frozen features. Their trajectories cannot be interpreted as retention or forgetting of one B classifier.

## Interpretation and what follows

The central result is a **benefit–failure pair**. Preserving spatial attribute evidence allows a compact supervised readout to make stronger count-controlled and counterfactual Binding discriminations than the eight-scalar readout. The strongest stable failure is that this benefit comes with increased sensitivity of independent-Conjunction predictions to association-only swaps. Shape recognition and joint correctness remain incomplete.

We can therefore narrow the design hypothesis to **task-conditioned use of spatial evidence**. A future end-to-end model should preserve spatial or object-local evidence, but distinguish (i) whether each attribute appears *anywhere* from (ii) whether both appear *on the same object*. Merely replacing the eight-scalar head, improving raw Binding accuracy, or exposing all tokens to an unconstrained decoder is not yet a demonstrated solution.

A suitable next ablation would compare a task-routed or explicitly operator-aware spatial readout against the current Spatial B under **identical supervision, training budget and parameter constraints**, then measure Binding swap both-correct, Conjunction invariance, missing-Shape negatives, and all-task joint correctness together. Activation patching on matched swaps may help localize contributing token or score-map regions, but would be a **new analysis**, not a result of the present experiment.

### Boundaries of the conclusions

- **Frozen-feature diagnostic, not an end-to-end architectural repair.** Three new readouts were trained for 30 epochs on task labels; the original head received no equivalent refitting. Differences from the pretrained model combine changed representation access, readout architecture, capacity and extra supervision.
- **Operator behavior is measured on rendered synthetic scenes.** Even count-controlled swaps and the ID-safe filter cannot exclude all local texture, outline, positional or selection cues; association sensitivity does not prove explicit object identity or systematic visual composition.
- **No sealed compositional test.** The original OOD-dev generator has a scene-level held-out-combination/label confound, so its accuracy cannot validate clean transfer. The present results emphasize ID-dev and validated edits instead.
- **Exploratory uncertainty.** Only five seeds, repeatedly inspected development scenes, many unadjusted paired intervals and shared scene/question IDs. The replicated leakage and direction-sensitivity patterns are more informative than isolated 1–2 pp differences; confidence intervals crossing zero do not prove equivalence.
- **No attribution to one layer.** The study does not distinguish a failure of score construction from a failure of spatial aggregation, query selection or training dynamics with a single causal intervention.

**Conclusion.** This experiment strengthens the evidence that the pretrained network contains useful spatial signals for same-object association, while exposing a second constraint: **those signals are not reliably used according to the logical requirements of each question**. For the larger *Order of Learning* project, learning order can only be evaluated as formation of reusable prerequisites once the architecture and the diagnostics distinguish independent attribute detection from binding-sensitive reasoning.

## Reproducible evidence

The executed [`architecture_localization.ipynb`](../notebooks/medium_world/architecture_localization.ipynb) defines and fits the three readouts and exports per-checkpoint `id_predictions.csv`, `edit_predictions.csv`, metrics, weights and provenance. The executed [`architecture_localization_analysis.ipynb`](../notebooks/medium_world/architecture_localization_analysis.ipynb) calculates and saves the following to `outputs/medium_world/experiments/architecture_localization/analysis/`:

- `id_metrics_all.csv`, `edit_metrics_all.csv`, `paired_readout_contrasts.csv`, `random_visual_contrasts.csv`;
- `count_controlled_binding_auc.csv`, `binding_swap_operator_leakage.csv`, `binding_swap_joint_consistency.csv`, `binding_conjunction_selectivity.csv`, `counterfactual_operator_response.csv`, `additional_paired_contrasts.csv`;
- diagnostic operator-flip matrices and Binding–Conjunction trade-off figures.

All numerical statements in this report derive from those executed notebooks and saved analysis products. No new model fitting, test-set evaluation, activation patching, CKA or TCAV was performed in preparing this report.
