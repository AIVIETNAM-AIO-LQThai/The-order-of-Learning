# Medium World: Representation Failure Diagnostics

**Status:** Full five-seed diagnostic implementation; no training or diagnostic results claimed.  
**Branch:** `experiment/phase-aware-lr` (continue on existing branch; do not create a new branch).  
**Research question:** Do Shape and same-object Binding errors originate in inaccessible visual features, query-conditioned evidence selection, or object-level association?

## 1. Evidence motivating the experiment

Completed LR-reset and scaffolded-curriculum experiments show high Color performance, substantially weaker Shape performance, a Conjunction negative-case asymmetry (missing Color is much easier than missing Shape), and OOD Binding errors. These are **observed behaviors**. They do not identify where the representation fails. In particular, poor Binding does **not** by itself prove the model ignores Shape.

This experiment is **diagnostic**: no modifications to `AttributeBindingViT`, frozen dataset, augmentation used in baseline reproduction, answer oracle, or main task loss.

## 2. Research methods and their actual scope

- **Same-scene capability matrix:** For every sampled scene and target `(color, shape)`, construct Color, Shape, Conjunction, and Binding questions using the existing oracle. Logical constraints include `Binding ⇒ Color ∧ Shape ∧ Conjunction` and `Conjunction = Color ∧ Shape`. Measure both label correctness and prediction consistency. A consistent shortcut can still satisfy logic on some scenes, so logical consistency alone is not sufficient.
- **Counterfactual intervention:** Create validated No→Yes pairs by changing only the intended causal feature. `binding_swap` swaps two objects' colors, holding the global color and shape **multisets**, spatial positions, sizes and angles fixed. `shape_insert` changes only Shape **when the original rotation is valid for the destination shape** (otherwise skips the candidate), preserving requested Color presence and angle; `color_insert` changes a Color while preserving requested Shape presence. The renderer and oracle validate every edit. Compare *both-correct rate*, signed `P(Yes)` change, direction accuracy, false positives/negatives and unchanged-label controls.
- **Frozen probes:** Use 16×16-pixel fixed square neighborhoods around **oracle object centers** on the 16×16 patch grid, from (A) raw patch embeddings, (B) transformer output, and (C) normalized residual-fused features. Fit standardized multinomial logistic regression for four-class Shape and five-class Color. Never update the ViT weights during probing. Fit on unique Train scenes; evaluate on disjoint ID-dev and OOD-dev scenes. Compare randomized encoder and shuffled-training-label controls. Accuracy, macro F1, per-class confusion, object radius/rotation and per-scene support are required.
- **Network Dissection-style channel alignment:** The original Bau et al. method aligns unit activation masks with semantic segmentation. This package **adapts it to ViT patch channels**, computing each channel's 99.5th-percentile threshold on Train scenes and aggregated IoU with renderer-derived Color/Shape masks on disjoint ID-dev scenes. The masks include dark outlines and are downsampled to 16×16 patches using area coverage; **quantiles 0.99 and 0.995 are compared on the same features**; mask and activation thresholds are methodological choices to sensitivity-test. It does **not** replicate Broden/CNN Network Dissection, provide a causal test, or prove that unaligned channels lack encoded concepts. An axis rotation can remove single-unit alignment while preserving information.
- **SpRAy-inspired spectral explanation clustering:** The original Lapuschkin et al. method clusters **LRP relevance maps**, not raw images or feature vectors. We instead compute **signed gradient×(image−background)** attribution maps for the `YES - NO` logit margin, reduce to 16×16 and L1-normalize. Build a symmetric cosine-distance kNN affinity, inspect normalized graph-Laplacian eigenvalues/eigengaps, and use spectral clustering to browse recurrent visual explanations. Cluster within a fixed target **label** (otherwise Yes/No may dominate). **ID-dev explanations** use balanced present-pair positives and hard negatives (requested Color and Shape both present, requested Pair absent), excluding held-out targets; **OOD-dev explanations** use the frozen split's original held-out Pair queries, balanced by label. These are distinct question distributions and their cluster assignments must not be compared directly. This is explicitly an *adaptation*, **not** LRP/SpRAy replication. Inspect actual scene images and metadata for every cluster. Test stability across kNN neighborhoods (8/12/16), random initialized models and seeds before interpreting cluster semantics; the analysis reports adjusted Rand agreement for fixed K=3. Alternative K values remain a possible follow-up, not an automatically validated model-selection method.

Primary method sources: [Network Dissection (Bau et al., CVPR 2017)](https://openaccess.thecvf.com/content_cvpr_2017/html/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.html), [Unmasking Clever Hans predictors (Lapuschkin et al., Nature Communications 2019)](https://www.nature.com/articles/s41467-019-08987-4), [Designing and Interpreting Probes (Hewitt & Liang, EMNLP 2019)](https://aclanthology.org/D19-1275/), [Sanity Checks for Saliency Maps (Adebayo et al., NeurIPS 2018)](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html).

## 3. How the original model determines what we can test

`AttributeBindingViT` embeds a query as `[operator, attribute1, attribute2/PAD]` and projects each of 256 4×4-image patches into 48 dimensions. After a one-layer transformer, it produces eight pooled scalars: max/mean of score for attribute1, attribute2, joint query, and their patchwise interaction. A shape appears in **different attribute slots** in Shape-only versus Conjunction/Binding queries. This motivates testing both shape feature accessibility and whether query roles/pooling produce a bottleneck. The implementation does **not** relabel operator/attribute positions, which would create out-of-training queries.

## 4. Execution stages and source isolation

| Phase | Notebook | Artifacts | What it may establish |
|---|---|---|---|
| 0. Reproduce checkpoints | `representation_checkpoints.ipynb` | `checkpoints/*.pt`, `checkpoint_manifest.csv`, `checkpoint_reproduction_audit.csv` | Baseline models available for inference; provenance and comparison to archived traces |
| A. Same-scene matrix | `representation_probing.ipynb` | `same_scene_*.csv`, `logical_consistency_*.csv` | Conditional task failures and logical inconsistency on identical scenes |
| B. Interventions | same | `counterfactual_*.csv`, `invariance_*.csv` | Behavioral sensitivity and failure to discriminate paired scenes |
| C. Linear probing | same | `probe_metrics_*.csv`, `probe_predictions_*.csv` | Readily decodable information at each frozen stage, with controls |
| D. Channel alignment | same | `dissection_*.csv` | Axis-aligned channels overlapping Color/Shape regions |
| E. Spectral explanation clusters | same | `explanation_maps_*.npz`, `explanation_clusters_*.csv`, `explanation_eigenspectrum_*.csv` | Recurrent attribution patterns requiring manual and controlled validation |
| F. Interpretation | `representation_probing_analysis.ipynb` | `joint_subgroup_accuracy.csv`, `counterfactual_summary.csv`, `dissection_best_concepts.csv`, figures | Which explanations remain plausible; explicitly unproven claims |

**Full predefined run:** Seeds **42–46**, both curricula (**Prerequisite-Mixed → Pair Focus** and **Fully Mixed**, Constant LR), and all three checkpoints (**5,400 / 7,200 / 9,000**). This is **10 trained models / 30 checkpoint evaluations**, all enabled from the outset. No one-seed pilot or staged expansion. Compare trajectories within each run and variations across five seeds.

**Checkpoint warning:** The original LR experiment stored prediction traces but not state dictionaries. The checkpoint notebook reproduces Constant-LR models using the unchanged model/training components, saves source-code and dataset hashes, and audits question-level predictions versus archived Constant-policy traces; a seed alone does not guarantee bitwise-equivalent reruns. The saved model-only checkpoints are intended for inference, not optimizer continuation.

The generated all-target grid includes combinations excluded from the original training question generator, including held-out target pairs and both-absent Conjunction questions. The grid repeats Color-only questions once per shape and Shape-only questions once per color; **deduplicate using `question_key` within (schedule, seed, step, split)** for task-accuracy summaries. The full Cartesian grid remains necessary for same-scene logical-consistency checks. Mark held-out-pair and `logic_case` metadata as not applicable for single-attribute task summaries. Every aggregate must be stratified by `heldout_target` and `logic_case`; do **not** label this expanded diagnostic grid as original ID-dev accuracy.

**Data:** Reuse immutable `outputs/medium_world/dataset/{train,id_dev,ood_dev}.jsonl` with hash checks. Scene-disjointness is mandatory for both diagnostic questions and object-level probes. **Never train probes on the dev/test scenes.** Reserve ID-test and OOD-test for the later architecture assessment.

## 5. Required post-experiment analysis

Every analysis must print actual denominator counts and retain per-scene records.

1. **Integrity:** exact hashes, checkpoint identity, train/dev scene disjointness, 3–5-object oracle validity, number of interventions accepted/rejected, label/subgroup balance, target-shape/rotation/radius coverage, evaluation per checkpoint and seed.
2. **Behavior:** accuracy, cross-entropy or probabilities where meaningful, Color/Shape missing-case specificity, Shape × task × label, all logical consistency violations, error when correct individual attributes still yield wrong Binding.
3. **Interventions:** per-edit type, target shape, seed and checkpoint: `N pairs`, both-correct, correct-direction, mean/median signed probability shift, pre/post errors, per-scene clustered confidence intervals. Explain no-change cases rather than treating them as correct by default.
4. **Probes:** Shape/Color accuracy and macro-F1 by layer, split, seed, class, rotation, radius. Contrast trained and random encoder controls, true-label and shuffled-label probes. Evaluate local region sensitivity; linear probe failure does not imply nonexistence of nonlinear information.
5. **Dissection:** best IoU by concept and channel, number of aligned units, rendered source image, concept mask, activation heatmap and binarized activation mask. Compare random networks and threshold sensitivity. High alignment is correlational, not a causal intervention.
6. **Explanation clustering:** raw signed maps, eigenspectrum/eigengaps, stable membership under different neighbor and cluster counts, cluster prevalence and correctness, positive/negative split, representative original images plus maps, shape/color and other scene attributes. Beware clustering by location, shape frequency, or query-label bias. Do not infer hidden mechanisms from cluster names alone.
7. **Uncertainty:** seed-matched contrasts as the primary replication unit; scene-bootstrap as secondary uncertainty (resample scene IDs, not derived questions). Do not treat thousands of generated questions as independent test samples. Mark all subgroup analyses exploratory unless pre-registered.
8. **Decision:** explicitly describe support, counterevidence, and remaining ambiguity for H1 visual encoding, H2 query/use, H3 same-object association.

Suggested operational minimum for reporting: ≥30 **distinct scenes** per subgroup, five seeds for reproducible contrasts, and full confusion matrices. If smaller, describe the subgroup descriptively and mark uncertainty as unresolved. These are practical analysis rules, not statistical significance guarantees.

## 6. How to interpret outcomes without overclaiming

| Evidence pattern | Provisional conclusion | What is NOT proved |
|---|---|---|
| Shape probe weak at patch, transformer, and fused stages, compared with chance/random controls | Shape poorly accessible to a **linear** local decoder | Complete absence of Shape or causal encoding failure |
| Shape probe strong at fused stage, Shape-only behavior weak, Shape counterfactual insensitive | Query selection/head may fail to use available Shape | Pooling is causally responsible |
| Shape and Color question responses strong, Binding swap both-correct weak | Object association is behaviorally weak | A particular internal binding mechanism is at fault |
| Strong channel IoU for Color and weak for Shape | More axis-aligned Color detectors than Shape under chosen threshold | Shape absent from distributed features |
| SpRAy cluster associated with wrong answers or background | Repeated explanation-map pattern associated with errors | Reliance on that region without a masking/counterfactual intervention |

If findings indicate a specific bottleneck, design a **single-factor architecture ablation** next, with old Mixed and Prerequisite-Mixed curricula rerun on the modified architecture. Do not use the diagnostic Dev splits as a final test-set claim.

## 7. Dependencies and run order

Check whether the kNN explanation graph has multiple disconnected components. A large eigengap or tight cluster may reflect component structure, visual layout, or query labels rather than a meaningful learned strategy. Compare trained and random-model explanation maps.

Use the existing Medium World environment with `torch`, `numpy`, `pandas`, `Pillow`, `scipy`, `scikit-learn`, `matplotlib`, `nbformat`. Install missing packages into that environment; no external Broden dataset, Captum, LRP toolbox, or remote API is needed for this full run.

From repository root:

```powershell
python scripts/medium_world/audit_representation_probing.py
```

Then open `notebooks/medium_world/representation_checkpoints.ipynb` from the notebook directory and run it. Follow with `representation_probing.ipynb`, then `representation_probing_analysis.ipynb`. Keep runs with different seeds/policies in separate artifact names; never overwrite the LR-reset `schedule_comparison.json` or original traces.
