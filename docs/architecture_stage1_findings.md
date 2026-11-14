# Architecture Protocol Stage 1 — Findings

**Project:** The Order of Learning · Medium World  
**Branch:** `experiment/architecture`  
**Status:** Exploratory selection and mechanism diagnostics completed; independent confirmation **not yet run**  
**Evidence:** Executed notebooks `00_architecture_evaluation_sets.ipynb` through `03_architecture_selection_analysis.ipynb`, including the final diagnostic extension, and saved per-question selection predictions. All figures and intervals in this note are **selection-set** results, not confirmation results.

## 1. Research question and setup

Medium World presents 64 × 64 RGB scenes with several colored shapes. The same scene can be queried about Color presence (C), Shape presence (S), independent Conjunction (J = C AND S, potentially on different objects), or same-object Binding (B). Earlier curriculum experiments did not establish a reliable final advantage for prerequisite-first training; Shape recognition and the distinction between J and B remained weaknesses.

This investigation tests two related but distinct interventions:

- **Spatial access and incentives:** freeze ten existing ViT checkpoints (Mixed and Prereq-Mixed × five seeds, 42–46), fit 3 × 3 spatial readouts and 1 × 1 controls, and vary effective Conjunction training weights. This measures information **accessible to a new readout**, not the original checkpoint's behavior.
- **Visual evidence:** train Patchify (A), parameter-matched Pointwise (B), and local-context Conv Stem (C) with the original shared head, canonical training data, 9,000 Mixed updates, and three paired seeds (42–44). An additional Shape-only experiment uses 2,250 updates.

The corrected **S-v1** set controls the number of queried-Color and queried-Shape objects in Conjunction cases. It includes 1,000 Conjunction questions (250 per stratum), 1,000 count-controlled Binding examples, 750 operator tuples, 500 recoloring pairs, 500 Binding swap pairs (400 ID-safe, 100 held-out-creating), and a repaired held-out set. The generated **C-v1 confirmation set has not been scored**. Scene/group dependence is respected in the interpretation.

## 2. Selection outcomes

| Gate | Criterion | S-v1 result | Interpretation |
|---|---|---|---|
| **G1** | Binding-heavy minus separate-heavy *bound-minus-separate Conjunction recall gap* ≥ 10 pp, ordered dose response, paired five-seed interval excluding zero | **+12.64 pp**, unadjusted paired 95% CI **[+3.45, +21.83] pp**; gate **passes** | The frozen 3 × 3 readout's association dependence is sensitive to training incentives. Not evidence of improved Conjunction discrimination. |
| **D1** | Conv Stem minus Patchify Shape balanced accuracy ≥ +5 pp, with Color and recoloring non-regression constraints | **+7.33 pp**, unadjusted paired 95% CI **[+4.45, +10.21] pp**, three seeds; gate **passes** | Conv Stem is selected for confirmation, **not yet adopted**. |

The paired Conv-minus-Patchify Shape gains are +8.1, +6.0 and +7.9 pp. The Conv-minus-Pointwise mean is +7.73 pp, providing a meaningful capacity control, although B and C differ in more than neighborhood access. All current intervals are exploratory and do not account for multiple primary tests or finite-item uncertainty.

## 3. Shape gains are substantial but concentrated

| Shape recognition on recoloring questions | Patchify | Pointwise | Conv Stem |
|---|---:|---:|---:|
| Overall balanced accuracy | 67.3% | 66.9% | **74.7%** |
| Crescent | 69.0% | 70.6% | **91.2%** |
| Circle / square / triangle pooled | 66.8% | 65.5% | **69.9%** |
| Both endpoints of a recoloring pair correct | 58.3% | 56.9% | **66.4%** |

The Conv-minus-Patchify gain is **+22.19 pp on Crescent queries**, versus **+3.06 pp** on the other three classes pooled. Shape-only training gives a similar architecture ordering (72.0%, 63.6%, 61.3% for Conv, Patchify, Pointwise respectively at 2,250 updates). Thus the advantage is not solely a by-product of multitask supervision, but neither does it establish broad Shape mastery. Triangle and Square remain comparatively difficult.

The single-instance diagnostic still favors Conv (Shape balanced accuracy **69.0% vs 60.4%**), so the gain is not explained simply by counting duplicate Shape objects. However, even with only one queried-Shape object, Color changes frequently flip the Shape answer: roughly **20–24%** of single-instance recoloring pairs across stems.

On the **63 held-out-creating swaps** in which the queried-Shape object receives an unseen Color–Shape pairing, Conv accuracy falls from **66.7% before** to **51.9% after** (Patchify 47.1% → 43.4%). Conv's advantage on these same pairs shrinks from roughly **+19.6 to +8.5 pp**. Its mean Shape Yes-logit change is **−0.94** versus Patchify's −0.36. These are narrow secondary slices, not independent confirmation. They reject a strong claim of color-invariant Shape evidence.

## 4. Conjunction discrimination improves, but correct composition is incomplete

Each corrected Conjunction stratum contains 250 questions per seed.

| Correct-answer rate | Patchify | Conv Stem | Difference |
|---|---:|---:|---:|
| Bound-positive Yes | 85.7% | 83.9% | −1.8 pp |
| Separate-positive Yes | 83.5% | 84.4% | +0.9 pp |
| Shape-missing No | 30.9% | 44.5% | **+13.6 pp** |
| Color-missing No | 100% | 100% | 0 pp |
| Four-stratum mean | 75.0% | **78.2%** | **+3.17 pp** |

Most of the net increase comes from correctly rejecting **Shape-missing negatives**: 19.9% of these questions are repaired and 6.3% regress in the paired Patchify-to-Conv comparison. The **threshold-independent AUC** distinguishing separate-positive scenes from Shape-missing scenes also rises **0.626 → 0.724** (gains positive in each seed), so the result is more than a favorable choice of the 0.5 threshold. Nevertheless, a Shape-only oracle would separate these two case types; this AUC is **not** proof of correctly computing logical AND.

The original head's Conjunction logit has approximately chance ability to distinguish *bound* from *separate* positives across count-controlled scenes (AUC about **0.50** for Patchify and **0.51** for Conv). This does **not** mean full counterfactual invariance: ID-safe swaps still change Conjunction logits when the correct J answer is unchanged. A population AUC and a paired intervention answer different questions.

## 5. Evidence availability versus evidence use

The new matched-scene analysis asks **Color, Shape and Conjunction on the identical scenes**, and reconstructs the exact signed head features from `s1`, `s2`, `sj` and `tanh(s1) × tanh(s2)`.

**Separating separate-positive from Shape-missing cases (ROC-AUC):**

| Score | Patchify | Pointwise | Conv Stem |
|---|---:|---:|---:|
| Standalone Shape-query Yes probability | 0.692 | 0.700 | **0.826** |
| Conjunction-query Yes probability | 0.626 | 0.640 | **0.724** |
| Conjunction-path `s2` maximum | 0.550 | 0.571 | **0.717** |
| Conjunction-path `s2` mean | 0.517 | 0.569 | 0.566 |

Conv provides more discriminative Shape information at the standalone query and in the Conjunction-path `s2max` feature. Yet its Conjunction score separates these cases less well than its standalone Shape score. Crucially, among **Shape-missing scenes where both standalone prerequisite answers are correct**, the Conjunction answer is still wrong about **45.8%** of the time (Patchify: **57.3%**). The denominator is the subset passing both prerequisite queries, not all Shape-missing scenes.

This is evidence for a **remaining evidence-integration or query-conditioning problem in addition to imperfect perception**. It does *not* isolate a faulty head causally: the three queries use different paths through a shared nonlinear model, and different scores need not be calibrated the same way. No head intervention was trained in Stage 1.

## 6. Binding and operator distinction remain unstable

| End-to-end outcome | Patchify | Conv Stem |
|---|---:|---:|
| Count-controlled Binding ROC-AUC | 0.636 | 0.709 |
| Three-case operator joint correctness | 53.2% | 57.6% |
| ID-safe eight-answer swap correctness | 11.6% | 17.3% |
| Repaired held-out Binding ROC-AUC | 0.535 | 0.522 |

The mean Binding AUC gain of **+7.30 pp** conceals paired seed differences **+13.58, +18.20 and −9.87 pp**; the unadjusted three-seed 95% interval spans zero widely. Both close and far negatives show this instability. The repaired held-out mean stays around chance, not supporting reliable new-pair generalization.

On discordant operator tuples (J = Yes, B = No), Conv correctly rejects Binding **68.7%** versus Patchify **53.9%**; the corresponding false-Yes rates are **31.3%** and **46.1%**. However, operator joint correctness is only 57.6%, and Conv loses accuracy on the both-Yes category relative to Patchify. Better differentiation in one logical case does not imply stable operator semantics.

Under the **400 ID-safe swap pairs** (Binding truth changes; C, S and J truth do not), mean Yes-logit shifts (after minus before) are:

| Query | Patchify | Conv Stem | Desired direction |
|---|---:|---:|---|
| Color | +0.01 | +0.01 | 0 |
| Shape | −0.05 | −0.13 | 0 |
| Conjunction | −0.35 | −0.32 | 0 |
| Binding | −0.56 | **−1.17** | Negative |

Conv's Binding response is stronger on average, but the undesired Conjunction response remains. Subtracting mean attribute-task logit shifts gives a useful **descriptive control**, not a causal measure, because all outputs pass through task-dependent nonlinear computations.

## 7. Frozen-readout incentive effects do not imply an end-to-end shortcut

The frozen spatial readout's bound-minus-separate recall gap changes monotonically with effective positive weights: about **25.5 pp** under binding-heavy, **19.5 pp** under balanced, and **12.9 pp** under separate-heavy. The bound-versus-separate Conjunction-score AUC also changes (about 0.695 → 0.617). But four-stratum Conjunction accuracy and separate-versus-Shape-missing discrimination show **no corresponding reliable improvement**.

With canonical readout weights, the 3 × 3 reader exceeds a capacity-matched 1 × 1 reader by **+9.71 pp in Binding AUC** but develops more association-dependent Conjunction behavior and worse discrimination of separate-positive versus Shape-missing scenes. These are **new readout** behaviors on frozen features, not evidence that the end-to-end models acquired the same shortcut. A G1 pass supports testing the end-to-end balanced objective as specified in Protocol v1; it does not establish its benefit.

## 8. Spatial maps: signed evidence and boundary artifacts

The extended notebook reconstructed and verified the four signed 16 × 16 maps and evaluated a fixed sample of 80 scenes per Conjunction stratum. Straight patch-level ROC-AUC of signed Shape maps is hard to interpret because some shapes are represented by **negative** scores on target regions; averaging opposite-polarity responses can approach 0.5 even when the representation is structured. A sign-free magnitude score would erase information that the actual pooling head uses.

The signed queried-Shape score contrast between target and non-target objects is positive for Conv on average, but uneven across Shape classes. More unusually, **46.8% of Conv joint-map maxima** and **32.8% of interaction-map maxima** land on the one-token-wide border (versus 3.2% and 1.7% for Patchify). The border occupies **23.4%** of tokens. This is consistent with a padding/boundary artifact; it does not prove padding causes the classification errors. Removing border tokens changes some localization summaries, but no controlled padding intervention was performed. Raw signed overlays should not be interpreted as segmentation or causal object representations.

## 9. Reproducibility and limitations

The later rerun **retrained all 18 stem runs and refit all 60 frozen readouts** after earlier output folders were deleted; it did not merely rescore original weights. S-v1 also overlaps substantially with the original evaluation for several families. Differences across those runs therefore do not isolate GPU nondeterminism: both weight realization and evaluation version changed. Preserve this history rather than treating v0 and v1 as independent confirmation.

Other limitations are three end-to-end training seeds, five frozen source seeds, repeated scene/pair endpoints, one fixed synthetic world, possible coupling of several stem design choices, 100 held-out-creating swaps, and no held-out model-based confirmation. Exploratory t intervals use **training seed** as the inference unit and omit **shared-item uncertainty** unless stated separately. Multiple plots were inspected on S; they generated hypotheses for C, not additional confirmed findings.

## 10. Conclusion and decision

**Supported now:** local-context Conv Stem makes Shape recognition more reliable under canonical training, with a very large Crescent-specific gain and a smaller non-Crescent gain. It improves certain Conjunction distinctions and reduces one type of Binding false positive. Both **G1 and D1 pass** on the corrected selection set.

**Not established:** color-invariant Shape knowledge, consistent Binding generalization, fully selective J/B behavior, a causal explanation for the padding pattern, or a curriculum benefit from the new stem.

**Decision:** freeze Conv Stem as the **selected candidate**, keep Patchify and Pointwise as controls, lock the confirmation plan in `architecture_confirmation_protocol_locked.md`, and run no further architecture search on S. Do not describe the stem as *adopted* until independent C results satisfy the predefined confirmation and safety criteria. The broader learning-order question remains for Stage 3.
