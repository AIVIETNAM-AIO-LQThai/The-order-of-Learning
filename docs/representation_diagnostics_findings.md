# Representation Diagnostics: What the Model Encodes, Uses, and Fails to Associate

*November 13, 2026*

## 1. Why task accuracy was no longer enough

The curriculum experiments had established a recurring pattern: Color recognition became highly reliable, Shape recognition remained substantially weaker, and concentrated Binding training could improve Binding while damaging performance on earlier tasks. A return to mixed training often restored those earlier-task scores but reduced some of the Binding gain. These movements were observable in the predictions, but their cause was not identifiable from accuracy curves.

Three explanations could produce superficially similar results:

1. **Representation failure:** the visual encoder does not preserve adequate information about geometric Shape or about which attributes belong together.
2. **Evidence-access failure:** useful information exists in the visual features, but the query-conditioned scoring, spatial reduction, or classifier does not extract and use it reliably.
3. **Task-dependent adaptation:** the internal features and decision pathway both change during training; improvement on a newly emphasized task may alter how evidence is used for older tasks without completely eliminating that evidence.

The representation-diagnostics investigation was designed to distinguish the *observable implications* of these explanations, not to assume one of them in advance. It examined the unchanged `AttributeBindingViT` through complementary measurements: related questions on the same scene, controlled image edits, frozen-feature probes, channel–concept alignment, explanation-map clustering, and a refined Shape probe that preserved local spatial layout.

The principal result is not that the encoder definitively lacks Shape, nor that the final shared head definitively causes forgetting. Rather, **the ability to recover visual information from a representation is different from the ability of the model's own query-to-answer pathway to use that information correctly**. Several initial diagnostics appeared to indicate weak Shape representation; their refinements revealed how much those conclusions depended on what the diagnostic decoder was allowed to see.

## 2. What was compared, and why the comparison is informative

The investigation reused the frozen Medium World and the existing 64×64-image model. It examined two Constant-LR training histories over seeds 42–46:

- **Mixed:** all four tasks are encountered throughout training.
- **Prerequisite-Mixed → Pair Focus:** Color, Shape, and Conjunction are trained together until step 5,400; Binding receives concentrated training until step 7,200; all four tasks return during the final 1,800-step mixed consolidation.

For each of the ten trained runs, the analysis used checkpoints at steps **5,400, 7,200, and 9,000**, giving **30 checkpoints**, not 30 independently trained networks. The same scene/question sets and intervention definitions were reused across models. The diagnostic probes fitted new classifiers to **frozen** feature tensors; fitting a probe did not further train the ViT.

The three checkpoints matter because they correspond to distinct questions:

- At **5,400**, what has been learned before the Prerequisite-Mixed model receives Binding-specific training?
- At **7,200**, what changes when Binding becomes the immediate objective?
- At **9,000**, which behaviors survive or reverse when all tasks are trained together again?

The study assessed existing development scenes and derived diagnostic questions, not a fresh final test set. The same five seeds are the main replication units. Derived scene–query combinations, object crops, and paired edits are correlated observations, not thousands of independent model replications.

### The decision pathway being diagnosed

The visual encoder maps the image into 256 patch features of width 48. The query supplies an operator and up to two attributes. For patch $u$, the model computes scores for the first attribute, second attribute, and complete query:

$$
s_1(u)=\frac{v_u^\top e_1}{\sqrt{48}},\qquad
s_2(u)=\frac{v_u^\top e_2}{\sqrt{48}},\qquad
s_J(u)=\frac{v_u^\top q}{\sqrt{48}}.
$$

It also computes a local interaction:

$$
r(u)=\tanh(s_1(u))\tanh(s_2(u)).
$$

The prediction head receives only the maximum and mean of each of these four maps:

$$
h=[\max s_1,\operatorname{mean}s_1,\max s_2,\operatorname{mean}s_2,
\max s_J,\operatorname{mean}s_J,\max r,\operatorname{mean}r].
$$

Thus, the final classifier sees **eight scalar summaries**, not the full spatial feature tensor. The interaction is computed locally *before* pooling, so the architecture is not inherently blind to spatial co-occurrence. Nevertheless, it has no explicit object slots, and spatial arrangements producing similar maxima and means can be difficult to distinguish in the final head. This is a plausible limitation, not an experimentally isolated cause of failure.

The distinction between **Shape evidence present in $v_u$**, **Shape evidence represented in the query-conditioned scores**, and **Shape evidence actually used by the final decision** guides the interpretation of every diagnostic below.

## 3. The original benchmark and the expanded question grid measure different things

The reproduced final checkpoints had the following mean performance on the **original saved development questions**:

| Task or evaluation | Mixed | Prerequisite-Mixed → Pair Focus |
|---|---:|---:|
| ID Color | 99.22% | 99.38% |
| ID Shape | 67.81% | 66.88% |
| ID Conjunction | 79.77% | 80.70% |
| ID Binding | 62.03% | 63.05% |
| Original OOD Binding | 56.45% | 57.25% |

Color is much stronger than Shape. Conjunction appears comparatively successful, but its negative-label construction rewards responding to Color presence without reliably checking Shape. Binding is only modestly above chance on its balanced original questions.

To test what these aggregate scores hide, the analysis queried each scene about **every Color–Shape target** and evaluated Color existence, Shape existence, independent Conjunction, and same-object Binding for that same target. This enables logical comparisons that cannot be made when separate task benchmarks use different scenes or target attributes.

For a target $(c,s)$, the intended relationships are:

$$
B(c,s)\Rightarrow C(c)\land S(s),\qquad
B(c,s)\Rightarrow J(c,s),\qquad
J(c,s)=C(c)\land S(s).
$$

The extended matrix contains 10,240 scene–target combinations on ID-dev per checkpoint, before expanding those combinations across tasks. It is useful for testing whether a model's *own answers* respect the relationships above. It is **not** interchangeable with the balanced Binding benchmark.

In particular, the ID-dev all-target grid has 1,835 true Binding positives and 8,405 negatives. An always-No rule obtains:

$$
\operatorname{Accuracy}_{\mathrm{always\ No}}=\frac{8405}{10240}=82.08\%.
$$

The models' approximately 73–75% Binding accuracy on that expanded grid is **below** this trivial raw-accuracy baseline. It would be incorrect to advertise it as improved Binding performance relative to the original 62–63% benchmark. The class distribution and target-selection procedure have changed. The expanded grid's primary value is *matched questions and logical consistency*, not raw Binding accuracy.

### Predictions violate relationships they should obey

At step 9,000, averaged across five seeds, the following fractions of scene–target combinations were logically inconsistent:

| Prediction-level violation | Mixed, ID-dev | Prerequisite-Mixed, ID-dev |
|---|---:|---:|
| Predict Binding Yes while predicting Color or Shape No | 14.41% | 12.47% |
| Predict Conjunction differently from predicted Color AND predicted Shape | 21.98% | 20.59% |
| Binding answer wrong despite correct Color, Shape and Conjunction answers | 10.88% | 10.94% |

These are contradictions **among the model's predictions**, not statements that the underlying oracle labels are inconsistent. They show that the learned operators do not compose into a globally coherent rule across related queries.

One especially revealing error is that the model can correctly reject an absent Shape in a Shape-only question, yet answer Yes when the same absent Shape is requested together with a present Color. This is difficult to reconcile with a model that simply applies a shared, reliable Shape-existence decision inside a learned logical AND. It is compatible with task-conditioned evidence use or a Color-dominant decision rule.

However, a logical violation does not by itself locate the defect. The model could contain incorrect visual features, operator-dependent joint features, or an inconsistent readout. We needed direct evidence about how the prediction changes when the scene itself changes.

## 4. Controlled edits reveal a pronounced Color–Shape asymmetry

The counterfactual study changed a scene while preserving specified properties and validating the answer with the symbolic oracle. Three edits tested different capabilities:

| Edit | What changes | What stays fixed | Intended label change |
|---|---|---|---|
| **Color insertion** | Recolor an object to introduce the requested missing Color | Object Shape and geometry; requested Shape already exists | Conjunction No → Yes |
| **Shape insertion** | Change an object's Shape to introduce the requested missing Shape | Requested Color exists; object Color, location and valid rotation remain | Conjunction No → Yes |
| **Binding swap** | Swap Colors between two objects so the queried pair is formed | All object positions and Shapes; full Color and Shape multisets | Binding No → Yes |

The two complementary outcomes answer different questions:

- **Correct direction:** does the probability of Yes increase after a No→Yes edit?
- **Both endpoints correct:** does the model answer No before the edit **and** Yes after it?

The latter is much stricter; a correctly directed probability change does not guarantee either hard prediction crosses the threshold.

### Final-checkpoint results on ID-derived interventions

| Edit | Mixed: both correct | Prerequisite-Mixed: both correct | Mixed: correct direction | Prerequisite-Mixed: correct direction |
|---|---:|---:|---:|---:|
| Color insertion | 88.20% | 88.55% | ~100% | ~100% |
| Shape insertion | 8.33% | 9.42% | ~64% | ~67% |
| Binding swap | 23.27% | 24.82% | ~64% | ~67% |

The Color-insertion result is consistent with the model's strong Color detection. Shape-insertion and Binding-swap strict success are much weaker.

But **88% versus 8% is not a clean causal comparison of Color and Shape detection**. Every edit starts from a negative example, and the strict metric requires the original negative to be correct. A Shape-insertion case begins with Color present and Shape absent—the exact Conjunction-negative subgroup on which the model frequently predicts Yes incorrectly. In a subsequent analysis of these *same intervention records*, only about 16.5–19.3% of those starts were correctly rejected; among the correctly rejected Shape-insertion cases, approximately half were correct after the edit. Color insertion succeeded after a correct start approximately 89% of the time. The large raw contrast therefore partly reflects a **different starting-state error**, not only the response to the edit.

The two edit populations also differ in eligible scenes, permitted rotations and geometric changes. They are not two randomly assigned treatments on an otherwise matched sample. What survives these cautions is a meaningful asymmetry: Color addition produces a large and nearly always correctly directed response, whereas changing Shape produces weaker, less reliable responses.

### Binding swaps are more diagnostic than ordinary Binding accuracy

A Binding swap leaves every requested attribute's *total count* unchanged. A rule checking only independent Color presence, independent Shape presence, or their global multiplicities would receive no new information from that edit. Yet the models do show a correct probability-direction response on roughly two-thirds of the original ID swap examples.

This demonstrates some **association-sensitive response** to changes in which attributes share an object or local image region. It does not show reliable Binding: only about one-quarter of swaps are correctly classified at both endpoints at the final checkpoint.

A local texture, edge, or Color–Shape co-occurrence detector could react to the swap without constructing explicit object identities. The experiment therefore supports **partial functional association sensitivity**, not symbolic object-centered reasoning. Nor does it by itself exclude all scene-level or position-dependent cues.

### Invariance controls distinguish responding to an edit from responding appropriately

Other questions retain their oracle label under the same intervention. For example, swapping object Colors must not change whether a requested Shape exists; recoloring an object does not alter its Shape.

At step 9,000, the original diagnostics found:

- About **18–19%** of Shape-existence predictions flipped after Color swaps.
- About **22–23%** flipped when the queried Shape object's Color was changed.
- Only about **1%** of Color-existence predictions flipped after Shape insertion.

The initial interpretation was that Shape answers were more entangled with Color. A later reanalysis of the *same saved invariance outputs* made the claim more precise. Shape answers are generally much less confident, so a simple comparison of **all binary flips** exaggerates the asymmetry. Nevertheless, when the analysis distinguishes previously correct answers that become incorrect from originally incorrect answers that become correct, Shape remains more vulnerable to Color-only edits. Confidence-standardized comparisons found a substantial right→wrong difference, although strict matching to the same pairs and same confidence bins yielded a smaller contrast, especially among confident predictions.

The defensible conclusion is **behavioral dependence of Shape decisions on Color changes that do not alter Shape**, strongest when Shape evidence is already uncertain. This is not, by itself, proof that the visual encoder has learned a particular entangled feature.

An even more specific post-hoc analysis of the archived recoloring pairs found that changing an object into one of the **four held-out Color–Shape combinations** reduced the Shape-answer logit by approximately **1.08** relative to permitted recolorings of the same original scene and object at step 9,000. The same-scene contrast was negative across all five seeds. The effect was stronger when that object was the only instance of the queried Shape.

This is evidence that Shape-answer behavior depends on the **Color–Shape combination**, not just Shape geometry. Its origin remains unresolved: the saved behavior alone cannot determine whether training produced the dependence or which layer expresses it. The result also does not make the original OOD benchmark a clean test of transfer.

## 5. Head-feature inspection narrows—but does not localize—the problem

Because the image features are computed independently of the query, the same learned Shape embedding produces the same raw patchwise attribute similarity whether Shape appears as the first attribute in a Shape-only query or the second attribute in a Conjunction query.

The saved audit confirmed **zero difference** between those corresponding max/mean Shape scores on matched images. This verifies the implementation, but it is predicted by the architecture; it is **not** a discovery that Shape recognition transfers successfully across task operators.

The inputs to the final head still differ:

- Shape-only and Color-only questions place their active attribute evidence in the same first two scalar positions.
- Conjunction and Binding place Shape evidence in the second-attribute positions.
- Operator-dependent joint-query scores differ.
- The same nonlinear MLP must interpret these distinct task contexts.

Thus identical underlying Shape scores need not lead to consistent Shape use across questions.

A single-scalar discrimination analysis at step 9,000 for Prerequisite-Mixed, averaging the five seeds, found approximately:

| Scalar evidence | Task comparison | Sign-invariant AUC |
|---|---|---:|
| First-attribute maximum | Color presence | 0.887 |
| First-attribute maximum | Shape presence | 0.604 |
| Second-attribute maximum | Hard-negative Binding | 0.527 |
| Joint-query maximum | Hard-negative Binding | 0.571 |
| Local interaction maximum | Hard-negative Binding | 0.526 |

The nonlinear head using all eight inputs achieved an approximately **0.71–0.72 AUC** in the corresponding hard-Binding analysis. Its combination of signals is more informative than any one listed scalar.

These numbers **do not** provide causal feature importance. Low marginal AUC does not show that an input is unused or worthless in combination with others. Nonetheless, the relative weakness of the Shape and interaction summaries makes it reasonable to examine whether relevant spatial information survives *before* these scores are aggregated.

## 6. The original frozen Shape probe was not measuring all accessible geometry

The first probe froze the trained visual features and fitted an independent, supervised linear classifier to recognize the Shape of an **oracle-selected object**. The object location was used to select its local feature region, and features within that region were averaged into a 48-dimensional descriptor. A separate probe was trained for each model stage and target attribute.

This distinction matters: an oracle-centered object classifier already knows *where to look*. The original scene-level VQA model does not receive object centers. Meanwhile, averaging the local feature region removes the relative positions of its constituent patches.

At step 9,000, the ID object-level Shape probe reached approximately:

| Frozen features and probe | Mixed | Prerequisite-Mixed |
|---|---:|---:|
| Random-encoder fused features | 50.84% | 50.84% |
| Trained-encoder fused features | 53.23% | 53.93% |

The trained-minus-random difference was small, with five-seed intervals spanning zero. By contrast, Color probes using similar object-centered feature access achieved approximately 99.9% on fused features even with a random encoder.

This initial Shape result showed that **geometry was not easily decoded from an averaged 48D region vector**. It did **not** establish that no useful Shape information existed in the original 256×48 visual tensor.

The controls reinforced that caution. Shuffling training labels brought Shape probing close to its four-class chance baseline, confirming that the fitted classifier used label information. But random visual features themselves preserved enough pixel content for strong Color decoding. Probe performance must therefore be attributed jointly to the frozen feature extractor, the privileged region selection, and the new supervised decoder—not to ViT training alone.

The mean-pooled probe also exhibited severe errors on withheld Color–Shape combinations. In the original-distribution probe, recognition on the held-out combinations was much worse than on permitted combinations. This could reflect a poor encoder, but it could also reflect the probe's own opportunity to learn Color–Shape correlations from its training examples. The next analysis was designed to separate those two issues more carefully.

## 7. Refined probing: preserving spatial layout changes the conclusion

The refined probe kept the ViT frozen and sampled a **fixed 5×5 grid of 48D features** centered on the oracle object location. It then compared two ways of presenting these same local features to a supervised decoder:

- **Mean readout:** average the 25 vectors into 48 values.
- **Spatial readout:** flatten the ordered 5×5 grid into 1,200 values.

Both were evaluated after fitting on original training scenes and after a separate diagnostic fitting intervention that recolored training scenes and balanced Shape×Color training weights.

This was a major measurement correction: the original result had tested **a particular spatially lossy readout**, not all representations of Shape potentially accessible in the frozen ViT.

### The 2×2 comparison

Five-seed means for the final trained fused features were:

| Probe-training distribution | Readout | ID-dev object Shape | OOD-dev object Shape |
|---|---|---:|---:|
| Original | Mean, 48D | 52.0–52.5% | 44.0–44.3% |
| Original | Ordered spatial, 1,200D | **79.5–79.6%** | 71.8–72.0% |
| Recolored and balanced | Mean, 48D | 47.5–47.8% | 46.4–46.8% |
| Recolored and balanced | Ordered spatial, 1,200D | **75.2–75.5%** | **74.6–75.5%** |

The improvement from retaining the ordered spatial grid was about **27–29 percentage points**, depending on the condition. This is much larger than the trained-versus-random difference in the original mean-pooled probe.

The spatial readout has 25 times as many input features, however. In this original refinement, **input dimensionality, decoder capacity and spatial preservation changed together**. The result demonstrates that the chosen spatially preserved decoder can recover more Shape information; it does not cleanly measure the causal effect of preserving spatial order *alone*. The later follow-up investigations were designed to examine that distinction more directly.

### Why the random encoder matters

The random fused-feature spatial probe reached approximately **79.4% ID-dev**, essentially the same as the trained models' approximately **79.5%**. The trained encoder therefore showed no meaningful general advantage under this particular privileged spatial decoder.

A random 4×4 patch projection of width 48 can retain a great deal of raw input information: a 4×4 RGB patch itself contains 48 scalar values, and a generic random 48×48 projection need not discard those values. The new supervised 1,200D decoder can learn to recognize geometric patterns from that preserved input.

The result does **not** show that a randomly initialized ViT can answer the original Shape-existence questions. Nor does it prove that training never learned Shape-related information. It shows that **high performance by an external Shape probe is not sufficient evidence of semantic representation learning by the ViT**, because a similarly expressive probe already performs well on random features.

### Why recoloring improves held-out Shape decoding but not the original model

For the four withheld Color–Shape combinations, the original-distribution spatial probe performed unevenly and poorly on average—roughly **34–35%** for trained features. After fitting the probe on recolored, Shape×Color-balanced diagnostic scenes, accuracy on those combinations rose to roughly **79–80%**.

This is a substantial reduction in *the probe's* reliance on the original Color–Shape training associations. It cannot be described as zero-shot generalization by the diagnostic classifier: recolored probe-training scenes now expose it to the combinations withheld from original ViT training. The ViT itself was never retrained.

Moreover, the recoloring procedure changes the probe-training support while inverse-frequency weights change the effective fitting objective. These effects were not isolated from one another. The executed standardization stage also was not sample-weighted. Therefore the result should be described as the effect of the **combined diagnostic fitting procedure**, not proof that recoloring alone or reweighting alone caused the improvement.

An additional foreground-silhouette control achieved **100%** four-way object Shape accuracy on the inspected oracle-centered object crops. It uses a known object center and the simple high-contrast renderer. This shows that rendered geometric information is available under privileged access; it is not evidence that the deployed scene-level ViT learned to localize and identify arbitrary objects perfectly.

### Geometry-specific limitations remain

Even after refining the readout, errors were not uniform across Shape and radius. Small squares, large triangles, and some small circles were difficult under various probe settings. The model and diagnostic probe operate on shapes whose visual footprint changes with geometry; radius alone is not the rendered area, because the shapes occupy different proportions of their enclosing regions.

This heterogeneity matters because apparent Shape weakness could reflect a mixture of geometry-specific ambiguity, Color–Shape correlations, locality and readout design. A single aggregate Shape score hides those mechanisms.

## 8. Channel alignment and explanation clustering did not identify a reliable latent mechanism

Two further analyses examined whether individual channels or visual explanations aligned with interpretable concepts. Both were useful as **negative or non-decisive diagnostics**, not as proof of how the model reasons.

### Network Dissection-style channel alignment

A thresholded-channel analysis compared visual feature activations with renderer-derived Color and Shape masks, using train scenes for activation thresholds and separate ID-dev scenes for overlap. The method measured best-channel intersection-over-union (IoU) in patch space, including random-encoder controls.

For fused features at step 9,000 with the 99th-percentile activation threshold, mean best-channel IoU by concept family was approximately:

| Concept family | Mixed random | Mixed trained | Prerequisite-Mixed trained |
|---|---:|---:|---:|
| Color | 0.207 | **0.227** | **0.232** |
| Shape | 0.073 | **0.079** | **0.079** |

At the 99.5th percentile, both families had lower absolute overlap: roughly **0.124–0.126 trained vs. 0.108 random for Color**, and **0.047 trained vs. 0.041 random for Shape**.

The ordering is consistent with strong Color evidence and weak localization of geometric concepts by **individual thresholded channels**. But best-channel IoU is affected by concept-mask size, activation threshold, patch resolution and selection of the best among many channels. Distributed or rotated representations can encode concepts without yielding an aligned individual unit.

Therefore the result cannot justify “the encoder has no Shape representation.” It supports only the narrower statement that **single-channel spatial alignment to Shape masks is weak under the tested criteria**.

### SpRAy-inspired explanation clustering

The analysis also formed signed gradient×(input−background) explanations for the Yes-minus-No logit margin, reduced them to spatial maps, and clustered those maps using a spectral nearest-neighbor graph. ID and OOD examples were analyzed separately, as were true Yes and No questions.

All inspected neighbor graphs were connected and had no entirely zero explanation maps. Those checks exclude gross numerical degeneracy but do not establish semantic validity.

The more important checks were negative:

- Cluster agreement after changing the number of graph neighbors was only moderate and varied across runs. Mean adjusted Rand Index was roughly **0.35–0.47** across the reported conditions.
- Association of clusters with target Shape, Color and prediction correctness was generally weak; target-Shape adjusted mutual information was often about **0.00–0.03**.
- No stable, reproducible partition into recognizable “Color-only,” “Shape-aware,” or “Binding-aware” explanation strategies emerged.

This was **SpRAy-inspired**, not a reproduction of the original method using layer-wise relevance propagation. The chosen attribution multiplies by the pixel's difference from the background, which makes exactly background-colored pixels contribute zero by construction. A blank-looking background in these maps is therefore not evidence that the model does not use contextual or background cues.

Clustering is also sensitive to the explanation method, graph construction and arbitrary cluster labels. These maps do not provide an independent causal localization of the errors uncovered by the behavioral interventions.

## 9. Training stages show why “forgetting Shape” is an incomplete explanation

The Prerequisite-Mixed checkpoints permit direct comparison of one model's behavior before Binding specialization, after it, and after common Mixed consolidation.

| Measurement | Step 5,400 | Step 7,200 | Step 9,000 |
|---|---:|---:|---:|
| Original ID Binding accuracy | 50.16% | **67.50%** | 63.05% |
| Original ID Shape accuracy | 64.38% | **57.11%** | 66.88% |
| Original ID Conjunction accuracy | 80.55% | **70.47%** | 80.70% |
| ID Binding-swap both-correct | 3.45% | **32.24%** | 24.82% |
| Original mean-pooled contextual Shape probe | 54.77% | 55.04% | 55.35% |

During the Binding-focused phase, original Binding improves by approximately **17.34 percentage points**, with a five-seed paired 95% interval of about **[+11.85, +22.84]**. Strict Binding-swap success improves by about **28.80 points**, while Shape and Conjunction performance decline. During final consolidation, Shape recovers by about **9.77 points**, Conjunction by about **10.23**, and both Binding accuracy and swap success decline from their specialized peaks.

These are **behavioral** changes. They do not directly measure the destruction and reconstruction of internal Shape features. The relative stability of the mean-pooled Shape probe across phases is compatible with a model retaining some linearly decodable information while changing how its trained query and decision pathway uses it. But the probe itself is weak and spatially lossy, so its stability alone does not establish intact representations.

The refined spatial probes further weaken a simplistic “Shape information disappeared during Binding training” account: substantial geometry remains recoverable from frozen visual tensors. Yet because those probes have privileged object centers, large external decoders and their own supervision, they do not prove that the *original* Shape decision pathway could have recovered that information without additional learning.

The most accurate interpretation is therefore a **specialization–retention tension in observed decision behavior**, with unresolved contributions from encoder changes, query-conditioned scoring, spatial compression and shared decision boundaries.

## 10. What the investigation actually established

The results collectively warrant several distinct conclusions.

**First, good performance on one task does not certify the expected prerequisite for another.** Color is near ceiling, but Conjunction can approach 80% by following a Color-dominant rule; independent Shape answers and Conjunction answers are frequently inconsistent. A curriculum stage evaluated only by aggregate accuracy may advance without having established a reusable visual skill.

**Second, the model's decisions react to some same-object association changes beyond independent attribute presence.** Binding swaps change local Color–Shape ownership while preserving attribute multisets, and the Yes probability often moves in the expected direction. Strict correctness is much lower. The capability is partial and fragile, not an established object-indexed representation.

**Third, Shape weakness is partly a question of information access and measurement.** A poor 48D mean-pooled probe did not show that the full visual tensor lacked geometric information. An ordered spatial probe recovered much more, but equally well from random and trained features. Thus representation *decodability*, learned semantic abstraction and the model's own deployed ability must not be equated.

**Fourth, trained Shape answers depend on Color in situations where the correct Shape label is unchanged.** Basic invariance flips exposed this dependence; later stratification and same-scene recoloring of the same archived examples showed particularly strong suppression on held-out Color–Shape combinations. The behavioral signature is real, while its origin and network location remain unverified.

**Fifth, neither individual-channel dissection nor explanation clustering established a clear latent reasoning mechanism.** Their failure to reveal stable Shape channels or semantic clusters is not proof that such information is absent; it is a limit of the chosen diagnostic procedures.

**Finally, the original OOD Binding results cannot independently establish compositional transfer.** All original OOD questions target withheld combinations that were absent from training queries, and the OOD generator samples positive scenes from the full pair inventory but negatives from permitted pairs only. Consequently, the presence of *any* held-out object predicts the original OOD label perfectly at the oracle-metadata level. The numerical OOD accuracies are valid for that benchmark but cannot isolate unseen-combination reasoning. Expanded OOD question-grid results have a different class distribution again and must not be read as a repaired transfer test.

These conclusions narrow the scientific claim. The study did not show that a learned object representation is absent, that pooling is **the** cause of failure, or that task-head competition is **the** mechanism of forgetting. It showed why ordinary task scores and one weak probe were inadequate to decide among those explanations.

## 11. How the findings led to the next diagnostic questions

The representation study left several specific doubts rather than a general demand for a bigger model.

**Does apparent Binding success survive removal of count-related cues?** Ordinary hard negatives contain both requested attributes, but requested Color/Shape multiplicities can still correlate with Binding labels. The original swaps help because they preserve counts, yet low strict accuracy and changes in edit eligibility leave the reliability of the acquired behavior unresolved.

**Can the same-object association be tested more directly across training stages?** The large stage-5,400 to stage-7,200 change in swap success and its partial reversal during consolidation suggest a dynamic phenomenon, but the original intervention set does not by itself fully control target multiplicities or initial prediction bias.

**Does a Shape readout fail because it discards geometry, because it cannot localize the relevant object, or because it has insufficient expressive capacity?** The original 48D mean versus 1,200D spatial comparison changed several things at once. The spatial probe's high random-encoder performance makes a more precise readout comparison essential.

**Is the shared eight-input task head responsible for the task-dependent contradictions?** Operator-dependent inputs and one shared MLP are plausible sources of interference, but observational contradictions cannot isolate that cause. Freezing the earlier pathway and fitting alternative readouts would test a narrower version of the hypothesis.

These were the reasons for the four subsequent diagnostic follow-ups: count-controlled curves, count-preserving Binding swaps, spatial Shape/readout controls, and task-head comparisons. Their outcomes belong in the separate follow-up findings report. The present report documents **the evidence that made those questions necessary**.

## 12. Evidence boundaries

The analysis rests on five training seeds, two curricula and three checkpoints, with repeated evaluations of the same development scenes. Paired edits and object-level crops must be grouped by their originating scene or run when estimating uncertainty. Most intervals are descriptive, unadjusted for repeated exploratory comparisons.

The checkpoint-reproduction audit recorded **71 disagreements among 42,720** archived-versus-reproduced predictions, associated with seed 46; the maximum observed probability discrepancy was about **0.116**. This is small as a fraction of all predictions, but it matters for near-zero differences and exact run equivalence. The reproduced diagnostic checkpoints must not be silently treated as bit-identical archived weights.

The analysis also **repeatedly inspected development data**. The original test splits were not a sealed independent validation of the many hypotheses developed from these diagnostics. Some diagnostic transformations introduce distributions unlike both original ID and OOD questions. Claims of object understanding or compositional generalization must therefore remain conditional on the particular measurement.

The executable evidence is in `notebooks/medium_world/representation_checkpoints.ipynb`, `representation_probing.ipynb`, `representation_probing_analysis.ipynb`, `representation_probe_refinement.ipynb`, and `representation_evidence_analysis.ipynb`, with the frozen model and diagnostic functions under `src/medium_visual_world/`. The report refers to the completed notebook outputs; it does not imply that any checkpoint was retrained for this document.

**The lasting result:** the experiments uncovered a mismatch between *visual information potentially available*, *the evidence a diagnostic readout can extract*, and *the answers the end-to-end model actually gives*. Resolving that mismatch requires controlled behavioral and readout tests—not another unqualified comparison of curriculum endpoint accuracies.
