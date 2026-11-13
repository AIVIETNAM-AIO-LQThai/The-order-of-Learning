# Diagnostic Follow-Up: What the Four Investigations Reveal

*12 November 2026*

The latest investigation was prompted by a weakness in our earlier curriculum comparisons: **a change in Binding accuracy does not, by itself, tell us what the model has learned**. A model can improve by exploiting incidental properties of scenes, shifting its Yes/No threshold, making better use of existing features, or learning to associate attributes on the same object. These mechanisms can produce similar aggregate scores but have very different implications for the learning-order hypothesis.

The four notebooks examine that ambiguity from complementary directions. The first removes a known count-based cue from historical predictions. The second changes the association between attributes while preserving their counts. The third asks whether Shape can be decoded from spatial visual features when the original prediction pathway struggles. The fourth tests whether the shared final classifier itself explains the remaining failures. Taken together, they narrow what we can claim about acquisition, retention, representations, and task-conditioned evidence use.

The important distinction is between **information present in an image**, **information recoverable from a frozen feature tensor by a newly trained decoder**, and **information the original model actually uses to answer a question**. These are not interchangeable measurements.

## 1. Binding improves beyond the tested counting shortcuts

### Why the ordinary Binding benchmark was insufficient

For a query asking whether an object has Color $c$ and Shape $s$, the intended rule is

$$
B(c,s)=\mathbb{1}\!\left[\exists o:\operatorname{color}(o)=c\land\operatorname{shape}(o)=s\right].
$$

Our regular Binding negatives already require the requested Color and Shape to occur somewhere in the scene. That prevents a trivial *both attributes exist* rule, but it does **not** prevent reliance on how often either attribute occurs. If requested Colors or Shapes appear more frequently in positive scenes, the model can partly predict the label without establishing that the two attributes belong to the same object.

Notebook [`01_count_free_curves.ipynb`](../notebooks/medium_world/01_count_free_curves.ipynb) therefore re-examines saved predictions only on questions for which **exactly one object has the requested Color and exactly one object has the requested Shape**. In a positive scene those attributes belong to the same object; in a negative scene they belong to different objects. Their marginal counts are identical in either case.

The resulting ID-development subset has **109 questions: 34 Yes and 75 No**. An always-No classifier would obtain **68.81% raw accuracy** but **50% balanced accuracy**. The appropriate primary score is consequently

$$
\operatorname{BA}=\frac{1}{2}\left(\operatorname{TPR}+\operatorname{TNR}\right).
$$

This is a **count-controlled** test, not a completely shortcut-free benchmark. Total object number, distractors, queried attribute identity, spatial arrangement, object geometry, and other scene properties may still differ between the labels.

### What changes during training

The notebook reads **145 historical prediction traces**, without retraining or changing the model, and finds all expected source traces. The clearest within-curriculum trajectory appears under Constant LR in `prereq_mixed_pair_focus`:

| Checkpoint | Training stage just completed | Count-controlled ID Binding BA |
|---|---|---:|
| 5,400 | Color, Shape, and Conjunction; before focused Binding | **48.22%** |
| 7,200 | Binding-focused training | **69.26%** |
| 9,000 | Final Mixed consolidation | **63.62%** |

The 5,400-to-7,200 increase is **+21.04 percentage points**, with a five-seed paired 95% interval of **[+7.89, +34.18]**. At 7,200, this curriculum exceeds the Constant-LR Mixed model by **+10.39 points [2.88, 17.91]** on the same count-controlled questions. By 9,000, the advantage is only **+1.28 points**, with an interval crossing zero.

The stage-dependent pattern matters more than the isolated final score. Before Binding-specific supervision, the prerequisite-trained model does not reliably distinguish the controlled positives and negatives. During focused Binding training, it develops discrimination that cannot be attributed to the **number of requested-Color or requested-Shape objects**. The later Mixed phase reduces part of that discrimination, even though it again trains all four tasks.

This establishes an effect on **observable decision behavior**. It does not establish that an internal representation was created at 7,200 and erased by 9,000. Decision boundaries, score calibration, query-conditioned weighting, and visual features can all change during those phases.

The final BA also conceals uneven class-specific behavior. At step 9,000 for Constant-LR Prereq-Mixed, mean sensitivity on the controlled subset is about **74.7%**, whereas specificity is about **52.5%**. The model is better at recognizing positives than rejecting negatives; partial discrimination is not the same as a dependable Binding rule.

### Does the effect survive other scene-level controls?

Additional analysis of the **same saved predictions** shows that Yes and No scenes still differ in total object number. A simple rule such as “predict Yes for at least four objects” reaches about **59.6% balanced accuracy** on the selected ID subset. That is an important warning against treating the first count control as conclusive.

Two further observations strengthen—but do not complete—the evidence. Within the 3-, 4-, and 5-object strata, Constant-LR models retain count-controlled discrimination. Their **object-count-stratified AUC** is approximately **67.7–70.2%**. Separately fitted metadata-only baselines that use global counts, queried attribute identities, distinct-attribute counts, and basic geometry summaries—but not the actual association between the queried attributes—reach roughly **50–59% stratified AUC**.

The reasonable conclusion is thus narrower and stronger than either extreme: **the Constant-LR models are responding to something beyond the tested counting and coarse metadata heuristics**, but these observational analyses cannot exclude all local-image or scene-selection cues. Notebook 02 addresses this limitation more directly.

### Learning-rate history changes whether this discrimination is visible

A second important result in Notebook 01 is that the count-controlled score is much more sensitive to LR policy than ordinary endpoint comparisons suggested.

| Learning-rate policy | Step-9,000 count-controlled ID BA across curricula |
|---|---:|
| **Constant LR** | **59.7–63.6%** |
| Global Cosine | 50.9–52.7% |
| Reset Cosine | 51.2–54.4% |
| Global Plateau | 45.3–54.0% |
| Reset Plateau | 53.1–58.0% |

The tested global cosine schedule decays toward `1e-6`. Its weak count-controlled performance does not mean its remaining Binding accuracy is “mostly counting.” Follow-up threshold-free analyses find a weak discrimination signal under some decayed policies, and the models do not closely reproduce a fitted count-only rule. It is more accurate to say that **the tested decays yield much weaker evidence of Binding discrimination independent of requested-attribute multiplicity**.

Reanalysis of the saved traces also finds a strong association between final count-controlled AUC and the **nominal LR-weighted exposure to Binding examples**, particularly exposure delivered later in training. The logged quantity is

$$
E_{B}=\sum_{u}\eta_u\,n_{B,u},
$$

where $\eta_u$ is the learning rate at update $u$ and $n_{B,u}$ is the number of Binding examples in that batch. Across the 20 curriculum-by-policy conditions, Spearman correlation with final count-controlled AUC is about **$0.89$ for total exposure**, **$0.88$ for exposure after update 5,400**, and **$0.29$ for earlier exposure**.

This is an informative **association**, not an identified learning mechanism. The logged measure weights example counts by nominal LR; it does not measure effective gradients or parameter changes. Later exposure coincides with changes in model age, task history, optimizer state, and often the task mixture. Likewise, a retrospectively defined “acquisition onset” between roughly 5,600 and 6,400 updates only describes **runs that eventually cross the chosen sustained-AUC threshold**. Many runs do not. These data do not establish a universal point at which prerequisites have formed.

The implication for the original hypothesis is important: a curriculum that introduces Binding late must still provide enough **usable optimization opportunity** for that task. A comparison that controls presentation counts but not the effective optimization history cannot isolate the effect of prerequisite order.

## 2. Binding swaps provide more direct evidence of association sensitivity

### What the intervention actually controls

Notebook [`02_binding_swaps.ipynb`](../notebooks/medium_world/02_binding_swaps.ipynb) asks a different question: **what happens if the scene keeps the same attributes, but those attributes belong to different objects?**

It swaps the Colors of two objects, leaving the number of objects, the entire Color and Shape multisets, their positions, sizes, and rotations unchanged. The oracle Binding label changes. A prediction rule based only on the requested attributes' independent presence or multiplicities cannot distinguish the original and edited images.

The notebook evaluates **30 frozen checkpoints**: Mixed and Prereq-Mixed→Pair, five seeds each, at updates 5,400, 7,200, and 9,000. Each checkpoint uses **109 eligible original questions**, with **75 original No** and **34 original Yes** labels.

Two outcomes answer different questions:

- **Correct probability direction:** does $P(B=\mathrm{Yes})$ move upward for a No→Yes edit and downward for a Yes→No edit?
- **Both endpoints correct:** does the original trained head classify *both* images correctly, at the existing decision threshold?

A model can react in the right direction without moving far enough to answer either endpoint reliably. Conversely, a large change in both-correct may partly reflect a shift in a biased decision threshold rather than wholly new association evidence.

### Results across the learning stages

| Curriculum | Update | Correct probability direction | Both endpoints correct |
|---|---:|---:|---:|
| Mixed | 5,400 | 58.35% | 22.57% |
| Mixed | 7,200 | 62.02% | 26.06% |
| Mixed | 9,000 | 68.99% | 28.26% |
| **Prereq-Mixed→Pair** | **5,400** | **53.03%** | **4.40%** |
| **Prereq-Mixed→Pair** | **7,200** | **71.92%** | **39.45%** |
| **Prereq-Mixed→Pair** | **9,000** | **70.64%** | **32.66%** |

During the focused Binding phase, Prereq-Mixed's strict both-correct rate rises by approximately **35.05 points**. That headline requires careful interpretation. At update 5,400, four of five Prereq seeds answer Yes to nearly every eligible original question. Since the edited counterpart has the opposite answer, a predictor that says Yes to almost everything will almost never be correct at **both** endpoints. The extremely low starting strict score is partly a **floor induced by response bias**.

The correct-direction metric is less vulnerable to this particular floor. Its rise from **53.03% to 71.92%** suggests that focused Binding supervision improves how the model's score responds to a changed association, not only how often its hard decision crosses 0.5.

Nevertheless, even at the best reported stage, **39.45% both-correct** is inadequate for claiming reliable Binding. It means the model often fails one of the two logically linked cases. The label-balanced both-correct score at update 7,200 is about **38.0%**, so the unequal 75/34 starting-label support does not create an artificially favorable result.

From 7,200 to 9,000, the Prereq model's both-correct rate decreases by approximately **6.79 points**; correct-direction performance changes little. This divergence matters. **Some probability-level sensitivity to the manipulated association persists while strict decision correctness deteriorates.** A change in decision threshold or class bias could explain part of the decline; representation loss is only one possible mechanism.

### Does creating an unseen combination invalidate the swaps?

Some swaps produce an object with a held-out Color–Shape pair. A follow-up split of these same swap outputs shows that the large 5,400→7,200 improvement survives in both the held-out-created and non-held-out groups. That rules out the simple explanation that the entire gain comes from edits that accidentally construct unseen combinations.

It does **not** show those two groups have equal difficulty. The target label direction, the selected swap partner, and the distribution of affected shapes can still differ. The important result is the persistence of the association-sensitive learning effect under both conditions.

**Interpretation.** Notebook 01 shows that the effect cannot be fully explained by the requested-attribute counts. Notebook 02 adds a paired intervention that changes **which attributes co-occur locally** while preserving those counts. Their agreement is the strongest evidence from this investigation that the model acquires **partial functional Binding sensitivity** during training.

This is not proof of symbolic object slots, abstract relational reasoning, or generalization to every new combination. A local visual co-occurrence detector could pass some swaps without creating explicit object identities. That would still constitute useful *functional* association sensitivity, but a narrower capability than the one imagined in the broad curriculum hypothesis.

## 3. Shape is recoverable from spatial features, yet the original task pathway uses it unreliably

### Representation availability and task performance are different questions

The weak Shape and Conjunction scores prompted a possible explanation: perhaps the visual encoder simply fails to preserve useful geometric information. Notebook [`03_shape_and_readout_probes.ipynb`](../notebooks/medium_world/03_shape_and_readout_probes.ipynb) tests this proposition by freezing the original network and fitting independent Shape decoders.

The distinction in supervision is critical. An **object-level** diagnostic receives an oracle object center and a local 5×5 grid of 48-dimensional visual features; it predicts one of four Shape classes. The deployed model instead receives an entire scene and a Shape-existence query, without being told where the target object is. Success on the first task does not automatically transfer to the second.

### Why spatially ordered features change the result

At the final checkpoints, the object-level probe comparisons are approximately:

| Object-level decoder | Random encoder | Trained encoder |
|---|---:|---:|
| **Ordered 5×5 feature grid; 1,200D linear decoder** | **87.9%** | **86.2–86.3%** |
| Same 1,200D inputs, **25 feature vectors shuffled independently per object** | 49.0% | 48.5–49.8% |
| **Mean-pooled 48D** representation | 51.1% | 51.9–52.9% |
| **Per-position class scores → max/mean → small MLP** | 65.2% | 69.0–70.3% |

The comparison between the first two rows is particularly informative. Both decoders receive **the same number of scalar features**. Shuffling the 25 local 48D vectors independently for each object disrupts their stable relative locations while preserving the individual vectors. The large accuracy loss shows that **spatial indexing matters for this supervised decoder**; the gain is not explained by merely supplying 1,200 rather than 48 input numbers.

The random encoder is equally important. An untrained patch projection can preserve substantial information from the original pixels; the subsequently trained linear probe can learn geometry from this information. Therefore the 86–88% score does **not** demonstrate that end-to-end curriculum training created a Shape abstraction. Indeed, for this particular spatial decoder, the random encoder is slightly better on average than the trained encoder.

At the same time, the head-like max/mean decoder shows a modest trained-over-random advantage of roughly **2.7–5.3 points** across the six curriculum/checkpoint conditions. It would also be wrong to conclude that training never changes Shape-relevant features. **What the probe can extract depends strongly on its readout form.**

The most defensible observation is that **geometric information exists in the spatially organized feature tensor and is accessible under privileged supervision**. Whether the original model's own Shape pathway can find and use it is a separate question.

### What happens on the original scene-level Shape task

The notebook also compares three decoders on the **same binary Shape-existence questions**, without oracle object locations: the original trained head, a new classifier fitted to its exact original eight scalar inputs, and a new spatial-convolutional readout fitted to the frozen visual features.

| Curriculum | Update | Original Shape head | Re-fitted eight-feature head | Spatial-convolutional readout |
|---|---:|---:|---:|---:|
| Mixed | 5,400 | 63.8% | 64.1% | 69.3% |
| Mixed | 7,200 | 65.1% | 66.2% | 68.8% |
| Mixed | 9,000 | 67.8% | 67.0% | 68.5% |
| **Prereq-Mixed→Pair** | **5,400** | **64.4%** | **65.1%** | **66.8%** |
| **Prereq-Mixed→Pair** | **7,200** | **57.1%** | **61.2%** | **66.1%** |
| **Prereq-Mixed→Pair** | **9,000** | **66.9%** | **67.7%** | **68.2%** |

The most informative checkpoint is **Prereq-Mixed at update 7,200**, immediately after focused Binding training. The original head has fallen from **64.4% to 57.1%** Shape accuracy, whereas the newly fitted spatial readout remains at **66.1%**. The difference between spatial readout and original head is **+8.98 points**, with a paired five-seed interval of **[+1.22, +16.75]**. After final Mixed consolidation, the original head recovers to **66.9%** and the spatial readout reaches **68.2%**; their gap is then only about **1.33 points**, with uncertainty crossing zero.

This is compatible with an account in which Binding specialization changes how the original model **uses** Shape evidence, while geometry remains recoverable by a newly supervised decoder. It helps explain why a curriculum can improve Binding while damaging Shape-task answers without necessarily removing all Shape information from the visual tensor.

But that mechanism is **not causally isolated**. The encoder itself changes across checkpoints; the new decoder receives extra supervised fitting and has a different capacity and information path. A larger readout might recover Shape anew from preserved low-level pixel information, even if the trained model's own Shape representation has deteriorated.

The `spatial_conv_probe` is also not a full layout-preserving object-level reader. It uses a nonlinear **3×3 convolution** to combine neighboring tokens, then applies global max/mean pooling. Thus the scene-level results cannot isolate **pooling alone** from object localization, limited receptive field, query scoring, or optimization. The successful ordered 5×5 object probe is *not* a matched-capacity, same-task ablation of that scene-level pathway.

### The geometry of the errors matters

A further analysis of the same object-prediction outputs suggests that some Shape confusions track **rendered object footprint**, especially among circles, squares, and triangles at different radii. Across 12 convex Shape×radius conditions, footprint similarity has a descriptive correlation of about **+0.65 for trained** and **+0.73 for random** layout-aware probes. Crescents depart from the trend, presumably because concavity supplies additional information. Variation across rotation angles is smaller than variation across radii in the summarized cells.

This is useful diagnostic context, but it does not prove that the decoder classifies objects by area. Contours, antialiasing, radius, position, and pixel occupancy covary in the rendered images. It reinforces a general point: **an externally trained probe can recognize Shape using cues that are available in the synthetic rendering, without revealing which cues the original task head learned to use**.

**Interpretation.** Poor end-to-end Shape accuracy cannot be equated with total absence of Shape information. But a high privileged-probe score cannot be equated with a learned, accessible, task-reusable Shape skill either. The investigation reveals an **information-access gap** and leaves its precise source unresolved.

## 4. Replacing the shared head does not repair the critical reasoning failure

### What hypothesis is being tested

The original model constructs eight query-conditioned scalars—maxima and means of two attribute-score maps, one joint-query map, and a local interaction map—and passes them to a shared `8 → 32 → 2` classifier. One possibility was that Color, Shape, Conjunction, and Binding compete through this final shared MLP.

Notebook [`04_task_head_interference.ipynb`](../notebooks/medium_world/04_task_head_interference.ipynb) freezes **everything through those eight original features** and fits alternative heads. The alternatives include a new shared head without explicit task ID, task-aware shared heads, a wider task-aware head approximately matched in total parameter count to four separately routed task heads, and the separate heads themselves.

This isolates a more precise question than an end-to-end architecture redesign: **given the same frozen evidence, does changing how the final classifier is fitted or task-routed resolve the observed failures?** It does not independently identify whether the encoder or score-pooling stage generated adequate evidence in the first place.

### Aggregate scores improve modestly, but the decisive subgroup does not

At update 9,000, averaged across two curricula and five seeds:

| Task | Original shared head | Four separate heads | Capacity-matched task-aware head |
|---|---:|---:|---:|
| Color | 99.3% | 99.6% | 99.6% |
| Shape | 67.3% | 67.9% | 67.9% |
| Conjunction | 80.2% | 80.5% | 80.9% |
| Binding | 62.5% | 65.8% | 66.0% |

There is no compelling aggregate advantage of the separate heads over the capacity-matched task-aware shared head. The alternative heads also receive **40 additional epochs of supervised fitting**, whereas the original pretrained head does not. Their improvement cannot therefore be attributed solely to task separation.

More importantly, ordinary Conjunction accuracy is not a convincing measure of the intended logical operation. In the standard ID-development set, **128 positive Conjunction questions** and **76 negatives where Color is absent** can all be answered correctly by the rule “Yes if the requested Color exists.” That shortcut scores

$$
\frac{128+76}{256}=79.69\%.
$$

The critical cases are the remaining **52 Conjunction negatives where Color is present but Shape is absent**. A Color-only predictor rejects **none** of them. The evaluated heads reject only:

| Final-checkpoint decoder | Correct rejection on the 52 critical negatives |
|---|---:|
| Original trained shared head | **25.0%** |
| Separate task heads | **34.2%** |
| Capacity-matched task-aware head | **34.2%** |

The original model is not literally implementing *only* Color presence—it succeeds on roughly a quarter of these cases—but it is **strongly Color-dominant**. The replacement heads remain far from a reliable logical AND. At update 7,200, some newly fitted heads perform **worse** on these critical negatives while raising overall Conjunction accuracy. A better total score can therefore reflect **greater exploitation of the benchmark's label structure rather than better reasoning**.

On the 109 count-controlled Binding questions, head replacement produces small gains that are not reliable across the five seeds. The independently fitted heads do not suddenly create a strong association capability from the frozen eight features. This is another reason to avoid treating the modest ordinary Binding improvements as proof that separating the heads resolves task interference.

**Interpretation.** The experiment does **not** establish shared-head gradient competition as the primary problem. It shows that simply learning a different map from the **same eight compressed inputs** is insufficient to repair the most revealing failures. A limitation could remain in the upstream visual representation, operator-conditioned query scoring, spatial aggregation, available supervision, or the distribution of training examples. Head sharing is not exonerated under every possible end-to-end protocol; it is simply **not identified as the bottleneck by this controlled frozen-feature comparison**.

## 5. What the four findings mean together

### Learning a relation is not the same as obtaining a good Binding score

Notebook 01 shows that discrimination grows during focused Binding training even after fixing the requested Color and Shape counts. Notebook 02 adds a stronger test: when those counts and the rest of the attribute multiset stay fixed, the model increasingly adjusts its answer in the direction required by a change in **which object carries which Color**.

These observations support **partial functional sensitivity to attribute association**. The qualification *partial* is essential: even after successful specialization, strict correctness on both ends of the swap remains below 40%. The model has learned something useful about association, but not a robust rule that can be assumed to transfer across new scenes or combinations.

The two notebooks also establish a difference between **acquisition and retention**. A concentrated Binding phase can improve association-sensitive behavior beyond the contemporaneous Mixed model. Later Mixed consolidation restores other task scores while reducing some of that Binding behavior. The effect on hard decisions is larger than the decline in correct probability direction, suggesting that readout adjustment or bias may contribute alongside any feature changes. We therefore observe **behavioral forgetting**, not proven representational erasure.

### A prerequisite can be present in features but unavailable to the dependent task

The curriculum hypothesis assumes that learning Color and Shape creates capabilities that Binding can reuse. Notebook 03 shows why that assumption is stronger than it first appears. An independently trained decoder, given a specific object's location and ordered local features, can recover Shape very well—even from a random encoder—while the original scene-level Shape head performs poorly.

A prerequisite must therefore be evaluated not only by whether its signal is **decodable somewhere in the network**, but also by whether the model's **own decision pathway** can access and use that signal under the relevant question. The same visual tensor can support a successful supervised probe and an unsuccessful task answer.

Notebook 04 extends this insight. If the final eight scalar features and their training distribution already favor shortcuts or omit useful spatial organization, separating the last MLP cannot recreate the missing relation or reliably impose logical AND. This explains why an apparent component-skill score does not automatically certify reusable competence.

### The architecture remains a possible constraint, not an established single cause

The combined evidence points toward the **entire information path from spatial features through query-conditioned evidence selection to the decision**. It does not identify a single failed operation.

The successful object-level spatial probe has oracle location and a different prediction task. The scene-level convolution mixes nearby patches but still performs global pooling. The retrained eight-feature heads have extra supervision and different routing. None is a controlled experiment that changes *only* spatial pooling while keeping object localization, supervision, readout capacity, and task conditions fixed.

Therefore statements such as “the Transformer did not learn Shape,” “max/mean pooling is the bottleneck,” or “the shared head caused catastrophic forgetting” are stronger than these four notebooks support.

### Learning-rate policy changes the interpretation of curriculum effects

Notebook 01 also reframes what a training schedule comparison can mean. Under aggressive LR decay, the model frequently shows little count-controlled Binding discrimination even after receiving the prescribed examples. With Constant LR, some controlled association sensitivity becomes measurable. Across saved traces, the correlation with late nominal Binding exposure is strong, but not causal.

The implication is **not** that a particular LR policy guarantees compositional learning. Rather, an apparent failure of prerequisite-first learning might reflect limited opportunity to adapt when Binding is introduced; an apparent success might reflect better task-specific optimization or a favorable decision boundary instead of superior prerequisite representations. The current diagnostics cannot disentangle those causes.

## 6. What remains unresolved after these investigations

**Whether the model has learned an object-level abstraction.** The swaps establish local association sensitivity but not object identity, systematic compositionality, or transfer to genuinely novel combinations. A detector of local co-occurrence could produce some of the same behavior.

**Which stage of the Shape pathway is limiting scene-level performance.** Shape geometry is recoverable from ordered local features, but the successful decoder is privileged and solves a different task. The current comparisons cannot separate localization, spatial aggregation, query scoring, supervision, and optimization.

**Why the Binding-focused gain partly disappears.** The change in strict swap correctness could result from representation modifications, task-specific evidence weighting, calibration, or their interaction. Score-direction retention makes pure “knowledge erased” explanations particularly premature.

**Whether the tested OOD score measures transfer.** The count-controlled OOD subset has **154 questions (46 Yes, 108 No)**, and Constant-LR balanced accuracy is near chance. However, the saved OOD generator also changes the queried Color–Shape combination and permits a scene-level label shortcut: **the presence of any held-out object predicts the OOD label**. These diagnostics do not establish either successful or failed genuine compositional transfer.

**How stable small comparisons are.** The same development scenes were reused across methods and stages. Intervals are exploratory, generally unadjusted paired intervals over **five training seeds**, not confidence intervals over hundreds of independent questions. Some nominally identical historical runs also diverge numerically. Large, repeated effects and converging controls deserve greater weight than one- or two-point endpoint differences.

## Conclusion

These four investigations substantially change the interpretation of our previous learning curves. **The model does acquire some ability to respond to same-object Color–Shape association beyond the tested count shortcuts**, and focused Binding training strengthens that behavior. But this ability remains incomplete and is not fully preserved during subsequent multitask consolidation.

At the same time, the Shape investigations show that geometric information can survive in spatial visual features without being reliably used by the model's own question-answering pathway. Replacing the final shared classifier does not resolve the hardest Shape-dependent Conjunction errors. The obstacle is therefore **not established to be an isolated final-head problem**, nor has spatial pooling or encoder representation learning been uniquely identified as its cause.

The scientific outcome of this stage is more precise than a claim that a curriculum won or lost: **we can now distinguish partial acquisition of association-sensitive behavior, its instability across training phases, and the gap between decodable visual information and task-effective information use**. Those distinctions are necessary before a future experiment can fairly test whether learning prerequisites in a particular order produces a genuinely more reusable visual capability.
