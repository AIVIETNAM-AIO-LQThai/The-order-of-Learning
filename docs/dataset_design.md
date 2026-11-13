# Dataset design: separating visual attributes, relationships and learning history

The dataset is an instrument for testing learning-order hypotheses, not merely a collection of balanced images. If a label can be predicted without the capability it was intended to measure, then improvement on that label does not establish improvement in the capability. That distinction became essential in interpreting Conjunction, Binding and the held-out-combination evaluations.

## Research question translated into data requirements

The original idea was that a model might benefit from mastering elementary attributes before evaluating relations and compositions. Three competing explanations guided the design:

1. **Reusable prerequisites:** preceding visual concepts make later relational learning more effective, at matched task exposure.
2. **Specialization and interference:** emphasizing a later task changes which earlier tasks the model can answer correctly, potentially creating an acquisition–retention trade-off.
3. **No lasting ordering advantage:** when total experience and final mixed training are matched, apparent advantages may disappear or reflect optimization timing rather than conceptual reuse.

To compare these explanations, the dataset must supply well-defined oracle answers, permit equalized question exposure across training schedules, and expose shortcuts that could imitate the intended reasoning. **The actual experiments fix the dataset when comparing curricula**; later diagnostic questions may re-query or intervene on scenes, but those are not silently substituted for the original benchmark.

Two generated worlds were used. They address related but different scientific questions.

## 1. Small World: grounding attributes, relations and Boolean composition

The first world contains **exactly three non-overlapping objects** in a 64 × 64 image. It has four colors (`red`, `blue`, `green`, `yellow`), three shapes (`circle`, `square`, `triangle`), two size classes and constrained positions. A constructive scene pattern creates an anchor object and two distractors: one shares its color and the other shares its shape. This makes a color–shape pair necessary to refer unambiguously to the anchor.

The implemented question families are:

- `EXISTS_COLOR(c)`, `EXISTS_SHAPE(s)` and `EXISTS_PAIR(c,s)`.
- `LEFT(REF(...), REF(...))` and `ABOVE(REF(...), REF(...))`, whose referenced objects exist.
- `XOR(LEFT(...), ABOVE(...))`, which combines two separate spatial truth values.
- `CLOSER(...)`, present in a **separate adaptation dataset** rather than the main curriculum questions.

The `XOR` data balance all four underlying truth combinations, so knowing only one constituent relation cannot by itself solve the composed task. Hard `EXISTS_PAIR` negatives contain the requested Color and Shape on different objects.

The original generated splits include **1,800 training questions**, **360 ID-dev**, **360 OOD-dev** and **128 adaptation-training** questions, with one distinct scene per question in the inspected splits. The OOD pair holdouts are `red–triangle`, `blue–square` and `green–circle`. The OOD distribution also changes other attributes: for instance, yellow is present in **75.5%** of training scenes but only **35.8%** of OOD-dev scenes. A performance change cannot be attributed solely to the three withheld combinations.

The initial modeling difficulties on this world and the limited nine-seed replication are part of why the later Medium World experiments use a more focused task hierarchy. **Small World's XOR and CLOSER results should not be interpreted as though they were tested by the Medium World Binding model.**

[Small World EDA](small_world_eda_findings.md) · [Small World nine-seed comparison](nine_seeds_comp_small_world.md)

## 2. Medium World: independent conjunction versus same-object Binding

Medium World uses 64 × 64 RGB scenes with **3–5 non-overlapping objects**, five colors and four shapes:

- Colors: `red`, `blue`, `green`, `yellow`, `purple`.
- Shapes: `circle`, `square`, `triangle`, `crescent`.
- Radii: 5–8 pixels; permitted rotations depend on shape. The renderer applies a common appearance pipeline.

The four binary question types have different logical demands. Let $O(I)$ be the objects in image $I$:

$$
C(I,c)=\mathbb{1}\{\exists o\in O(I):\operatorname{color}(o)=c\}
$$

$$
S(I,s)=\mathbb{1}\{\exists o\in O(I):\operatorname{shape}(o)=s\}
$$

$$
J(I,c,s)=C(I,c)\land S(I,s)
$$

$$
B(I,c,s)=\mathbb{1}\{\exists o\in O(I):\operatorname{color}(o)=c\land\operatorname{shape}(o)=s\}.
$$

Here $J$ is **Conjunction**, an independent existence test; $B$ is **Binding**, a same-object test. A positive Binding implies positive Conjunction, but a positive Conjunction need not imply Binding. Neither the scene graph nor the objects' coordinates are passed to the ViT; the model sees rendered pixels and the symbolic query tokens.

### Split sizes and what is shared

| Split | Scenes | Questions | Question families |
|---|---:|---:|---|
| Train | 5,000 | 10,000 | 2,500 each of Color, Shape, Conjunction, Binding |
| ID-dev | 512 | 1,024 | 256 each |
| ID-test | 512 | 1,024 | Same four-task structure |
| OOD-dev | 400 | 400 | Binding only, held-out target pairs |
| OOD-test | 400 | 400 | Binding only, held-out target pairs |

In Train and ID, **two questions share each scene**. OOD records contain **one question per scene**. Scene IDs are disjoint between splits; related questions within one scene are *not independent images*. Question labels are balanced Yes/No within the generated task and split.

The four held-out combinations are:

```text
blue–square, green–circle, purple–triangle, red–crescent
```

They do not occur as rendered pairs or queried pair targets in Train and ID. That isolation is important, but **it is not sufficient to make the OOD evaluation a clean compositional test**.

## 3. Sampling choices change which shortcuts are possible

### Conjunction: a positive label is genuine, but negatives are asymmetric

The generator forms Conjunction negatives only when **exactly one** of the requested attributes is present. As a result, on the training questions:

| Conditional on the correct answer being No | Fraction of negatives |
|---|---:|
| Queried Color absent, Shape present | 57.28% |
| Queried Color present, Shape absent | 42.72% |

Thus a rule that predicts Conjunction from **Color presence alone** gets approximately **78.6%** training accuracy and **79.7%** ID-dev accuracy, despite not checking Shape. A Shape-only rule reaches approximately **71.4%** training accuracy. These are *oracle-metadata heuristics*, not evidence that the trained model explicitly implemented either rule.

There is a second subtlety: in **60.72%** of positive training Conjunction questions, the exact queried Color–Shape pair also appears on one object. The remaining positives can have the attributes on different objects. Consequently, even success on Conjunction positives does not by itself reveal whether the model learned independent logical AND or responded to a more local correlation.

The intended measurement of Conjunction should include the hard case **Color present / Shape absent**. Aggregate accuracy around 80% is not evidence of mastery if performance on that case remains poor.

### Binding: hard negatives remove simple attribute-existence rules, not every shortcut

For a negative Binding question, the queried Color and Shape are both present but **never on the same object**. A classifier that merely checks whether the requested Color exists, whether the requested Shape exists, or whether both exist independently is therefore at **50%** on the balanced original Binding task.

That is useful but incomplete. The number of matching-color objects, matching-shape objects, total object count, object positions, distractor composition and rendered geometry can still correlate with the label. For example, a later diagnostic subset containing **exactly one queried-Color object and one queried-Shape object** has 109 ID-dev Binding questions, of which only 34 are Yes. Its label cannot be *logically determined* from those two attribute counts, but total object count remains predictive: a simple `n_objects >= 4` heuristic attains about **59.6% balanced accuracy** on that subset.

The logical ability to solve a scene and the statistical ability to exploit a sampling bias are therefore distinct. Counts must be controlled or stratified, and comparisons should include **counterfactual color swaps** that hold color/shape multisets fixed while changing which attributes are paired.

### Train/ID attribute frequencies are not uniform over objects

The four withheld combinations remove one allowed pair for red, blue, green and purple, but none for yellow. Yellow therefore has more allowed pairings and occurs more frequently: **5,009** of **19,905** training object occurrences are yellow, versus approximately 3,700–3,739 for each other color. Shape object frequencies are closer to uniform.

The target-selection code attempts to balance queried attributes and labels. This does not eliminate correlations between the requested query, scene composition and visual evidence. Uniform question targets are not equivalent to uniformly sampled scenes.

## 4. The OOD benchmark contains a label-construction confound

The original Medium World OOD generator samples:

- **Yes scenes** using all 20 possible Color–Shape combinations, and chooses a held-out pair that is actually present.
- **No scenes** using only the 16 training-allowed combinations, and chooses a held-out target whose Color and Shape occur separately but not together.

This creates a scene-level predicate:

$$
H(I)=\mathbb{1}\{\text{at least one held-out Color–Shape object appears in }I\}.
$$

For all **400 OOD-dev questions**, $H(I)$ equals the correct label (200 Yes and 200 No). This is a **perfect metadata-level shortcut** that does not require answering whether the *specific queried pair* exists. It need not be easily recognizable from the pixels, and its existence does **not** establish that a trained model uses it. It does establish that the benchmark cannot isolate systematic generalization to the target held-out composition.

OOD also changes scene composition: average total objects are **4.19** versus **3.98** in Train; average distinct colors are **3.06** versus **2.89**, and distinct shapes **2.87** versus **2.67**. OOD contains only Binding questions, unlike ID-dev. Its score is consequently a mixture of unfamiliar combinations, scene-distribution changes, recognition difficulty and the label-generation rule.

**A defensible conclusion is that the current OOD benchmark measures performance on a particular shifted distribution; genuine held-out-composition transfer remains unresolved.** All earlier OOD accuracy values should be read with that qualification.

A clean compositional test would need to **match Yes and No scenes for the presence of unrelated held-out pairs**, balance relevant scene composition and keep the queried target as the deciding factor. This describes the identification requirement; it does not retroactively change the frozen datasets used by the completed studies.

## 5. What is actually controlled when curricula are compared?

The current Medium World curriculum studies can fix the rendered dataset, query/label oracle, model architecture, initialization seed, total optimizer updates and total presentations of every original training question. Their treatment is the **order and timing of those presentations** (and, in explicit LR studies, the learning-rate policy).

This creates three distinct causal targets:

- **Full-policy comparison:** What happens if we use curriculum A rather than B, with all of their actual differences in stage timing and task mixture?
- **Matched-order comparison:** What does an extra sequencing decision contribute when Binding introduction and later Binding exposure are held fixed?
- **Mechanism diagnosis:** Does a behavior depend on same-object association, scene counts, local Shape geometry, final-score aggregation or optimization state?

Those are not interchangeable. For example, a shared final training block supplies the same *planned examples*, not identical model and optimizer states at its start. Matching total training questions does not make the gradients or their effective learning rates identical. A genuine continuation fork can start from a common serialized model, optimizer and RNG state, but its suffix still changes learning dynamics.

## 6. Consequences for interpreting the project

The design successfully makes **independent attribute presence insufficient** to solve original Binding hard negatives. That supports a study of more difficult visual association, but does not prove models have acquired object slots or robust local binding.

It does **not** make Conjunction a strong test of both attribute detectors; the negative sampler gives Color an asymmetric advantage. Nor does the current OOD sampler isolate transfer to unseen pair targets. These are substantive facts about *what the data measure*, not cosmetic dataset defects.

The remaining informative question is whether curriculum interventions affect **association-sensitive behavior under stronger controls**, independently of easy label or scene statistics. This is why later work examines count-controlled Binding, attribute-preserving swaps, and frozen feature readouts instead of relying only on benchmark accuracy.

## Related research and what we do not infer from it

Controlled symbolic questions and rendered scenes connect this project to [CLEVR (Johnson et al., 2017)](https://cs.stanford.edu/people/jcjohns/clevr/), but these small worlds are not CLEVR-scale benchmarks. [Attribute diversity and systematicity in VQA (Berlot-Attwell et al., 2024)](https://aclanthology.org/2024.emnlp-main.537/) motivates examining whether familiar attributes transfer to new combinations. [Progressive curriculum learning for VQA (Akl et al., 2025)](https://bmvc2025.bmva.org/proceedings/294/) shows that progressive VQA training is not itself a novel concept. This project adds controlled learning-history and counterfactual questions to that broader line of work; it does not claim human-like developmental learning or intrinsically biological plasticity.

[Medium World EDA](medium_world_eda_findings.md) · [Evaluation and statistical interpretation](evaluation_methods.md)
