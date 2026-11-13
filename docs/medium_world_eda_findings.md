# Medium World EDA: what the dataset makes easy—and what it does not measure

*November 13, 2026*

The Medium World is meant to distinguish detecting Color and Shape from combining them across, or within, objects. Its label balance and hard Binding negatives are useful, but the way scenes and questions are sampled also creates shortcuts. This report examines those facts as constraints on how the model's later accuracy should be interpreted.

The principal distinction is **label balance versus capability balance**. A dataset can have exactly half Yes and half No answers while a model can still obtain high accuracy by ignoring an attribute or detecting a scene-level correlation rather than answering the question as intended.

## 1. What is in the frozen data?

| Split | Questions | Unique scenes | Object occurrences | Tasks |
|---|---:|---:|---:|---|
| Train | 10,000 | 5,000 | 19,905 | Color, Shape, Conjunction, Binding (2,500 each) |
| ID-dev | 1,024 | 512 | 2,044 | Same four tasks (256 each) |
| OOD-dev | 400 | 400 | 1,677 | Binding only |

The generator also defines separate **ID-test** and **OOD-test** splits (1,024 and 400 questions respectively); their outcomes are not part of this EDA. No inspected scene ID overlaps across splits. Every original task/split combination has **50% Yes and 50% No** labels.

Train and ID scenes each receive **two distinct questions**, whereas each OOD scene is queried once. A collection of question rows is therefore not a collection of independent images. This matters especially for uncertainty estimates and the expanded diagnostic grids that query one image many times.

Scenes contain **3–5 objects**. Their mean counts are 3.98 in Train, 3.99 in ID-dev and 4.19 in OOD-dev. The visual world has five colors and four shapes, with radii 5–8 pixels, shape-dependent rotations, a fixed renderer, and non-overlap constraints.

## 2. Scene attributes are repeated and not equally frequent

The original training scenes contain these object occurrences:

| Color | Count |
|---|---:|
| Yellow | **5,009** |
| Green | 3,739 |
| Blue | 3,738 |
| Purple | 3,719 |
| Red | 3,700 |

Yellow occurs in **67.8%** of unique training scenes; each of the other colors occurs in roughly **54.8–56.1%**. This is related to the pair holdout design: Yellow retains four allowed Shape partners, while Red, Blue, Green and Purple each lose one of their four possible combinations.

The four Shape counts are much closer (Circle 5,136; Triangle 4,972; Square 4,919; Crescent 4,878). A typical training scene has **2.89 distinct colors** and **2.67 distinct shapes** among its 3.98 objects. Repeated colors occur in approximately **77%** of training scenes and repeated shapes in **84%**.

These repetitions make object association necessary for some queries: one attribute need not identify a unique object. They also create possible *statistical* signals—how many objects match an attribute, where they are placed, and what distractors coexist with them—that can correlate with a label without enforcing the intended logic.

## 3. Balanced questions do not guarantee balanced evidence

The training target selection is close to uniform over *allowed* Color–Shape pairs (approximately 153–159 Pair queries per allowed combination). Simple Color and Shape queries are also nearly evenly distributed. At the aggregate level, a query-only majority lookup has little target-specific label advantage because targets are balanced across Yes and No labels.

But the **scene conditional on the question and answer** differs substantially across tasks:

| Task and label | Queried Color present? | Queried Shape present? | Queried same-object pair present? |
|---|---|---|---|
| Color / No | No | Not relevant | Not relevant |
| Color / Yes | Yes | Not relevant | Not relevant |
| Shape / No | Not relevant | No | Not relevant |
| Shape / Yes | Not relevant | Yes | Not relevant |
| Conjunction / No | Present in 42.72% of negatives | Present in the other 57.28% | No |
| Conjunction / Yes | Yes | Yes | May or may not occur |
| Binding / No | Yes | Yes | **No** |
| Binding / Yes | Yes | Yes | **Yes** |

For Conjunction negatives, **42.72%** contain the queried Color but lack the queried Shape; the remaining **57.28%** contain the queried Shape but lack the queried Color. When both independent attributes occur, Conjunction is always Yes, even when they occur on different objects.

For Binding negatives, both attributes occur but are on **different objects**. Therefore detecting their independent presence alone yields no information about the balanced Binding answer. This is an important strength of the generator—but it does not remove all non-binding cues.

## 4. Conjunction permits a strong Color-only shortcut

Consider the heuristic:

$$
\widehat{J}(I,c,s)=C(I,c),
$$

which ignores the Shape entirely. Because all Conjunction positives contain the requested Color, and 57.28% of negatives lack it, the heuristic reaches

$$
\frac{1}{2}\left(1+0.5728\right)=78.64\%
$$

on balanced training Conjunction questions. The measured ID-dev value is **79.69%**. The opposite Shape-only heuristic reaches approximately **71.4%** in Train and **70.3%** in ID-dev.

| Oracle-derived heuristic | Train Conjunction | ID-dev Conjunction | Train Binding |
|---|---:|---:|---:|
| Color present | **78.6%** | **79.7%** | 50% |
| Shape present | 71.4% | 70.3% | 50% |
| Both attributes present independently | 100% | 100% | 50% |

These are **oracle-metadata rules**, not measurements of what the model has internally learned. They establish the difficulty of the benchmark: a model that completely neglects Shape may appear proficient at Conjunction.

Even the Conjunction positives are heterogeneous. In **60.72%** of positive training cases, the requested Color and Shape happen to coexist on the *same* object; in the remaining cases they can occur on different objects. A strong Conjunction score alone cannot distinguish whether the model learned independent attribute detection, a same-object co-occurrence rule, or a Color-dominant approximation.

The most discriminating negative subgroup is **Color present, Shape absent**. On the original ID-dev set, this subgroup contains **52** of the **128** Conjunction negatives; a Color-only classifier rejects **none** of them. Later constant-LR model diagnostics show only about **24–28%** correct rejection at the final checkpoint. This directly supports the description **Color-dominant, not strictly Color-only**: the model sometimes uses additional evidence, but its performance remains weak where Shape is the deciding missing attribute.

**Meaning:** Conjunction accuracy around 80% should not be used as proof that the model mastered a prerequisite logical-AND operation. The subgroup behavior contradicts that interpretation.

## 5. Binding negatives are hard, yet counting remains a possible confound

A correct Binding answer is

$$
B(I,c,s)=\mathbb{1}\{\exists o:\operatorname{color}(o)=c\land\operatorname{shape}(o)=s\}.
$$

Every original negative Binding question contains the queried Color and Shape somewhere in the scene. A rule using only either attribute's binary presence—or their independent logical AND—achieves **50%** accuracy on the original balanced task. Hence the dataset removes the most obvious shortcut from separate attribute detectors.

It does **not** guarantee that the model bases its answer on same-object comparison. Total object count, the number of objects bearing each queried attribute, other attribute combinations, object geometry and the spatial arrangement of distractors may all correlate with Binding labels.

The original full-dataset EDA found only small differences in mean object count between negative and positive training Binding questions (**4.04 versus 3.98**). But similar marginal means do not imply the absence of useful count-conditioned rules.

### What the later count-controlled audit clarified

A targeted analysis selected ID-dev Binding questions where there is **exactly one queried-Color object and exactly one queried-Shape object**. It contains **109 questions: 34 Yes and 75 No**.

On this subset, the two requested-attribute counts are identical regardless of the correct answer; their equality does not specify whether the same object supplies both. But other features remain imbalanced:

- Positive scenes average **4.15 objects**, compared with **3.80** for negative scenes.
- A simple `n_objects >= 4` metadata rule achieves approximately **59.6% balanced accuracy**.
- Stratifying by total object count weakens that particular alternative: subsequent Constant-LR model analyses retain above-chance discrimination within the 3-, 4- and 5-object strata. That supports sensitivity beyond the tested count cues, not exclusive reliance on a symbolic object-level mechanism.

Because this selected subset has a No-majority, an **always-No predictor gets 68.81% raw accuracy** but **50% balanced accuracy**. For this diagnostic, raw accuracy cannot be compared directly with the original balanced benchmark's 50% majority-class baseline. Always report positive recall, negative specificity and balanced accuracy or AUC.

The separate **attribute-preserving Color-swap experiment** is more direct: it holds a scene's Color and Shape multisets, positions and shape geometry fixed while changing which object bears which Color. Success on both versions of such a pair is evidence of *behavioral association sensitivity* beyond global counts; it still does not identify an explicit object-slot representation.

## 6. The held-out split has a decisive label-construction shortcut

The four held-out pairs are **Blue–Square, Green–Circle, Purple–Triangle and Red–Crescent**. They appear as neither rendered objects nor queried Pair targets in Train or ID-dev.

| Held-out pair | OOD-dev queries | Rendered OOD-dev object occurrences |
|---|---:|---:|
| Blue–Square | 97 | 65 |
| Green–Circle | 109 | 77 |
| Purple–Triangle | 111 | 76 |
| Red–Crescent | 83 | 54 |

The counts of rendered objects need not equal target-query counts, because a negative query refers to a pair not present on any object.

The deeper problem is how the generator creates OOD labels. It constructs **Yes scenes** from all permitted and held-out combinations, requiring the queried held-out pair to appear. It constructs **No scenes** using only the 16 training-allowed combinations, then asks a held-out pair whose two attributes appear separately.

Therefore, for the **400 OOD-dev questions**:

$$
\mathbb{1}\{\text{any held-out pair appears anywhere in the scene}\}
=\text{correct answer}.
$$

This metadata rule is correct on **400/400** questions. It is independent of which held-out pair the query actually asks about.

This **does not establish that the ViT discovered the shortcut**. Detecting those unfamiliar objects visually might itself be difficult; in fact, later count-controlled OOD analysis found approximately **49.8–52.1% balanced accuracy** for Constant-LR models. But the benchmark is incapable of cleanly identifying whether the model has generalized the *queried same-object relation* to novel pairs.

OOD-dev also differs from Train in scene complexity:

| Quantity per scene | Train | OOD-dev |
|---|---:|---:|
| Objects | 3.98 | **4.19** |
| Distinct colors | 2.89 | **3.06** |
| Distinct shapes | 2.67 | **2.87** |

And OOD-dev contains only Binding questions, unlike the four-task ID-dev distribution. All of these differences matter when interpreting apparent OOD gains or losses.

**Conclusion:** the existing OOD split is a shifted Binding evaluation, **not an identified test of zero-shot compositional transfer**. Statements such as “curriculum X generalizes to unseen combinations” or “the architecture cannot transfer composition” are not warranted by its aggregate accuracy.

## 7. What the dataset still supports

The generator does establish several useful things:

1. **The oracle is consistent with the recorded labels.** The original EDA audited **11,424** train/ID-dev/OOD-dev question records, matching their symbolic answers. This is an audit of those records, not a claim that every potential visual shortcut is excluded.
2. **Independent attribute presence is insufficient for Binding hard negatives.** Their construction requires distinguishing where attributes are paired, or using an alternative scene-correlated signal.
3. **Color-only Conjunction is a strong known baseline.** A model must be tested on missing-Shape negatives before Conjunction proficiency can be claimed.
4. **Attribute and scene count analyses are necessary but not decisive.** Count-free controls and fixed-multiset swaps are more informative about association behavior than raw Binding accuracy alone.
5. **Query-level label balance is not a guarantee of distributional comparability.** The OOD label construction supplies a perfect metadata-level shortcut and also changes scene complexity.

Taken together, these observations change how the learning-order results should be read. We can still ask whether curricula change acquisition, retention, count-controlled Binding and response to interventions **on this frozen dataset**. We cannot conclude that any schedule demonstrates a reusable concept hierarchy merely because it reaches high Conjunction accuracy or modestly exceeds chance on the original OOD split.

The most important distinction is between **what the oracle question logically requires** and **what the sampled distribution permits the model to exploit**. The evaluation must measure the gap between the two.

[Dataset design](dataset_design.md) · [Evaluation methods](evaluation_methods.md) · [Diagnostic follow-up](diagnostic_followup_findings.md)
