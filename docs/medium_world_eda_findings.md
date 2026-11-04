# Medium-world EDA — answers and research implications

**Source:** executed `eda_medium_world(1).ipynb` supplied on 4 November 2026. Results come from the saved outputs; the held-out test splits were not inspected. Counts of attributes in images below are **deduplicated by scene**.

## 1. Dataset inventory and label balance

| Split | Question rows | Distinct scenes | Object occurrences | Tasks |
|---|---:|---:|---:|---|
| train | 10,000 | 5,000 | 19,905 | 2,500 each of color, shape, conjunction, pair |
| id_dev | 1,024 | 512 | 2,044 | 256 per task |
| ood_dev | 400 | 400 | 1,677 | pair only |

Each train/ID-dev scene is queried **twice**; each OOD-dev scene once. Thus **question-weighted counts are not independent image counts**. All tasks are **exactly 50% Yes / 50% No**, with no scene-ID overlaps between inspected splits. Mean object count is **3.98 train**, **3.99 ID-dev**, **4.19 OOD-dev**.

## 2. Which colors and shapes most often occur in scenes?

Training color objects: **yellow 5,009** (largest), green 3,739, blue 3,738, purple 3,719, red 3,700. Percent of distinct training scenes containing each: **yellow 67.8%**, green 56.1%, blue 55.3%, purple 55.3%, red 54.8%. Yellow is more common but the other four are close to one another.

Training shape objects: **circle 5,136** (largest), triangle 4,972, square 4,919, crescent 4,878. Distinct-scene presence: circle **68.3%**, triangle 66.7%, square 66.2%, crescent 65.9%. Shape imbalance is modest.

A scene has on average **2.89 distinct colors** and **2.67 distinct shapes** in train, among ~3.98 objects. Hence objects sharing a color or shape are frequent and object-level binding cannot be reduced to assuming every attribute identifies a unique object.

## 3. Which attributes are most often asked, and how does this vary by task?

The simple training `color` task is near uniform: red 502, blue 498 and each other color 500 requests. The simple `shape` task is also near uniform: circle 626, triangle 626, square 625, crescent 623.

In `conjunction` and `pair`, **yellow is asked more frequently**: conjunction asks yellow **626** times versus **466–472** for other colors; pair asks yellow **621** times versus **468–473** otherwise. This parallels yellow's higher object prevalence. Query shape counts are close to uniform in both tasks (~621–628 each).

All 16 allowed training color–shape pairs have close frequencies (for example `pair`: **153–159** asks per pair); the four held-out combinations are not asked in train.

## 4. Does the requested attribute appear in the image? Does this depend on answer?

**Presence probabilities in train** (percentage of *questions*, conditional on task and answer):

| Task | Requested attribute | No (label 0) | Yes (label 1) |
|---|---|---:|---:|
| Color | Color exists | 0% | 100% |
| Shape | Shape exists | 0% | 100% |
| Conjunction | Color exists | **42.72%** | 100% |
| Conjunction | Shape exists | **57.28%** | 100% |
| Conjunction | Both independent attributes exist | 0% | 100% |
| Pair | Color exists | 100% | 100% |
| Pair | Shape exists | 100% | 100% |
| Pair | Exact same-object pair exists | 0% | 100% |

For conjunction, **60.72% of positive examples also contain the exact queried pair**, but that coincidence is *not required* by the conjunction rule. The remaining positive cases still need both attributes but can have them on different objects.

## 5. Are there question-only target/label associations?

Every task's overall Yes rate is 50%. In the notebook's training target-conditioned audit, deviations from 50% are at most roughly **0.33 percentage points** among the displayed top rows; queried attribute identity alone provides no substantial label-rate shortcut in those aggregates. This does **not** rule out visual shortcuts or correlations involving multiple question tokens.

## 6. Is conjunction solvable by a shortcut? Are pair negatives hard?

The notebook computes *oracle rules on scene attributes*, not model predictions:

| Training task | Color-only rule | Shape-only rule | Both independent attributes |
|---|---:|---:|---:|
| Conjunction | **78.6%** | 71.4% | **100%** |
| Pair | 50.0% | 50.0% | 50.0% |

In ID-dev, conjunction color-only is **79.7%**, shape-only **70.3%**. This is an important **generator-induced asymmetry**: a model with imperfect shape recognition can score close to 80% on conjunction by using only color.

For pair questions, **100% of negative examples contain both requested attributes on different objects, without the target pair**. Therefore independent attribute detectors are chance-level (50%) on the balanced pair task. The scene-derived oracle agrees with **all 11,424** inspected question labels, indicating the metadata and simple-task semantics are consistent.

**Caution:** high shortcut-rule accuracy does not prove the trained model actually relies on that rule. Prediction-conditioned analyses are required for that claim.

## 7. Does object count explain the answers or OOD difficulty?

Medium-world scenes have **3–5 objects**, with means ~4.0. Train mean by task/label (No → Yes): color **3.95 → 4.00**, conjunction **3.98 → 4.00**, pair **4.04 → 3.98**, shape **3.88 → 4.02**. These are relatively small differences; they do not provide a compelling *simple mean-count shortcut* for the task labels, although a stratified test could quantify any residual association.

OOD-dev has **4.19** objects/scene versus **3.98** in train. It also has more distinct colors (**3.06** vs **2.89**) and shapes (**2.87** vs **2.67**). Hence OOD-dev differs in scene complexity, and OOD performance cannot be attributed strictly to held-out combinations without a matched-complexity analysis.

Repeated color appears in **77%** and repeated shape in **84%** of unique training scenes. Geometric variation is present: object radii 5–8 pixels and several allowed angles (0° most frequent). This EDA does not test model accuracy conditioned on these geometry factors.

## 8. Are held-out combinations truly absent from training?

Yes. The four OOD queried pairs are **blue + square, green + circle, purple + triangle, red + crescent**. For each, **zero occurrences as rendered objects** and **zero pair-question targets** in train and ID-dev, while OOD-dev contains both examples and rendered target instances:

| Held-out pair | OOD-dev question targets | Rendered OOD-dev object occurrences |
|---|---:|---:|
| Blue + square | 97 | 65 |
| Green + circle | 109 | 77 |
| Purple + triangle | 111 | 76 |
| Red + crescent | 83 | 54 |

The four OOD target counts sum to 400, matching all OOD-dev questions (pair only). Their object occurrence counts are not equal to their question counts, because negative examples have no true instance of the queried pair.

## 9. Differences across train, ID-dev and OOD-dev

The **train and ID-dev question mixes** are matched (four tasks with balanced labels), and both have two questions per image. OOD-dev contains **only binding (`pair`)** questions and distinct held-out combinations, plus mildly more objects and attribute diversity. We should report it as an OOD binding evaluation, not a directly comparable overall-task split.

## 10. Implications for staged-learning experiments

- The dataset strongly differentiates **independent conjunction** from **same-object binding**; pair negatives cannot be solved by detecting attributes separately.
- A substantial **color-only conjunction shortcut** is built into the generator. Therefore use **binding accuracy as the primary metric**; conjunction accuracy alone is not sufficient evidence of composition.
- The current object-count plot needs only a clearer *visualization*: all numerical object-count summaries were already recorded. **No rerun is necessary to support the findings above.**
- To isolate whether OOD failure comes from unseen bindings, an additional **matched-object-count OOD breakdown** would be valuable later, but it is not needed to update this notebook's graph.
