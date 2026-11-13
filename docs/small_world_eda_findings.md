# Small World: Dataset Structure and Shortcut Audit

*13 November 2026*

The first visual world was designed to test attribute recognition, grounded spatial relations, and their logical composition. This audit asks whether those tasks were correctly generated and which distributional cues could affect a learning-order comparison. Dataset checks establish properties of the examples; they do not establish what a trained ViT uses to make predictions.

## 1. Data and task balance

| Inspected split | Questions | Distinct scenes | Objects | Task distribution |
|---|---:|---:|---:|---|
| Train | 1,800 | 1,800 | 5,400 | Color 200; Shape 200; Pair 200; Left 300; Above 300; XOR 600 |
| ID-dev | 360 | 360 | 1,080 | Color 40; Shape 40; Pair 40; Left 60; Above 60; XOR 120 |
| OOD-dev | 360 | 360 | 1,080 | Pair 120; Left 60; Above 60; XOR 120 |
| Adapt-train | 128 | 128 | 384 | CLOSER 128 |

Every scene has **three objects**, two distinct Colors and two distinct Shapes. Each inspected question has a balanced Yes/No label, and each scene receives one question in this world. The four relational predicates are defined by the symbolic scene oracle rather than image classification heuristics.

The full generator also produced ID-test and OOD-test (360 questions each) and adaptation-test (240 questions), for **3,608 scene-disjoint records** across seven partitions. The generation audit checked oracle consistency and structural properties of those splits. Those checks are not evidence that a model was evaluated successfully on held-out tests.

## 2. What appears in the images

- **Colors in training:** Yellow occurs on 2,231 of 5,400 objects and in **75.5% of scenes**, compared with Red 1,087, Blue 1,067, and Green 1,015 object occurrences. Yellow is consequently overrepresented in many relational questions.
- **Shapes in training:** Circle 1,849 occurrences; Square 1,797; Triangle 1,754. Scene presence is approximately 65–68%, considerably more even than Color.
- **Sizes:** 2,711 large and 2,689 small objects. Object count and repeated attributes cannot distinguish labels within Small World, since every scene has exactly three objects and repeats both a Color and a Shape.
- **Question targets:** Simple Color questions ask each Color exactly 50 times. More complex spatial questions reference Yellow disproportionately, reflecting the underlying scene construction.

The scene-level skew matters even when Yes and No labels are balanced. A model can encounter some colors or relations more frequently, leading to uneven learned evidence without any mistake in the oracle.

## 3. Which shortcuts were excluded?

For every negative `EXISTS_PAIR` question in the inspected train, ID-dev and OOD-dev samples, the requested Color and Shape both occur in the scene, but **not on the same object**. Color-only and Shape-only existence rules therefore score **50%** on this balanced task. That removes the simplest independent-attribute shortcut; it does not rule out all image-level or query-conditioned correlations.

`LEFT` and `ABOVE` reference present objects. XOR combines their two truth values, with all four constituent truth patterns balanced. A correct result must therefore go beyond deciding whether the referenced attributes exist. `CLOSER` uses an oracle geometric distance comparison; the 128 adaptation examples split into 64 Yes and 64 No with a separation margin, not overlapping distance ties.

The recorded oracle audit found agreement on the **840** checked simple Color/Shape/Pair questions. Additional generator checks passed for graph disjointness, holdout isolation, scene geometry, unique references, balanced XOR truth patterns and manifest hashes. Eight oracle-validated counterfactual pairs also flipped their intended answers (three Pair, one Left, one Above and three XOR). These are *generator validation*, not trained-model counterfactual results.

A basic exact-query majority lookup, trained without seeing image pixels, scored **50%** across the reported task/split cells. This excludes that particular text-only heuristic on the tested generator; it does not establish that every possible query-only classifier must fail.

## 4. An important correction in the original generator

The first sampling rule required the *absent fourth Color–Shape pair* to be outside the held-out set in every ID task. That restriction unintentionally made Yellow present in every ID image. It was corrected before the audited dataset was released: the strict absent-pair rule remains only where it is needed to generate hard-negative Pair questions, while the other tasks draw from a broader set of valid scenes.

The correction matters scientifically because deterministic Color presence could make an attribute question solvable from a class prior rather than from pixels. Passing a balanced-label check alone would not have discovered that problem.

## 5. What does OOD change?

Held-out combinations are **Blue–Square, Green–Circle and Red–Triangle**. None appears in training objects or Pair targets. They do occur as OOD questions and objects.

But OOD also changes Color prevalence: Yellow is present in **75.5%** of training scenes, **70.8%** of ID-dev scenes and only **35.8%** of OOD-dev scenes. The OOD contrast therefore includes more than unseen pair composition. Failure on a relational OOD question might reflect reference grounding, attribute-distribution shift or the relation itself. These explanations require separate controls.

## Interpretation

Small World established a consistent symbolic generator with genuinely nontrivial Pair negatives and balanced logical truth patterns. It also exposed meaningful sampling asymmetries and the limits of a small, rigid scene grammar.

The audit cannot certify model learnability or generalization. The dataset's structure, plus the limited dev support (only **40 Pair questions**), explains why a single favorable curriculum run required replication and why subsequent experiments moved to a larger Medium World.

Evidence: [EDA notebook](../notebooks/small_world/eda.ipynb), [generator and audit code](../src/small_visual_world/), and the saved Small World dataset/audit outputs under [`outputs/small_world/`](../outputs/small_world/).
