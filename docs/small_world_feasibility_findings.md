# Small World: What the Early Model-Feasibility Tests Established

*13 November 2026*

Before comparing learning schedules, the first task was to check whether the visual-question model could be trained at all. This was a **feasibility investigation**, not a completed curriculum comparison. Its negative results are important because a model at chance on development data cannot support conclusions about which task order is better.

## The tested setup

- The original data had **1,800 training questions**, **360 ID-dev**, and **360 OOD-dev**, with Color/Shape/Pair, Left/Above, and XOR tasks. A separate CLOSER adaptation pool contained 128 examples.
- The early architecture used a randomly initialized ViT over **64×64 RGB images**, symbolic query tokens, a Transformer query encoder, tokenwise cross-attention, fusion, and a binary prediction head. The baseline protocol's larger configuration used **8×8 patches**, width 96, three image-encoder layers and two query-encoder layers. A smaller 48-dimensional variant was used for an overfit check.
- Object coordinates, scene roles, oracle program trees and labels were used by the generator or evaluator, **not provided as model inputs**.

A smoke test checks that forward passes, loss calculation and parameter updates run. A tiny-set overfit test asks a different question: can the network memorize a few examples? Neither demonstrates that it can answer new scenes correctly.

## Observations

| Check | Recorded result | What it establishes |
|---|---|---|
| Forward/backward/optimizer smoke | Completed on CPU | The training pipeline executes |
| Eight-example overfit, 48D model | **100% training accuracy after 10 steps** | The model can fit this tiny set; not generalization |
| First full-data Mixed run | **~50% ID/OOD-dev** after 12 epochs | No demonstrated learning on unseen scenes in that run |
| Revised tokenwise-fusion Level-1 run | **~50% ID/OOD-dev** after 20 epochs | The revised setup also did not establish development learnability |
| Image-only/text-only CLI smoke | Completed for one CPU epoch | Wiring was tested; shortcut resistance was not measured |
| CLOSER adaptation smoke | One-step frozen-vision and full-finetune checks | The adaptation code runs; no reliable transfer conclusion |

The 128 CLOSER examples were split internally into **96 adaptation-training** and **32 adaptation-development** questions. One-step results around 0.5–0.6 on such a small development sample were pipeline diagnostics, not evidence of successful adaptation.

One methodological distinction must remain explicit: **“frozen vision” froze only the image encoder**. The query, cross-attention, fusion and head components were still trainable. It is not equivalent to probing a fully frozen end-to-end representation.

## What the failed development runs mean

The ability to memorize eight examples does not establish a usable representation for held-out scenes. Chance-level validation can arise from optimizer behavior, scene complexity, query/visual fusion, insufficient data or architectural limitations. The early checks did not isolate which explanation was responsible.

The correct historical conclusion is narrow: **the first attempted multimodal baselines had not demonstrated learnability at the point those feasibility tests were recorded**. Later experiments used different implementations and training budgets, including runs that learned simple tasks above chance. Their successful outcomes do not erase the early failure, but the early failure must not be described as a permanent incapacity of ViTs or as a result from the later Medium World architecture.

## Implication for the research

A curriculum comparison is interpretable only after checking that the backbone can learn the tasks under the same data and evaluation rules. The feasibility stage also showed why tiny-set memorization, balanced labels, and a single query-only heuristic are individually inadequate as proof of generalizable visual reasoning.

Sources: the saved [`transformer_baseline` outputs](../outputs/small_world/experiments/transformer_baseline/), [`src/small_visual_world/learning.py`](../src/small_visual_world/learning.py), and the dataset audit described in [Small World EDA](small_world_eda_findings.md). The results above report what the original preliminary checks observed; they are not newly executed tests.
