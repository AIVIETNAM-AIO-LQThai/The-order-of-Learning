# Small-world EDA — answers and research implications

**Source:** executed `eda_small_world(1).ipynb` supplied on 4 November 2026. All numbers below come from the notebook's saved outputs, not from a new model evaluation. Splits inspected: `train`, `id_dev`, `ood_dev`, `adapt_train`; held-out tests are excluded.

## 1. How many scenes, questions, objects, and labels?

| Split | Questions | Distinct scenes | Object occurrences | Tasks |
|---|---:|---:|---:|---|
| train | 1,800 | 1,800 | 5,400 | above 300, color 200, left 300, pair 200, shape 200, xor 600 |
| id_dev | 360 | 360 | 1,080 | above 60, color 40, left 60, pair 40, shape 40, xor 120 |
| ood_dev | 360 | 360 | 1,080 | above 60, left 60, pair 120, xor 120 |
| adapt_train | 128 | 128 | 384 | closer 128 |

Every question uses its own scene in the inspected small-world splits (one question per scene); no scene-ID overlap was detected between splits. Every task in every inspected split is **exactly 50% Yes / 50% No**. The scene-count and question-count denominators therefore coincide here (but do **not** coincide in medium world).

## 2. Which colors/shapes appear most often? Does repeating them matter?

The most frequent training color is **yellow**: **2,231 object occurrences** versus red 1,087, blue 1,067 and green 1,015. Yellow appears in **75.5% of distinct training scenes**, compared with red 42.3%, blue 41.7%, green 40.4%. Yellow contributes ~41.3% of the 5,400 training object occurrences, even though there are four color categories.

The most frequent training shape is **circle** (1,849 occurrences, in 68.1% of scenes), followed by square (1,797; 66.5%) and triangle (1,754; 65.4%). Shape counts are much more balanced than colors.

All scenes contain exactly **three objects, two distinct colors and two distinct shapes**. Therefore a color and a shape always repeat within each scene; object-occurrence counts are **not equivalent** to scene-presence rates. The repeated attributes are a construction property, not evidence of naturally emerging repetition.

## 3. What attributes are asked most often, and does this depend on task?

**Simple attribute questions are deliberately balanced.** In the 200 training `color` questions, each of red/blue/green/yellow is asked exactly 50 times. In the 200 training `shape` questions, circle is asked 68 times, square 66, triangle 66.

The **nested/relational tasks favor yellow**, matching its high scene prevalence. Examples (distinct target mentions, training split): `above` asks about yellow 220 times versus red 130, blue 126, green 124; `left` 144 yellow versus 48–56 other colors; `xor` 440 yellow versus 248–260 others. For pair references, each of the three yellow combinations is queried 240–244 times in XOR, versus 172–188 for most nonyellow combinations. `EXISTS_PAIR` target counts are closer to uniform (yellow pairs 22 each; other valid pairs 22–24 each).

Thus **most asked** and **most depicted** coincide for yellow in relational questions, but **not** in the simple `color` task. Circle is the most depicted shape and slightly the most mentioned; the differences among shapes are small.

## 4. When a question mentions an attribute, is it actually in the image?

In training `EXISTS_COLOR` questions: requested color present in **0% of No**, **100% of Yes**. In `EXISTS_SHAPE`: requested shape present in **0% of No**, **100% of Yes**. In `EXISTS_PAIR`: both requested individual attributes are present in **100% of both labels**, but the *joint pair* exists in **0% of No**, **100% of Yes**.

For `LEFT`, `ABOVE`, `XOR` and `CLOSER`, all queried reference attributes and queried reference pairs are present irrespective of the answer. These tasks cannot be solved by merely checking reference existence; they require spatial, Boolean or distance relationships.

## 5. Can the answer be guessed from the question wording alone?

No strong target-specific label imbalance is reported. Overall Yes rate is 50% for each task/split; the notebook's high-count training target-by-task rows have Yes rates of 50% (zero percentage-point shift from task baseline). This is a **marginal audit** only: it does not prove the absence of every possible multifeature/text-only shortcut.

## 6. Are pair negatives truly difficult? How good are shortcut rules?

For the 200 training, 40 ID-dev, and 120 OOD-dev `EXISTS_PAIR` questions, the notebook reports:

- **100% of negative pair questions** show *both individual attributes* somewhere, while the requested pair itself is absent.
- Predicting Yes from the requested color's presence, the shape's presence, or both independently gives **50% accuracy**, i.e., the balanced-label baseline.
- The scene-derived oracle matches all **840** audited simple-operation questions (color/shape/pair across the splits).

So the generator **forces same-object binding** to distinguish the pair labels; color-only or shape-only existence rules are insufficient.

## 7. Do object count or repeated attributes make some questions easier?

**Object count cannot explain label differences in small world:** every question in every inspected task/split has exactly 3 objects (mean 3.0, SD 0.0). Each scene also always repeats one color and one shape. These variables have no variance within this dataset; their effect on model performance cannot be estimated here.

Object size *does* vary: train has 2,711 large versus 2,689 small object occurrences, approximately balanced. A model-performance breakdown by size was **not** performed in this EDA.

## 8. Which combinations are genuinely held out? What shifts across splits?

Three combinations are absent from **both training objects and training pair targets**: **blue + square, green + circle, red + triangle**. The OOD-dev split contains them as rendered objects (116, 68, 116 occurrences respectively) and as targets (136, 88, 136 mentions across tasks respectively, according to the notebook's aggregate target table).

The color distribution changes noticeably: yellow occurs in **75.5% of train scenes**, **70.8% of ID-dev scenes**, but only **35.8% of OOD-dev scenes**. OOD therefore differs in **color prevalence** as well as withheld attribute bindings. We should not attribute every OOD-performance change exclusively to unseen pairs.

## 9. What do the nested relational tasks reveal?

For `LEFT` and `ABOVE`, each question contains on average **two distinct reference pairs**, all present in the scene, so the model has to resolve the relationship rather than existence. XOR contains **three distinct reference pairs** on average, again all present. Across train/ID/OOD, the four XOR subquestion truth combinations `[0,0]`, `[0,1]`, `[1,0]`, `[1,1]` are evenly represented, with answer 1 for exactly the two mixed-truth cases, as expected from XOR.

`CLOSER` appears in **adapt_train only** (128 questions: 64 Yes / 64 No). Its signed margin is negative for all 64 No examples (max −4.39) and positive for all 64 Yes examples (min +4.80). No margin overlap is visible in the summary; whether the model exploits pixel-distance cues is not evaluated here.

## 10. Consequences for learning-order research

- The small world is balanced by labels, but **not uniformly distributed by object color** (yellow dominates), especially in relational questions.
- The held-out OOD split changes both **attribute combinations and color frequencies**; note this in generalization claims.
- No need to rerun to interpret object count: the notebook conclusively reports exactly 3 objects in every inspected small-world question.
- These are **dataset-level** findings; the notebook does not load model predictions, so it cannot establish that a trained ViT used any particular shortcut.
