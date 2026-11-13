# Five-seed curriculum experiment — answers and interpretation

**Source:** executed `analyze_plateau_five_seeds.ipynb` and its saved five-seed summary for seeds **42–46** (4 November 2026). Only **development** evaluations; no held-out test split is used. All accuracy differences below are **percentage points (pp)** unless noted.

## 1. Was the schedule comparison fair in training exposure?

**Yes for total optimizer updates and per-task presentations.** Every one of 15 runs completed **3,000 updates**, **150,000 example presentations**, and exactly **37,500 presentations per task** (`color`, `shape`, `conjunction`, `pair`). The schedules differ in order, not total amount of labeled training exposure.

**However, realized learning rates differ:** the adaptive scheduler responds to a schedule-dependent ID-dev loss. Therefore this experiment is not a pure _order-only, fixed-LR_ causal contrast.

## 2. Which schedule performs best on final metrics?

Mean ± sample SD over five seeds, fixed-final checkpoint:

| Metric                  |            Mixed |           Staged |       Reverse |
| ----------------------- | ---------------: | ---------------: | ------------: |
| Overall ID accuracy (%) | **72.46 ± 2.39** |     71.68 ± 2.35 |  63.63 ± 6.33 |
| Color ID (%)            | **96.80 ± 4.36** |     95.16 ± 3.99 | 79.22 ± 12.90 |
| Shape ID (%)            | **61.33 ± 1.33** |     60.47 ± 1.45 |  56.25 ± 5.12 |
| Conjunction ID (%)      | **78.13 ± 3.75** |     76.95 ± 4.48 |  66.25 ± 7.70 |
| Binding ID (%)          |     53.59 ± 2.73 | **54.14 ± 3.78** |  52.81 ± 1.69 |
| OOD binding (%)         | **54.20 ± 1.98** |     52.30 ± 5.52 |  51.70 ± 2.29 |
| Binding AULC (fraction) |        **0.530** |            0.521 |         0.514 |

**Reading:** staged has the highest _mean final ID binding_ accuracy but by only **+0.55 pp** over mixed. Mixed wins on overall ID, OOD binding, and binding efficiency. Binding remains near chance under all schedules, despite much stronger color and conjunction scores.

## 3. Do paired seeds consistently favor staged over mixed?

Final ID-binding differences (staged − mixed):

| Seed | Difference (pp) |
| ---- | --------------: |
| 42   |          −3.516 |
| 43   |          +1.172 |
| 44   |          −0.781 |
| 45   |          +2.734 |
| 46   |          +3.125 |

**3 of 5 seeds** favor staged on final binding. Mean paired advantage **+0.547 pp**, descriptive paired bootstrap 95% interval **[−1.64, +2.58] pp**. This interval spans zero; the data do **not** justify claiming a robust staged advantage.

Reverse is lower than mixed on overall ID for **all five seeds**, by an average **−8.83 pp** (descriptive bootstrap interval **[−14.02, −3.63] pp**). Reverse underperformance is much more consistent than staged's small binding advantage.

## 4. Does staged improve _learning efficiency_?

**No in this experiment.** Binding learning-curve area (AULC), averaged over the entire 0–3,000-update budget:

- Mixed: **0.5299**
- Staged: **0.5208**
- Reverse: **0.5136**

Staged − mixed AULC is **−0.0091** (about **−0.91 pp** of average binding accuracy across the training trajectory). Staged has higher AULC in **only 1 of 5 seeds**. The notebook's descriptive bootstrap interval is **[−0.0156, −0.0009]**. With five seeds, treat this as suggestive rather than decisive inference.

**Metric nuance:** AULC includes steps before binding is first introduced in staged training; that is intentional for measuring end-to-end sample/update efficiency, not an assessment of how rapidly _the binding-only stage itself_ learns.

## 5. What happens at stage transitions? Is there forgetting?

The most consequential transition is **staged conjunction → binding**, measured from step **1,800** to **1,900** (the first 100 binding updates), mean changes across five seeds:

| ID metric   |        Change |
| ----------- | ------------: |
| Color       | **−14.84 pp** |
| Shape       |      −2.97 pp |
| Conjunction | **−12.97 pp** |
| Binding     |      +3.59 pp |
| Overall     |  **−6.80 pp** |

This is strong observational evidence of **short-term interference/forgetting** when earlier tasks disappear from training. Starting **common mixed consolidation** (2,400 → 2,500) recovers on average **+15.70 pp color**, **+12.34 pp conjunction**, **+7.52 pp overall**, while binding dips **−2.27 pp** over the first 100 updates. This looks like a retention–adaptation trade-off; it does **not** show perfect recovery by the final checkpoint.

The attributes → conjunction boundary reduces color by **−3.67 pp** and increases conjunction by **+7.42 pp**, on average. Transition deltas are descriptive; they depend on the fixed 100-update observation window and contain schedule changes and continued optimization.

## 6. Is OOD binding better when ID binding is better?

**Not consistently.** Mixed mean OOD binding = **54.2%**, staged **52.3%**, reverse **51.7%**. Staged − mixed OOD difference = **−1.9 pp**, with a wide descriptive bootstrap interval **[−6.25,+3.90] pp**. Staged is higher on OOD in only **one of five seeds**.

Illustrative discrepancy: seed **46** staged reaches **55.47% ID binding** but **46.5% OOD binding**; seed **45** staged reaches **60.16% ID** and **61.25% OOD**. Results are variable, so higher ID performance cannot yet establish robust transfer to unseen pairs.

## 7. Are adaptive LR trajectories an interpretation confound?

**Yes.** All runs start at `1e-4` and use the same `ReduceLROnPlateau` rule, but actual reductions differ:

- Mixed: **0 reductions in all five seeds**; ends at `1e-4`.
- Staged: **1–3 reductions**, ending around `1.25e-5` to `5e-5`.
- Reverse: **1–5 reductions**, ending as low as roughly `3.125e-6`.

The scheduler monitors **overall ID-dev cross-entropy** even when the staged phase hasn't trained some tasks yet. This makes LR reductions coincide with changes in task exposure; the run combines order effects with adaptive optimization effects. It is still a valid comparison of **full training procedures**, but not clean evidence for order in isolation.

## 8. Conclusion and next experiment

The five-seed data show that **task order matters as part of the training procedure**: reverse is consistently poor overall, and staged shows sizeable short-term forgetting at its transitions. **They do not demonstrate that non-rehearsal staged training is better than mixed training** in efficiency, OOD binding, or stable final ID binding.

The next fair experiment should preserve the same architecture, examples, final exposure counts, and total updates while introducing a **rehearsal-based staged schedule** as a fourth, explicitly new variant. Compare it against existing schedules, and separately control or replay LR trajectories if the aim is to isolate task order from scheduler behavior. Avoid inspecting held-out test sets while designing the new strategy.
