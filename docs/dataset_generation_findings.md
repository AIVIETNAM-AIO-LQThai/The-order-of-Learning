# AIO Conquer Module 5 — dataset generation Dataset Audit

**Date:** 2026-10-25  
**Protocol:** `docs/dataset_design.md`  
**Status:** **PASS for generator/oracle/scene-split structural gates**, **NOT YET cleared for causal training conclusions**.

## Generated next curriculum studyorpus

| Partition | Records |
|---|---:|
| `adapt_test` | 240 |
| `adapt_train` | 128 |
| `id_dev` | 360 |
| `id_test` | 360 |
| `ood_dev` | 360 |
| `ood_test` | 360 |
| `train` | 1,800 |
| **Total disjoint scene graphs** | **3,608** |

The preliminary uses 600 training queries per level; this is much smaller than the provisional 12,000/level research scale and is only an engineering feasibility dataset. RGB images are produced deterministically from scene graphs at load time.

## Automated checks

| Check | Status |
|---|---|
| `closer_margin` | **PASS** |
| `counterfactual_integrity` | **PASS** |
| `graph_disjointness` | **PASS** |
| `holdout_isolation` | **PASS** |
| `manifest_hashes` | **PASS** |
| `oracle_crosscheck` | **PASS** |
| `per_template_answer_balance` | **PASS** |
| `primitive_coverage` | **PASS** |
| `reference_grounding` | **PASS** |
| `role_feasibility` | **PASS** |
| `scene_geometry` | **PASS** |
| `xor_truth_balance` | **PASS** |

Unit test suite: **8 passed**. Full-sized preliminary regenerated independently and all **7 manifest SHA-256 digests matched**.

## Critical sampling correction made during implementation

The first version restricted every ID task to configurations where the *fourth absent pair* was also outside the OOD holdout set. That unintentionally made yellow present in every ID image: `EXISTS_COLOR yellow` could be answered using a class prior. **This was corrected before releasing the preliminary:** only `EXISTS_PAIR` hard negatives use that strict subset; all other ID tasks draw from the larger feasible all-visible-allowed pool. We also explicitly match the distribution of exact query strings across positive and negative labels, as far as the fixed task plan permits.

## Text-only shortcut diagnostic (non-neural)

The audit implements a deliberately simple exact-query majority lookup fitted on training question tokens without seeing any image pixels. Zero-lookup query strings default to class 0. Accuracy on these balanced preliminary samples:

| Split | Color | Shape | Pair | LEFT | ABOVE | XOR |
|---|---:|---:|---:|---:|---:|---:|
| `id_dev` | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% |
| `id_test` | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% |
| `ood_dev` | — | — | 50.0% | 50.0% | 50.0% | 50.0% |
| `ood_test` | — | — | 50.0% | 50.0% | 50.0% | 50.0% |

**Caution:** Exactly 50% performance by this heuristic does not demonstrate that all possible text-only or image-only classifiers are at chance. We still need trained ablation baselines in model feasibility study.

## Counterfactual checks

Generated **8 paired interventions**: three attribute-binding questions; one `LEFT` query; one `ABOVE` query; three `XOR` questions with a held-out target reference. Each intervention flips its oracle label; XOR interventions flip exactly one constituent predicate. Every rendered pair is available in `counterfactuals_8_pairs.png` with structured provenance in `counterfactuals.json`.

## Remaining research design risks

- The train-to-OOD change includes novel color–shape anchor binding, so weak Level-3 OOD accuracy could be caused by failing Level-1 reference grounding rather than Boolean composition. Always report the full task decomposition.
- Synthetic scenes are exceptionally small and clean; broad claims about object perception or human cognitive development are not warranted.
- The independent orbit of scene positions is balanced at task level, but fine-grained nuisance-factor marginals may differ and need explicit evaluation before the final freeze.
- Initial `CLOSER` train has only 128 examples and 240 held-out evaluation examples in this preliminary; benchmark uncertainty will be limited.
- A common strict 0.5 query-only result is insufficient to certify the future neural model uses the image. Perform trained query-only and image-only ablations.
- No Transformer has been trained, therefore there are **no research findings** yet about curricula, abstraction, transfer, or plasticity.

## Next gate — model feasibility study

Implement a small from-scratch ViT with a symbolic query-conditioning module and one binary head. First overfit 32 samples (sanity), then train only one mixed-learning baseline on the preliminary dataset; use `id_dev` and `ood_dev`, keeping both test splits untouched for full research. Verify L1 and L2 learn above chance; inspect whether L3 XOR is learnable. Add a frozen-encoder CLOSER probe and a same-data query-only baseline. If these fail, fix model/input contracts before comparing schedules.
