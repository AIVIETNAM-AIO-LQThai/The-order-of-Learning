# AIO Conquer M5 — model feasibility study (Model Learnability and Adaptation) v0.2

## Scope and boundaries

model feasibility study checks **engineering feasibility** with the dataset generation dataset. It is **not** the comparison of progressive / mixed / reverse curricula. Only development splits are used; the final `id_test`, `ood_test`, `adapt_test` are intentionally not accessed by `baseline_training.py`. Run `python train_multimodal_baseline.py smoke` first, then `overfit`, `train`, `baselines`, and `adapt`.

## Architecture

- RGB input is **64×64**, split into **8×8 patches** by a stride-8 convolution (64 visual tokens).
- A small randomly initialized ViT contextualizes image patch tokens (default 3 layers, width 96, 4 attention heads).
- Symbolic prefix query tokens are embedded with trainable positional vectors; a question Transformer (default 2 layers) represents the query.
- Each query token separately cross-attends to the patch features. An additional fusion Transformer contextualizes this evidence before a binary MLP predicts truth/falsehood. This avoids reducing the whole image to one attention-pooled token before evaluating multi-relation queries.
- The model *never* sees scene-role identifiers, object coordinates, split identifiers, oracle AST, the label, or scene hashes. These are used only to generate rendered pixels and metadata for evaluation.
- The vocabulary includes `CLOSER` in advance, but it is absent from all original training questions. It is explicitly reset to a fixed random vector at adaptation onset to make its starting state comparable across conditions.

This is a **ViT plus a Transformer question encoder**, not a language-model pretrained VQA system. The chosen architecture is a feasibility baseline, not an optimized or biologically realistic learner.

## Data

dataset generation: 1,800 training, 360 ID-dev, 360 OOD-dev examples. Dev-level metrics are computed separately to avoid macro/micro confusion. Levels 1/2/3 correspond to attribute existence, binary relations, and XOR of relations. OOD questions involve color-shape attribute conjunctions withheld from original training.

Novel `CLOSER` examples are **only** from the 128 examples in `adapt_train.jsonl`; these are further deterministically partitioned into 96 adaptation-training and 32 adaptation-development examples (16 positives / 16 negatives in dev). `adapt_test.jsonl` remains sealed until the final protocol.

## preliminary sequence

1. **Smoke:** a model forward/backward pass for two optimizer steps on eight records; does not establish learnability.
2. **Small-batch memorization:** overfit a fixed subset of 8–16 records using the *same multimodal inputs*. Goal: ≥97% training accuracy if the model has adequate capacity/steps.
3. **Full mixed baseline:** train on the 1,800 train examples with mixed task levels, seed 42 (starting baseline); select the best checkpoint by **ID-dev accuracy only**, monitor OOD-dev but do not choose based on it.
4. **Shortcut checks:** train the same architecture with text-only (constant pixels) and image-only (constant question tokens), using the same examples. Record overall and template-level accuracy; expect near chance *if* the data distribution doesn't reveal shortcuts.
5. **Adaptation diagnostic:** start from the best full checkpoint; reset CLOSER operator embedding, then train on 96 novel-task samples. Compare **frozen-vision** (ViT image encoder frozen, text/fusion/head trainable) against **full-finetune** (all modules trainable). Plot adaptation-dev learning curves and changes in original ID-dev accuracy. Neither result by itself proves biological plasticity. Larger balanced adaptation datasets are needed for final claims.

## Interpretation and stop/go

- A passing smoke test is an engineering sanity check.
- Tiny-batch overfitting indicates finite representational capacity but no evidence of broad learning.
- If ID-dev stays near 50% over dozens of epochs, investigate input encoding, model capacity, optimizer, and query geometry **before** curriculum experiments.
- If a text-only or image-only baseline scores well above chance on stable repeated runs, examine sampling biases / spurious cues before proceeding.
- An OOD gap requires caution: held-out object pairs also affect grounding, so it does not isolate compositional reasoning from attribute binding.
- preliminary sample sizes and random-seed count are insufficient for statistical inference. Final curriculum comparison requires multiple matched seeds, larger balanced data, fixed training budgets, separate ID and OOD held-out tests, and an adapted-task evaluation protocol pre-registered prior to accessing tests.

## Reproducibility

Training checks the dataset generation audit and SHA-256 dataset manifests before accessing any rows. Best checkpoint includes architecture, source-training manifest digest, training seed, best epoch, and selected ID-dev metric. It uses a deterministic Python/NumPy/Torch seed (GPU execution may still be nondeterministic due to kernel selection). No automatic download, external API, or GPU cloud dependency is required.

## Known limitations

- Early baseline is single seed with a small synthetic dataset; it is NOT a curriculum study.
- Images are 64×64 and only a few objects are shown. Relations may be difficult for an unpretrained ViT to learn at this scale, and XOR has dependencies on two imperfectly learned predicates.
- `frozen-vision` is *not* a fully frozen representation probe. It only freezes the visual encoder; query/cross-attention/classification components update.
- Test labels are programmatically generated with dataset generation oracle; successful answers do not prove relational reasoning beyond the synthetic program grammar.
- No practical GPU results are available from this environment. Reported CPU overfit and smoke execution do not predict RTX 5050 throughput.
