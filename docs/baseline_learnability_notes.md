# model feasibility study implementation and empirical status — 2026-10-25

## Completion boundaries

| Component | Status | Evidence |
|---|---|---|
| dataset generation manifests, hash audit, scene and query oracle | PASS | Reused unchanged; `outputs/dataset` |
| Model input contract (pixels + query tokens) | PASS | `tests/test_learning.py` |
| ViT encoder + question Transformer + tokenwise cross-attention fusion | IMPLEMENTED | `src/visual_world/learning.py` |
| CPU model forward/backward/optimizer smoke | PASS | `outputs/experiments/transformer_baseline/smoke_result.json` |
| 8-example overfit (48-dimensional minimal ViT) | PASS | Reached 100% **training** accuracy after 10 steps in `outputs/experiments/transformer_baseline/overfit_result.json` |
| Full training CLI + best checkpoint and metric export | PASS (engineering) | One-epoch end-to-end CPU smoke completed |
| Text-only / image-only baseline CLI | PASS (engineering) | One-epoch CPU runs completed; cannot establish absence of shortcuts |
| CLOSER adaptation CLI with stratified internal dev split | PASS (engineering) | Frozen-vision and full-finetune one-step CPU integration test completed |
| Full-dataset representation learning | **NOT YET DEMONSTRATED** | 12-epoch first-architecture mixed run remained 0.50 ID/OOD-dev; 20-epoch revised-tokenwise-fusion Level-1 diagnostic remained at 0.50 ID/OOD-dev |
| RTX 5050 execution | **NOT TESTED** | GPU is on user's laptop; only local CPU available here |
| Progressive/mixed/reverse curriculum comparison | **NOT STARTED** | Requires prior learned baseline |

## Why a 50% result is important

The dataset intentionally balances binary labels. Repeated 50% development accuracy means the next curriculum studyurrently provides no evidence of learning on unseen scenes. No paper-style claims should be made, and starting comparative curricula now would be premature. The 8-sample overfit checks gradient flow and capacity to memorize, not learning of a transferable visual concept.

## Immediate diagnostic actions

1. Install a CUDA 12.8-capable PyTorch wheel, confirm `torch.cuda.is_available()==True`, run `pytest` and `smoke` on GPU.
2. Re-run the 8-example overfit check on the RTX 5050. It should reach ≥97% training accuracy.
3. Run Level-1-only preliminary (`train --levels 1 --epochs 30`). Check training loss and **per-template** and aggregate ID-dev accuracy. If still near chance, stop and send results.
4. If Level-1 remains flat, debug in this order: (a) the positive/negative rendered scenes with queries, (b) single-template `EXISTS_COLOR` or `EXISTS_PAIR` learning, (c) multimodal feature gradients / query-conditioned attention, (d) increase examples or change inductive bias in the same fixed backbone. Preserve all data manifests and compare paired seeds after the issue is understood.
5. Only once Level-1 generalizes should we train all three levels, text-only and image-only baselines, and adaptation. Then freeze a revised next curriculum study curriculum protocol. Don't tune using final test sets.

## Honest interpretation

No actual local RTX 5050 training was performed. One-step adaptation accuracies ~0.5–0.6 from 32 development examples are **pipeline diagnostics** and have high sampling uncertainty; they are not estimates of adaptation or model intelligence. The research question remains open.
