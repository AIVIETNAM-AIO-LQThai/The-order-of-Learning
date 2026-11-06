# Learning-order control: frozen protocol

**Project:** The Order of Learning · Medium World  
**Status:** Design freeze for a seed-42 pilot; no new training has been performed.  
**Repository branch:** `experiment/staged-rehearsal`

## 1. Questions

1. Do separately staged prerequisite tasks improve subsequent same-object Binding compared with learning those prerequisites together?
2. If a mixed curriculum ends with concentrated Binding practice, does it approach the staged curriculum's final performance?
3. Are any differences explained by learning speed, individual-question forgetting, or generalization to withheld attribute combinations?

Keep the existing 64×64 visual world, four question types (`color`, `shape`, `conjunction`, `pair`), `AttributeBindingViT(dim=48, patch_size=4, depth=1)`, renderer, vocabulary, and frozen dataset hashes unchanged. There are 2,500 training questions per task, 10,000 total. ID-dev has 256 questions per task; OOD-dev has 400 Binding questions about four withheld color–shape combinations.

## 2. Controlled training conditions

- Paired seeds: **42, 43, 44, 45, 46**. First execute seed 42 only for protocol validation, not schedule selection.
- Optimizer: AdamW, constant LR **1e-4**, weight decay 0, batch size **50**; retain the current gradient clipping and image augmentation unchanged.
- **9,000 updates** (450,000 training-question presentations) per schedule.
- Every exact training question is presented **36 times** in the main phase and **9 times** in identical final common-mixed consolidation: **45 presentations per example**.
- This is **112,500 presentations per task**, with exactly the same examples, initialization seed and task-specific example order across schedules.
- Final **1,800 updates** are identical common-mixed consolidation for all schedules. For the first three schedules, the immediately preceding **900 updates** contain the *same* Pair examples in the *same* order.
- Do not change the dataset, model, LR, number of updates, or schedule proportions after looking at the seed-42 results. Checkpoint selection for primary results is the **fixed final step 9,000**.

## 3. Schedules and exposure allocations

Each task-epoch means that every question in that task appears once. `C` = Color; `S` = Shape; `J` = Conjunction; `P` = Binding/Pair. The schedules below exactly match 36 main passes per task, regardless of order.

| Curriculum | Main-phase sequence (updates) | Final common phase | Experimental role |
|---|---|---|---|
| `staged_pair_focus` | C36 + S36 mixed (**3,600**) → J36 (**1,800**) → P18 (**900**) → P18 focus (**900**) | All tasks 9 passes (**1,800**) | Prerequisite-first sequence |
| `prereq_mixed_pair_focus` | C36 + S36 + J36 mixed (**5,400**) → P18 (**900**) → P18 focus (**900**) | Same (**1,800**) | **Controlled prerequisite-order comparison**: Pair starts at the same step and receives the same exposure as staged |
| `mixed_pair_focus` | C36 + S36 + J36 + P18 mixed (**6,300**) → P18 focus (**900**) | Same (**1,800**) | Reviewer's **Mixed → Pair** control: Binding introduced earlier, then emphasized late |
| `mixed` | C36 + S36 + J36 + P36 mixed (**7,200**) | Same (**1,800**) | Standard mixed-training baseline |

The first two curricula both introduce Binding after **5,400 updates**, train on identical Pair questions afterward, and share the final 1,800 updates. Their key difference is whether Color, Shape and Conjunction are taught sequentially or jointly before Binding. This is the cleanest planned comparison for the benefit of *ordering prerequisite tasks*.

`mixed_pair_focus` is a distinct test of the reviewer's suggestion. Its first 18 Pair passes are distributed through earlier mixed learning instead of appearing immediately before Pair focus. It has the same **total Pair exposure** and same last **900 focused-Pair updates** as the two staged variants. Consequently, its contrast includes earlier Binding exposure and its timing; do not attribute all differences to prerequisites alone.

The main `mixed` baseline completes the same 45 passes per question, but has no focused-Pair phase. It tests whether late specialization matters relative to continual mixing.

## 4. Planned comparisons

| Priority | Contrast | Question |
|---|---|---|
| Primary | `staged_pair_focus` − `prereq_mixed_pair_focus` | Does sequencing the prerequisites help, with matched Pair introduction, Pair order and total exposure? |
| Secondary | `staged_pair_focus` − `mixed_pair_focus` | Does full mixed pretraining plus late Binding focus match prerequisite-first training? |
| Secondary | `mixed_pair_focus` − `mixed` | Is late Binding focus helpful relative to continued mixing, under equal total exposure? |
| Diagnostic | `prereq_mixed_pair_focus` − `mixed_pair_focus` | Does showing Binding before the prerequisites are complete change later acquisition? |

No comparison here perfectly separates *all* mechanisms. The first contrast most directly identifies prerequisite ordering under the fixed model and dataset; the remaining contrasts measure effects of the complete schedule changes.

## 5. Evaluation specification

Use `docs/medium_world_evaluation_protocol.md`. Record aggregated evaluation every **100 steps**, including step 0 and every phase boundary (both sides of each transition where possible). Use the same ID-dev and OOD-dev questions and the same model-evaluation code across all runs.

**Primary outcome:** fixed-final step-9,000 **ID-dev Binding accuracy**, compared by paired seed.  
**Secondary outcomes:** fixed-final OOD-dev Binding accuracy, Binding AULC against global optimizer updates, Binding accuracy versus cumulative Pair presentations, task retention and recovery, TPR/TNR, error patterns on each withheld pair, and confusion matrices.

Save predictions for the same `example_id` at evaluation checkpoints in one compressed `.npz` per run. Save the aggregate `schedule_comparison.json` and a phase/exposure manifest. Save model weights locally at phase boundaries and final step if feasible; do not commit model weights to Git.

For fixed schedules, ID-dev is used only to *measure*, not to trigger any stage transition or learning-rate change. OOD-dev is not used to tune the schedule. Keep the historical test-use record explicit if a final test is ever reported.

## 6. Acceptance checks before the five-seed study

1. `python -m valid_tests.test_order_control` passes. On the real 10,000-question dataset, each schedule contains exactly **450,000 presentations**, **9,000 updates**, and **112,500 presentations for each task**.
2. Every question occurs exactly **45 times**; for the same seed, the late Pair-focus block is identical across the first three schedules, as is the final common-mixed block across all four.
3. The same seed gives identical initial weights and per-task question permutations across schedules. Phase boundaries occur at the declared update numbers.
4. Step-0 and every-100-step evaluations produce aggregate metrics and per-question traces; there are no missing `example_id`s or duplicate `(split, example_id, step)` entries.
5. A full seed-42 pilot finishes without NaNs, records the expected transitions, and writes outputs to a **new** directory, not existing completed experiment directories.

## 7. Output locations and study execution

Suggested new notebook: `notebooks/medium_world/mixed_pair_control.ipynb` (training); `notebooks/medium_world/mixed_pair_control_analysis.ipynb` (analysis). Keep the established `PROJECT_ROOT = Path().cwd().parent.parent` convention when running in `notebooks/medium_world/`.

Output directory: `outputs/medium_world/experiments/mixed_pair_control/` with `schedule_comparison.json`, `prediction_traces/`, `phase_exposure.csv`, `final_by_seed.csv`, `paired_differences.csv`, and focused visualizations. Store seed-42 pilot data in `outputs/medium_world/experiments/mixed_pair_control/pilot/` until validation succeeds. Do not mix old training summaries with new traces.

1. Validate the schedule generator and run the seed-42 pilot for all four curricula.
2. Confirm all acceptance checks above **without selecting a curriculum on pilot results**.
3. Train paired seeds 42–46 with the frozen configuration; reuse pilot seed 42 only if it was the complete, unchanged main-protocol run and no settings were edited.
4. Compare paired differences and individual prediction changes; save the report to `docs/mixed_pair_control_findings.md`.
