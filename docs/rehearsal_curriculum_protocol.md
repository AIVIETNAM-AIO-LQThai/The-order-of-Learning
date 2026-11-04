# The Order of Learning: staged rehearsal protocol (v1)

## Question

With the same 64×64 medium-world dataset and unchanged `AttributeBindingViT`, does
rehearsal during later stages improve retention, learning efficiency, and final
same-object binding accuracy? Compare against the original mixed, staged, and
reverse schedules under a *shared learning-rate trajectory*.

## Controlled factors

- Model: `AttributeBindingViT(dim=48, patch_size=4)` (unchanged)
- Seed-matched initial weights; same optimizer settings, augmentation and dataset
- 10,000 training examples; four tasks, 2,500 questions per task
- Batch size 50; **3,000 optimizer steps per schedule**
- Each exact training example appears **15 times**, of which 12 are in the main
  training phase and 3 in the common mixed phase
- **Exactly 37,500 example presentations per task per schedule**, including
  rehearsal; rehearsal replaces earlier-stage exposure rather than adding it
- Final metrics at step 3,000; ID-dev pair AULC from steps 0–3,000; OOD-dev
  binding accuracy is secondary; ID-test and OOD-test remain untouched

## New schedule: `staged_rehearsal`

Each row gives the number of *complete passes through the examples of that task*
allocated to the phase. For example, `color=1` means every color example
appears once in that phase. Within each phase, question examples are interleaved
at the sample level, not as separate task blocks.

| Phase | Color | Shape | Conjunction | Pair | Updates |
|---|---:|---:|---:|---:|---:|
| Attributes | 10 | 10 | 0 | 0 | 1,000 |
| Conjunction + earlier rehearsal | 1 | 1 | 11 | 0 | 650 |
| Binding + earlier rehearsal | 1 | 1 | 1 | 12 | 750 |
| Common mixed consolidation | 3 | 3 | 3 | 3 | 600 |
| **Total** | **15** | **15** | **15** | **15** | **3,000** |

During the conjunction phase, rehearsal represents 2/13 (~15.4%) of questions.
During the binding phase, rehearsal represents 3/15 (20.0%). No pair questions
are introduced before the binding stage.

Phase lengths change relative to pure staged: this is a meaningful aspect of
redistributing the same exposure budget, not an isolated causal estimate of
"memory replay alone". The primary experimental intervention is the **whole
rehearsal-based curriculum**.

## LR comparison

Original five-seed mixed runs (42–46) using ReduceLROnPlateau recorded **no LR
reductions**. Run with `--lr-policy mixed_reference --lr-reference <old_json>`
to replay each corresponding mixed run's LR trajectory for *all* schedules.
For this dataset and these seeds the replay is a constant **1e-4** throughout
3,000 steps. Thus LR cannot explain differences among new schedules.

The existing `--lr-policy plateau` still exists for exploratory training but
must **not** be used for the primary schedule comparison; it adapts LR
independently according to schedule-dependent validation losses.

## Study comparisons

1. `staged_rehearsal` versus `staged`: effect of replacing pure-stage exposure
   with interleaved later-stage rehearsal at an identical exposure budget.
2. `staged_rehearsal` versus `mixed`: whether the proposed curriculum improves
   final ID binding and binding learning-curve area.
3. `staged` versus `reverse`: check order effects under the shared LR policy.
4. Inspect color, shape, conjunction, and pair accuracy immediately before and
   after phase transitions, and after common consolidation.

Analyze *paired seed differences*, not just pooled best accuracy. Five seeds
provide exploratory evidence; avoid claiming significance without uncertainty
intervals. OOD-dev guides analysis only; do not tune against ID-test/OOD-test.

## Run commands (repository root)

```powershell
python -m pytest valid_tests/test_rehearsal_curriculum.py valid_tests/test_plateau_curriculum.py -q
python -m experiments.medium_world.compare_learning_schedules --device cuda --smoke --seeds 42 --schedules mixed staged staged_rehearsal reverse --lr-policy frozen --out outputs/medium_world/experiments/rehearsal_smoke
python -m experiments.medium_world.compare_learning_schedules --device cuda --seeds 42 --schedules mixed staged staged_rehearsal reverse --lr-policy mixed_reference --lr-reference outputs/medium_world/experiments/plateau_five_seeds/schedule_comparison.json --save-models --out outputs/medium_world/experiments/rehearsal_seed42
python -m experiments.medium_world.compare_learning_schedules --device cuda --seeds 42 43 44 45 46 --schedules mixed staged staged_rehearsal reverse --lr-policy mixed_reference --lr-reference outputs/medium_world/experiments/plateau_five_seeds/schedule_comparison.json --save-models --out outputs/medium_world/experiments/rehearsal_five_seeds
```

Remember: `--smoke` uses a tiny 16-dimensional model with patches of size 8 and
runs only two minibatches per phase; it verifies execution, *not performance*.

### Expected output

`.../rehearsal_five_seeds/schedule_comparison.json`, containing:
- Matched exposure plans, including `phase_task_presentations` for rehearsal
- Global ID-dev/OOD-dev curves (per task), each schedule's LR trajectory
- Best overall ID-dev CE and final per-task results
- Binding area under learning curve, phase-boundary observations

**Do not commit `.pt` checkpoints to Git.**
