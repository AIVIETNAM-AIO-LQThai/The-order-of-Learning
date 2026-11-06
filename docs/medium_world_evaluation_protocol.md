# Medium World: evaluation and logging protocol

**Applies to:** Subsequent fixed-schedule experiments using the existing Medium World and `AttributeBindingViT`.  
**Principle:** Distinguish task acquisition, retention, and transfer; evaluate the same questions at multiple checkpoints.

## 1. Data and evaluation checkpoints

- Train: 10,000 questions (2,500/task); ID-dev: 1,024 questions (256/task); OOD-dev: 400 held-out Pair questions.
- Evaluation checkpoints: step 0, every 100 updates, each stage boundary, and the fixed final step. The same examples and IDs are evaluated across all schedules and seeds.
- Evaluate all tasks on ID-dev and Binding on OOD-dev. Do not replace fixed-final results with the best observed checkpoint.
- For the frozen curriculum study, no evaluation result changes the phase plan or constant learning rate.

## 2. Outcome metrics

| Outcome                       | Definition                                                                                                              | Interpretation                                                                                        |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Final ID Binding              | Correct Pair predictions / 256 at step 9,000                                                                            | Main performance outcome on familiar permitted pairs                                                  |
| Final OOD Binding             | Correct Pair predictions / 400 at step 9,000                                                                            | Transfer to withheld color–shape combinations                                                         |
| Global-step Binding AULC      | Trapezoidal mean of Binding accuracy over steps 0–9,000                                                                 | Efficiency of the complete training procedure, including prerequisite training time                   |
| Pair-exposure curve           | Binding accuracy plotted against cumulative _Pair question presentations_                                               | How efficiently Pair performance develops per Pair example seen; report earlier non-Pair exposure too |
| Positive recall (TPR)         | TP/(TP+FN)                                                                                                              | Recognizing present queried pairs                                                                     |
| Negative specificity (TNR)    | TN/(TN+FP)                                                                                                              | Rejecting nonexistent pairs                                                                           |
| Predicted-positive rate       | (TP+FP)/N                                                                                                               | Detecting a Yes/No prediction bias                                                                    |
| Cross-entropy by task         | Mean CE across examples of each task                                                                                    | Prediction confidence and error magnitude                                                             |
| Worst accuracy decline        | Largest decrease from accuracy at the chosen start of a stage to any later evaluation in that stage (percentage points) | Aggregate skill interference during a specific stage                                                  |
| Individual forgetting rate    | `correct_before & incorrect_after` / `correct_before`, on _the same IDs_                                                | Previously correct questions that become incorrect                                                    |
| Individual newly-learned rate | `incorrect_before & correct_after` / `incorrect_before`, on the same IDs                                                | Previously incorrect questions that become correct                                                    |
| Correctness churn             | (`forgotten` + `newly learned`) / N                                                                                     | Prediction correctness instability hidden by aggregate accuracy                                       |
| Recovery rate                 | `correct_before & incorrect_mid & correct_final` / (`correct_before & incorrect_mid`)                                   | Fraction of questions forgotten at the middle checkpoint that are correct at final                    |

Never label a single incorrect→correct switch proof of a stable learned concept. Report recovered examples separately from newly correct examples.

## 3. Per-example predictions

Store one compressed trace per seed and curriculum, with fields equivalent to:

```
seed, schedule, step, phase, split, example_id, task,
true_label, predicted_label, probability_yes
```

Use the existing dataset's `example_id` to join scene metadata without duplicating scenes in the trace. Avoid synthetic examples in results. Keep a single common trace schema across all curricula.

To interpret errors, derive the confusion matrix, per-label cross-entropy and subgroup performance at each checkpoint. For each Pair example, join the queried color/shape, number of objects, repeated colors/shapes, and whether its combination is withheld. Evaluate Conjunction negatives separately when color is missing versus shape is missing.

The main OOD subgroup table should report **support (n), TP, FN, TN, FP and accuracy**, not accuracy alone. This prevents class imbalance within a subgroup from being confused with genuine Pair recognition.

## 4. Stage-level records

Every run should contain:

- Dataset hashes, model configuration, train seed, optimizer/augmentation, device, schedule name.
- Exact stage names, start/end steps, total and cumulative presentations by task at evaluation checkpoints.
- Learning rate and mean training loss since the previous evaluation.
- Any stage transition event, including actual step and reason (for adaptive schedules).

For the new fixed-order controls, fixed transition times make the comparisons simpler; do not introduce mastery criteria or plateau LR adjustment in the same experiment.

## 5. Statistical comparisons

Compare paired seed differences for 42–46, keeping _seed_ as the unit of replication. Report per-seed differences, group mean, sample SD, paired 95% interval and number of positive differences. Keep ID and OOD outcomes separate.

At five seeds, intervals are descriptive estimates; use consistent definitions rather than searching across many metrics for the largest advantage. If reporting subgroup comparisons, include sample counts and consider scene-level dependence when estimating precision. Preserve old experiment results and model settings unchanged for historical comparison.

## 6. Files to retain

- `schedule_comparison.json` (config, hashes, full aggregate histories)
- `phase_exposure.csv` and `final_by_seed.csv`
- One `.npz` prediction trace per `(seed, schedule)` and `prediction_error_subgroups.csv`
- `paired_differences.csv` and a concise Markdown interpretation

All derived CSVs should be generated from the _same_ `schedule_comparison.json` and trace files. Do not mix an old run summary with new-run traces.
