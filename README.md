# The Order of Learning

**Does the order in which a vision model encounters concepts change what it ultimately learns?**

This project studies that question with small, randomly initialized vision transformers trained on synthetic scenes. The intended progression is from detecting visual attributes to reasoning about relationships between attributes. We compare training histories while holding the model, labeled examples and total exposure fixed as far as each experiment permits.

The central scientific issue is not whether a staged schedule achieves a higher accuracy in one run. It is whether **learning prerequisites first produces a reusable capability that survives subsequent learning**, rather than merely shifting when a model practices a task or which answers its final decision rule favors.

## Questions and task semantics

The primary Medium World contains colored objects and four binary questions. For objects $o$ in image $I$:

| Task | Correct answer | Capability being tested |
|---|---|---|
| **Color** $C(I,c)$ | Some object has color $c$ | Detect an attribute |
| **Shape** $S(I,s)$ | Some object has shape $s$ | Detect geometric form |
| **Conjunction** $J(I,c,s)$ | $C(I,c)\land S(I,s)$ | Determine whether *both* attributes occur, possibly on different objects |
| **Binding** $B(I,c,s)$ | Some **single object** has both color $c$ and shape $s$ | Associate attributes with the same object |

Therefore $B(I,c,s)\Rightarrow J(I,c,s)$, but not conversely. A model that detects Color and Shape separately need not have learned Binding.

The earlier **Small World** additionally tests `LEFT`, `ABOVE` and their `XOR` composition, with a separate `CLOSER` adaptation dataset. The later **Medium World** deliberately narrows the question set to Color, Shape, Conjunction and Binding while increasing scene and attribute diversity. Results from the two worlds are not interchangeable.

[Dataset design and what the generators actually implement](docs/dataset_design.md) · [Medium World EDA and benchmark limitations](docs/medium_world_eda_findings.md)

## What the experiments found

### Learning order did not yield a reliable final prerequisite-first advantage

A nine-seed Small World replication reduced the promising single-seed Staged-versus-Mixed difference to **+2.22 percentage points**, with a paired interval **[−6.12, +10.57]**. In the later Medium World matched-exposure comparison, teaching Color/Shape before Conjunction and then Binding finished at **62.19%** ID-dev Binding, versus **62.89%** when the three prerequisites were interleaved before identical Binding training. The paired difference was **−0.70 pp [−5.22, +3.81]**.

These experiments do **not** establish equivalence between curricula. They do not support a dependable final advantage for the specific prerequisite-first schedules and budgets tested.

### The training history matters even when final accuracy is similar

Binding-only practice can raise Binding accuracy while Shape and Conjunction deteriorate. Returning to mixed practice often restores those earlier-task answers but weakens some recently acquired Binding answers. Different schedules with similar overall accuracy also differ in positive recall, negative specificity and question-by-question prediction stability.

This is an observed **specialization–retention trade-off in behavior**, not proof that internal features have been erased.

### Optimization and dataset structure complicate the curriculum story

Across the tested learning-rate policies, Constant LR produced stronger final Binding performance than aggressive global decay. More refined count-controlled analysis found that final discrimination tracks **nominal learning-rate-weighted Binding exposure late in training**, but total optimization opportunity, schedule timing and optimizer state remain confounded. No universal acquisition step has been demonstrated.

Conjunction is particularly misleading: a Color-only rule scores about **79.7%** on the original ID-dev questions. Moreover, the OOD generator creates a scene-level label shortcut: **a held-out combination occurs somewhere in an OOD-dev scene if and only if its question's answer is Yes**. This invalidates an unqualified claim that OOD-dev accuracy measures clean novel-combination transfer.

### The model shows partial association sensitivity, not established object-level reasoning

The subsequent diagnostics identify performance beyond the tested marginal-attribute-count cues: under Constant LR, count-controlled Binding discrimination and count-preserving Color swaps show **partial sensitivity to which attributes share an object**. That behavior is unstable across phases and does not imply explicit object slots.

Oracle-centered, spatially ordered frozen-feature probes recover Shape much better than pooled probes—even from **untrained** encoders. Thus high probe accuracy does not by itself demonstrate that training learned a robust Shape representation. Separately, unchanged Shapes can become less likely to be recognized after certain Color recolorings. The origin and network location of that dependence remain unresolved. Replacing the final eight-feature classifier does not resolve the hardest Conjunction errors.

**Current interpretation:** there is evidence of incomplete association-sensitive behavior, but not a demonstrated internal hierarchy of mastered visual concepts or clean compositional transfer. The model's access to spatial Shape evidence and its task-dependent decisions deserve scrutiny before attributing failures to the curriculum alone.

## Follow the evidence

The reports are organized by scientific question. They retain the controls and numerical results needed to interpret each outcome, rather than duplicating training runbooks.

| Question | Findings |
|---|---|
| How balanced and shortcut-resistant are the generated worlds? | [Small World EDA](docs/small_world_eda_findings.md), [Medium World EDA](docs/medium_world_eda_findings.md) |
| Could the first multimodal architecture learn beyond tiny memorization tests? | [Small World feasibility](docs/small_world_feasibility_findings.md) |
| Does the original sequence replicate? | [Small World nine-seed comparison](docs/nine_seeds_comp_small_world.md), [Medium World five-seed curriculum](docs/five_seed_curriculum_findings.md) |
| Does a matched Binding focus expose a prerequisite-order advantage? | [Mixed → Pair controls](docs/mixed_pair_control_findings.md) |
| Does mastery-guided scheduling change the result? | [Mastery progression](docs/mastery_evaluation_findings.md) |
| What happens when task mixtures change? | [Early staged rehearsal](docs/rehearsal_curriculum_findings.md), [Transition rehearsal](docs/transition_rehearsal_findings.md), [Smoothness × Recency](docs/smoothness_recency_findings.md), [Scaffolded Spiral](docs/scaffolded_spiral_findings.md) |
| How does the optimizer's learning-rate history matter? | [LR policy findings](docs/lr_reset_findings.md) |
| What explains Shape and Binding failures? | [Representation diagnostics](docs/representation_diagnostics_findings.md), [Diagnostic follow-up](docs/diagnostic_followup_findings.md) |

[Evaluation methods and interpretation rules](docs/evaluation_methods.md) explain why fixed-final endpoints, seed pairing, class balance, question-level interventions and test-split boundaries matter.

## Code, data and reproduction

- `src/small_visual_world/` and `src/medium_visual_world/` implement the scene generators, query grammars, symbolic oracles, models and training utilities.
- `configs/` contains the frozen generation settings.
- `notebooks/small_world/` and `notebooks/medium_world/` contain training and analysis notebooks. A notebook may include saved outputs from an executed run; its code and the corresponding saved experiment data should be read together.
- `outputs/` contains dataset manifests, experiment histories, traces and derived summaries. The presence of multiple saved evaluations of the same scene does **not** create independent samples.
- `valid_tests/` contains programmatic checks of generators and training contracts.

From the repository root, the frozen dataset manifests can be checked without regenerating them:

```powershell
python -m scripts.generate_dataset --world small --verify-existing
python -m scripts.generate_dataset --world medium --verify-existing
```

New model training is not required to inspect the saved findings. Full replay depends on the corresponding notebooks, their environment and, for some representation diagnostics, locally stored checkpoints and helper outputs that are not all committed as ordinary Git files. Reproducing a result means matching its **dataset, model, training history, evaluation examples and statistical unit**, not merely executing a notebook with a similar title.

## Boundaries of the conclusions

The reported comparisons use small synthetic worlds, a compact ViT and limited paired seeds. Development examples have been inspected repeatedly; exploratory comparisons and their unadjusted intervals should not be treated as confirmatory final-test evidence. Nominally identical training procedures have also exhibited small numerical divergences, so differences of only a few percentage points require reproducibility checks.

Neither a null final curriculum contrast nor a successful diagnostic readout establishes how a general vision model—or a human—must learn. The contribution is an increasingly precise account of **which effects survive controlled comparisons, which apparent successes rely on shortcuts, and which mechanisms remain unidentified**.
