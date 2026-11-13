# Evaluation methods: from benchmark scores to interpretable learning evidence

This document defines how to compare learning histories without conflating final accuracy, speed of acquisition, retention, question shortcuts or representation mechanisms. The definitions apply to the implemented Small and Medium World studies; **each study's actual budget and question distribution take precedence over a single assumed universal setup**.

## 1. Identify the outcome before comparing curricula

A curriculum intervention changes when examples enter training. Its effect depends on *what* and *when* we measure:

- **Fixed-final performance:** ability to answer a frozen development set after the same predeclared total training budget. This is the primary comparison when asking whether an advantage persists.
- **Training trajectory:** how performance changes during learning, including transitions, specialization and subsequent mixed practice.
- **Same-exposure comparison:** accuracy after equal cumulative presentations of the target task; this answers a different question from equal global optimizer updates.
- **Error structure:** positive recall, negative rejection, confidence, logical consistency and performance on controlled subgroups.
- **Mechanistic evidence:** whether controlled interventions change decisions, and whether frozen features contain information accessible to an independently fitted readout.

The benchmark label itself is not synonymous with a learned concept. The dataset audit determines whether alternative rules can reach high accuracy without performing the intended reasoning.

For the main long-budget Medium World studies, the final checkpoint is **9,000 updates** with **45 presentations of every training question**. Earlier experiments used shorter budgets (for example 3,000 updates), and must retain their own declared endpoints. Do not retrospectively select the checkpoint where one curriculum looked best and call it the primary final comparison.

## 2. Accuracy, class balance and discrimination

For a binary question set, let $TP,FN,TN,FP$ denote the confusion counts and $N$ their sum. Then

$$
\operatorname{Accuracy}=\frac{TP+TN}{N},\qquad
\operatorname{TPR}=\frac{TP}{TP+FN},\qquad
\operatorname{TNR}=\frac{TN}{TN+FP}.
$$

Positive recall/TPR asks whether present pairs are detected; negative specificity/TNR asks whether absent pairs are rejected. Their mean is **balanced accuracy**:

$$
\operatorname{BA}=\frac{\operatorname{TPR}+\operatorname{TNR}}{2}.
$$

If both labels are present, BA gives an always-one-class predictor **50%**, even on an imbalanced subset. Raw accuracy instead depends on that subset's prevalence.

This matters for **count-free Binding**: the 109-question ID-dev diagnostic contains 34 Yes and 75 No answers. Always predicting No obtains **68.81% raw accuracy but 50% BA**. A model with lower raw accuracy can still distinguish classes above chance by BA. Report the support and the full confusion matrix; do not compare raw and balanced scores as though they had the same baseline.

Additional metrics have different roles:

| Measure | Answers | Important limitation |
|---|---|---|
| Predicted-positive rate $(TP+FP)/N$ | Is the decision rule biased toward Yes? | Not correctness |
| Binary cross-entropy / log loss | Are probabilities compatible with the labels? | Sensitive to overconfident errors and class prevalence |
| ROC AUC | Do scores rank positives above negatives? | Threshold-free but not a causal mechanism and may conceal class-specific utility |
| PR AUC | Is positive retrieval useful at the observed prevalence? | Changes when prevalence changes |
| Balanced accuracy | Is classification above the two-class majority shortcut? | Depends on a chosen threshold; prevalence shift can affect calibration |
| Per-shape / per-pair support and confusion | Which geometries or attribute combinations fail? | Small subgroups may have large uncertainty |

**Do not call a model calibrated merely because it was trained with balanced labels.** Calibration requires its own probability-versus-frequency assessment, and any selected subgroup can have a different prevalence from training.

## 3. Acquisition, specialization, forgetting and recovery

For evaluation accuracies $a(t)$ at global update $t$, define normalized **global-step area under the learning curve** over the observed grid $0=t_0<\cdots<t_K=T$:

$$
\operatorname{AULC}_{\mathrm{step}}
=\frac{1}{T}\sum_{i=0}^{K-1}\frac{a(t_i)+a(t_{i+1})}{2}(t_{i+1}-t_i).
$$

AULC rewards early ability. A curriculum that postpones Binding until late training will tend to have lower global-step Binding AULC even if it learns rapidly after introduction. This is a legitimate **end-to-end training-procedure** result, not necessarily evidence of worse **efficiency per Binding example**.

Also plot accuracy against cumulative **Binding question presentations**, with the other task exposures and absolute update count visible. Equal Binding exposure does not imply equal pretraining, optimizer states or time of evaluation.

Aggregate accuracy may conceal very different changes to individual questions. For the *same IDs* observed before and after a phase:

- **Retained correct:** correct at both checkpoints.
- **Forgotten:** correct before, incorrect after.
- **Newly correct:** incorrect before, correct after.
- **Still incorrect:** incorrect at both checkpoints.

With counts $R,F,L,U$ for these four groups, define

$$
\operatorname{ForgettingRate}=\frac{F}{R+F},\qquad
\operatorname{NewlyLearnedRate}=\frac{L}{L+U},\qquad
\operatorname{Churn}=\frac{F+L}{R+F+L+U}.
$$

A three-checkpoint **recovery rate** conditions on questions correct at the starting checkpoint, incorrect in the middle, and correct again at the final checkpoint:

$$
\operatorname{RecoveryRate}=
\frac{\#\{\text{correct}\to\text{wrong}\to\text{correct}\}}
{\#\{\text{correct}\to\text{wrong at middle}\}}.
$$

These quantities must be left undefined when their denominator is zero, not silently set to zero. They measure **prediction transitions**, not the erasure, creation or recovery of latent features. A question's answer can flip because of confidence, query scoring, shared-head decisions or changed representations.

An additional **worst observed accuracy decline** measures the largest fall below the accuracy at a declared stage-start checkpoint over later checkpoints in that same stage. State the stage and reference checkpoint: it is a descriptive maximum over sampled evaluation times, and its size depends on the checkpoint grid.

The immediate 100-update window after a transition answers a different question from the entire final consolidation period. Report both, rather than allowing eventual recovery to hide short-term disruption.

## 4. Keep the causal comparison aligned with the intervention

**Equal final exposure** can be controlled by giving every training question the same total number of presentations. **Matched task streams** additionally preserve the per-task example order under the same seed. **Matched late blocks** preserve the intended final question sequence. These are valuable controls, but do not establish equal gradients or equal realized model state when the phase starts.

In a strict parent-fork continuation, each branch restores one shared **model state, optimizer state and relevant RNG state** at a specified checkpoint. This rules out a pre-branch history difference for that realized parent. It does not guarantee identical numerical continuation under different training inputs or under nondeterministic operations.

Likewise, a learning-rate schedule is part of the treatment. If adaptive LR follows validation loss, changing task order can also change the **realized LR trajectory**. A schedule comparison with such an adaptive optimizer is a comparison of **whole procedures**, not isolated ordering.

For learning-rate diagnostics, a nominal exposure quantity such as

$$
E_{\mathrm{Binding}}=\sum_{t=1}^{T}\eta_t\, n_{\mathrm{Binding},t}
$$

weights the number of Binding presentations in update $t$ by the logged LR $\eta_t$. It is useful for describing optimization opportunity, but **not an effective gradient norm, causal learning budget or measure of information acquired**. Optimizer moments, gradient directions, stage timing and state maturity remain confounded.

Ask at every comparison: *Does the contrast isolate prerequisite order, late task timing, a full continuation policy, or an LR policy?* The same final accuracy difference has a different interpretation under each design.

## 5. Diagnostic questions must test the claimed ability

### Independent Conjunction versus Binding

In Medium World, positive Conjunction only requires both attributes somewhere; Binding requires them **on the same object**. A Color-only Conjunction oracle rule achieves approximately **79.7%** ID-dev accuracy. Report Conjunction negatives separately for **Color missing / Shape present** and **Color present / Shape missing**, because high aggregate accuracy can hide failure to check Shape.

Original Binding negatives contain both independent attributes, so a binary presence-only heuristic is at chance on that balanced task. However, differences in scene counts and attribute multiplicities may still predict its label. Stratify count-based diagnostics and include explicitly calculated metadata-only heuristics before claiming association-sensitive behavior.

### Controlled association edits

For a validated counterfactual $(I,I')$ in which the queried Binding answer flips while selected nuisance factors are fixed, distinguish:

1. **Both-correct:** the original prediction is correct and the edited prediction is correct.
2. **Correct direction:** the change in continuous Yes probability or logit margin has the intended sign.
3. **Binary flip:** the predicted class changes, irrespective of correctness.
4. **Invariant task:** an answer whose oracle label is unchanged under the edit also remains stable.

Both-correct is a demanding functional test but has a **floor** when the original prediction is already almost always wrong. Correct direction uses more of the score information but is not equivalent to correct classification. Report the pre-edit accuracy, number of valid edits, number of original scenes, target Shape, and intervention success/rejection counts.

An edit that preserves Color and Shape multisets can rule out specific **global counting** explanations. It does not prove a model has symbolic object slots. A response to local co-location may be a functional association detector without a discrete object representation.

### Invariance, confidence and attribute dependence

A Color-only edit that leaves the queried Shape present is a logical **Shape-label invariant**. If the predicted Shape class changes, that is an operational invariance failure. But comparing the Shape-flip rate with a Color-flip rate is not a clean measure of asymmetry unless their original confidence distributions are considered.

Confidence matching can explain part of an asymmetry in **binary flips**, especially when one task has many low-margin predictions. It does not make a change in the underlying continuous logit disappear. Use before/after **paired margins**, matched original scenes, consistent edit magnitudes and seed-wise uncertainty. If comparing a held-out recolor with an allowed recolor, keep the **same original scene and target object** whenever possible. Record shape multiplicity and target Color.

### Feature probes are not deployed model performance

A frozen-feature probe fits a **new supervised decoder** without updating the original vision model. When the probe is given an oracle object center, it solves an *object-centered* recognition task whose localization problem the deployed model still faces. A successful linear probe establishes accessibility **under that decoder and its supervision**, not how the original model used the feature or whether visual pretraining created it.

Compare with untrained random encoders and shuffled-label controls. Preserve ordered spatial features versus pooled features under matched object crops to test information loss. A trained-versus-random difference depends on the probe family; a high spatial-probe score from both encoders is not proof of learned Shape abstraction.

Network Dissection-style single-channel IoU and SpRAy-inspired gradient clustering are descriptive diagnostics. Weak channel alignment does not imply absence of distributed semantic information; weak explanation-cluster stability does not establish one universal attribution mechanism.

## 6. Statistical units and uncertainty

The main replication unit for matched curriculum and model comparisons is the **training seed**. With five seed-wise differences $d_1,\dots,d_5$, a conventional paired 95% Student-$t$ interval for their mean is

$$
\overline d\;\pm\;t_{0.975,4}\frac{s_d}{\sqrt5}.
$$

Report the mean difference in **percentage points** when comparing accuracies, the between-seed sample SD, the interval, and how many seed differences have each sign. Five seeds yield limited precision; a confidence interval crossing zero is **not an equivalence test**.

The same development questions are reused across seeds and checkpoints. The thousands of predictions created by a Cartesian scene–query grid are *not* thousands of independent models or scenes. Do not use a naive question-level bootstrap as a substitute for seed replication. A secondary uncertainty analysis can resample **scene IDs as clusters**, preserving all derived questions/edits from each scene. When averaging two curricula per seed, average **within each seed first**, and still use five independent seed units.

Many analyses are exploratory and include multiple curricula, checkpoints, subgroups and metrics. Label unadjusted 95% intervals as such; do not silently convert one selected significant contrast into a confirmatory claim. For support-poor groups, provide descriptive counts before interpreting percentages.

### Reproducibility discrepancies

Some nominally identical procedures in the archived studies generated slightly different predictions. Approximate numerical reproducibility is not the same as exact trajectory identity. Before interpreting differences of only 1–2 points, compare the real example stream, initialization, optimizer and RNG states, augmentation draws, LR schedule and runtime settings. A shared checkpoint removes one class of pre-split differences; it cannot guarantee identical future floating-point execution.

## 7. ID, OOD and final-test boundaries

The **original ID-dev** has balanced questions from the same generation regime as training. The **original OOD-dev** asks only held-out Binding targets and has different scene composition. More seriously, its generator makes the presence of *any* held-out Color–Shape object perfectly coincide with its label (400/400 by scene metadata). Therefore OOD-dev accuracy **cannot by itself identify systematic transfer to the specific queried unseen combination**.

The expanded same-scene logical grid has still another class balance. For instance, its ID-dev Binding grid contains **1,835 Yes and 8,405 No** answers; an always-No classifier obtains **82.08% raw accuracy**. Never compare that raw score to the original balanced Binding benchmark without conditioning on the altered label distribution. Report class-balanced metrics and within-scene logical relations separately.

The development splits have been inspected repeatedly. Their results support exploratory research and hypothesis refinement, not fresh held-out estimates. A final-test claim would require verifying the historical test-access record and fixing the evaluation plan; the existence of a named test split does **not** certify that it was never examined.

## 8. Preserve enough evidence to reconstruct an interpretation

For each completed study, the essential saved evidence consists of:

- A fixed **data identity** (manifest hashes, exact question IDs and split membership), and actual model/optimizer/training settings.
- The **realized training treatment** (schedule phases, cumulative per-task question presentations, LR history and transition checkpoints).
- **Predictions** on exact IDs across checkpoints, including labels, probability/score margins, and sufficient scene metadata to compute error strata.
- **Seed-level summaries and paired contrasts**, with denominators, uncertainty and data limitations.
- A concise findings analysis distinguishing **observed outcomes**, **ruled-out alternatives** and **open mechanisms**.

Files and code should be linked where they enable verification—not presented as ceremonial `Project`, `Branch`, `Status` or `Predecessor` fields. The goal is to make every scientific claim traceable to a comparison that actually tests it.

[Dataset design](dataset_design.md) · [Medium World dataset audit](medium_world_eda_findings.md)
