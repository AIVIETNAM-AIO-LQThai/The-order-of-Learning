# Mixed-to-Binding Controls: What Does Prerequisite Order Actually Contribute?

*November 13, 2026*

## The question behind the comparison

The learning-order hypothesis is not simply that a model trained on Color and Shape before Binding can eventually answer Binding questions. A model trained on all tasks together might achieve the same outcome, and a model trained on prerequisites together might perform just as well as one that learns them in separate stages.

To identify what ordering contributes, the investigation separates three questions:

1. **Prerequisite sequencing:** Does learning Color and Shape before Conjunction improve subsequent same-object Binding compared with learning all three prerequisites jointly?
2. **Timing of Binding exposure:** Must Binding be withheld until the prerequisites have been taught, or can early mixed exposure followed by Binding-focused practice work just as well?
3. **Specialization and retention:** Does concentrated Binding practice produce a lasting gain, or does that gain deteriorate when the model resumes mixed training?

The distinction matters because an apparent advantage for a staged curriculum could come from a late burst of Binding practice rather than from the model having constructed reusable visual concepts in a particular order. The original curriculum comparisons did not isolate these explanations sufficiently.

In this experiment, **Binding** means that the queried Color and Shape occur on the *same* object. **Conjunction** only requires both attributes somewhere in the scene, possibly on different objects. A model can therefore detect both attributes and still fail Binding.

## What the four training histories distinguish

All four schedules use the same frozen Medium World, the same `AttributeBindingViT`, the same per-question exposure budget and a **constant learning rate of $10^{-4}$**. This removes the adaptive-learning-rate confound present in earlier curriculum experiments.

Five paired initialization seeds (42–46) produced **20 completed runs**, each with 9,000 optimizer updates, 450,000 question presentations and 45 presentations of every training question. Evaluations were recorded every 100 updates on the same development questions. The final 1,800 updates use an identical *planned* mixed question sequence for a given seed across schedules.

| Curriculum | First 7,200 updates | Why it is included |
|---|---|---|
| **Staged → Pair focus** | Color + Shape jointly for 3,600 updates; Conjunction for 1,800; Binding only for 1,800 | Tests the proposed prerequisite hierarchy |
| **Prerequisites mixed → Pair focus** | Color + Shape + Conjunction mixed for 5,400; Binding only for 1,800 | Isolates the main *prerequisite sequencing* contrast against Staged |
| **Mixed → Pair focus** | All four tasks mixed for 6,300, with half the main Binding examples distributed through this period; Binding only for 900 | Tests earlier Binding exposure followed by late specialization |
| **Fully Mixed** | All four tasks mixed for 7,200 | Tests continuous interleaving without a Binding-only phase |

Every schedule then receives the same 1,800-update mixed consolidation phase.

The cleanest designed contrast is **Staged versus Prerequisites mixed**. Both first encounter Binding at update 5,400, receive the same Binding question stream afterward, and have identical total exposure. Their difference lies in the preceding organization of Color, Shape and Conjunction, not in the amount of Binding supervision.

The other comparisons are informative but less isolated. Mixed → Pair changes both *when* Binding first appears and the distribution of practice. Fully Mixed has no Binding-only block. Different schedules also reach later phases with different learned weights and AdamW optimizer states; matching examples and learning rates does not make those internal states identical.

**The central experimental question is therefore whether sequencing the prerequisite tasks adds value once Binding introduction, Binding examples and the total training budget have been controlled.**

## 1. Prerequisite-first sequencing did not improve final Binding reliably

The fixed-final results, averaged over five seeds, are:

| Schedule | ID-dev Binding | OOD-dev Binding | ID Binding AULC |
|---|---:|---:|---:|
| Staged → Pair focus | 62.19% | 55.90% | 54.55% |
| Prerequisites mixed → Pair focus | **62.89%** | **57.00%** | 54.93% |
| Mixed → Pair focus | **62.89%** | **57.00%** | 56.06% |
| Fully Mixed | 61.33% | 56.60% | **56.79%** |

*AULC is mean Binding accuracy integrated over global optimizer updates 0–9,000. It measures the performance of the entire training history, including periods when Binding was not yet being trained.*

The prespecified prerequisite-order comparison is:

| Contrast: Staged minus Prerequisites mixed | Mean difference | Paired 95% interval |
|---|---:|---:|
| Final ID Binding | **−0.70 percentage points** | [−5.22, +3.81] |
| Final OOD Binding | **−1.10 pp** | [−3.75, +1.55] |
| ID Binding AULC | **−0.38 pp** | [−1.70, +0.95] |

Staged is better on final ID Binding in three seeds and worse in two. The direction of the mean favors teaching prerequisites jointly, but the uncertainty interval accommodates advantages for either organization.

This does **not** establish that the schedules are equivalent. It establishes something more specific: **the tested separate prerequisite stages do not demonstrate a dependable benefit over joint prerequisite training at the fixed final budget**.

### Why this challenges the original learning-order interpretation

The proposed hierarchy assumes that separately practicing visual attributes, then their independent conjunction, prepares a better internal basis for same-object Binding. Yet the alternative schedule encounters exactly the same prerequisite questions for the same number of presentations without separating the attribute and conjunction phases, and it achieves at least as good a mean final result.

Thus, the evidence does not justify interpreting staged training as constructing a uniquely advantageous conceptual foundation. The organization of prerequisite experience might influence internal representations or individual predictions, but **no lasting performance advantage from that organization was resolved here**.

A further concern is whether the supposed prerequisites were mastered at all. Immediately before Binding introduction (update 5,400), the models averaged:

| Prerequisite performance | Staged | Prerequisites mixed |
|---|---:|---:|
| Shape accuracy | 62.58% | 64.77% |
| Conjunction accuracy | 80.39% | 80.31% |
| Binding before direct training | 50.16% | 50.16% |

Shape remains far from reliable. Moreover, a Color-only rule can score about **79.7%** on the original ID-dev Conjunction questions because of how their negatives are generated. Conjunction accuracy near 80% therefore does not establish that a model has mastered the logical AND of both attribute detections.

These facts weaken a necessary premise of the original hypothesis: **finishing the planned prerequisite stages is not the same as acquiring the intended prerequisite capabilities**. An ineffective prerequisite stage cannot be expected to confer all the benefits predicted for a genuinely mastered one.

## 2. Introducing Binding earlier changes the learning curve, not the final conclusion

Mixed → Pair focus presents some Binding examples throughout the early mixed phase. The two prerequisite-first schedules withhold Binding completely until update 5,400.

This distinction produces visibly different learning histories:

| Schedule | ID Binding at 5,400 | At 6,300 | At 7,200 | Final at 9,000 |
|---|---:|---:|---:|---:|
| Staged → Pair focus | 50.16% | 60.94% | 64.92% | 62.19% |
| Prerequisites mixed → Pair focus | 50.16% | 62.89% | **66.95%** | **62.89%** |
| Mixed → Pair focus | 56.48% | 56.17% | 64.06% | **62.89%** |
| Fully Mixed | 58.05% | 61.17% | 60.70% | 61.33% |

Before update 5,400, the prerequisite-first models have received **no direct Binding questions** and remain near the balanced binary baseline of 50%. Mixed training has already raised Binding accuracy above that level.

Once Binding-only practice starts, however, the initially untrained Binding behavior improves quickly. By update 7,200, Prerequisites mixed has the highest mean Binding accuracy among the four procedures. Concentrating direct Binding supervision can thus produce a substantial intermediate gain even when the model starts learning that task later.

But the two best final values are identical: **62.89%** for Prerequisites mixed and Mixed → Pair focus.

Their paired final difference is exactly **0.00 pp in the five-seed mean**, with a 95% interval of **[−2.76, +2.76] pp**. Staged versus Mixed → Pair is **−0.70 pp** [−3.11, +1.71].

The correct interpretation is not that early Binding exposure is useless or that late introduction is always optimal. Rather:

- **Early exposure** gives the model an opportunity to acquire some Binding behavior sooner.
- **Late concentration** can yield faster improvement over a particular interval and a higher intermediate peak.
- **Neither strategy has demonstrated a lasting final advantage** over the other under these matched total exposures.

### Why learning efficiency depends on the clock

Fully Mixed has the highest global-step Binding AULC (56.79%), while Staged has the lowest (54.55%). This is unsurprising: Fully Mixed receives Binding questions from the beginning; Staged cannot learn that task directly until update 5,400.

Staged also has lower AULC than Mixed → Pair by **1.51 pp** [−2.21, −0.80], with the difference negative in all five seeds. This is a real end-to-end learning-history difference, but not a fair measurement of **efficiency per Binding example**. The schedules differ greatly in prior exposure and initial conditions when Binding appears.

To distinguish overall training efficiency from efficiency of *acquiring Binding after introduction*, one must examine curves against both global updates and cumulative Binding presentations, while retaining the number of prerequisite updates as context. Simply moving the horizontal axis to Binding exposure does not erase the different earlier training histories.

## 3. Binding specialization creates a retention trade-off

The strongest evidence that the schedules differ in more than final accuracy appears at the transition back to mixed training.

At update 7,200, each focused schedule completes its Binding-only phase. At 7,300, it has received only 100 updates of the common mixed consolidation phase.

| Schedule | ID Binding at 7,200 | ID Binding at 7,300 | Change |
|---|---:|---:|---:|
| Staged → Pair focus | 64.92% | 60.16% | −4.77 pp |
| Prerequisites mixed → Pair focus | **66.95%** | 61.95% | **−5.00 pp** |
| Mixed → Pair focus | 64.06% | 62.19% | −1.88 pp |
| Fully Mixed | 60.70% | 61.25% | +0.55 pp |

The drop is accompanied by **question-level disruption**, not just a change in average score. Among ID Binding questions answered correctly at update 7,200, the percentage that become incorrect 100 updates later is:

| Schedule | Previously correct Binding questions becoming incorrect | Binding correctness churn |
|---|---:|---:|
| Staged → Pair focus | 21.74% | 23.52% |
| Prerequisites mixed → Pair focus | 18.68% | 20.16% |
| Mixed → Pair focus | 14.14% | 16.41% |
| Fully Mixed | 9.63% | 12.27% |

The forgetting rate is conditional on a question being correct *before* the transition. Churn is the fraction of questions whose correctness changes in either direction. These quantities measure **prediction stability**, not the destruction of underlying visual features.

The focused schedules had just optimized a task distribution containing almost exclusively Binding; the final phase abruptly returns the other three tasks to the gradient stream. The model's decisions become less favorable to Binding in this immediate period.

At the same time, performance on earlier tasks rebounds. Between updates 7,200 and 9,000:

| Schedule | Change in Shape | Change in Conjunction | Change in Binding |
|---|---:|---:|---:|
| Staged → Pair focus | +9.77 pp | +11.09 pp | −2.73 pp |
| Prerequisites mixed → Pair focus | +9.92 pp | +10.31 pp | −4.06 pp |
| Mixed → Pair focus | +7.11 pp | +3.59 pp | −1.17 pp |
| Fully Mixed | +2.66 pp | +0.16 pp | +0.63 pp |

**This is a specialization–retention trade-off in observable answers.** The mixed consolidation period restores Shape and Conjunction accuracy most strongly where focused Binding training had allowed them to decline. It also tends to reduce the Binding advantage acquired during specialization.

That finding matters for the learning-order hypothesis because the model's latest task performance and the quality of its overall task repertoire are not interchangeable. A curriculum can appear particularly successful immediately after a specialized phase yet lose much of that benefit by the common endpoint.

### What this does *not* prove

It is tempting to call the deterioration "catastrophic forgetting of Shape representations" or "overwriting of Binding features." Those are mechanistic claims the accuracy traces cannot establish.

The same model could retain shape-relevant information in its frozen visual features while changing the query-conditioned evidence it emphasizes or the decision boundaries produced by the shared head. Conversely, some representational change may indeed occur. Without inspecting internal representations, the current observation cannot discriminate these explanations.

The more defensible statement is: **task-distribution transitions alter which questions the model answers correctly, sometimes very quickly, even when the examples and final training budget are matched**.

## 4. Similar Binding accuracy can hide very different answers

A balanced binary task can reach the same accuracy through different mixtures of positive detection and negative rejection.

At the fixed final checkpoint, the ID-dev Binding results decompose as:

| Schedule | Positive recall (TPR) | Negative specificity (TNR) |
|---|---:|---:|
| Staged → Pair focus | 69.53% | **54.84%** |
| Prerequisites mixed → Pair focus | 74.06% | 51.72% |
| Mixed → Pair focus | 75.78% | 50.00% |
| Fully Mixed | **75.94%** | 46.72% |

Staged is less willing to predict that the queried pair exists. It rejects more negatives but also misses more positives. Fully Mixed tends more strongly toward predicting existence.

The same pattern is pronounced on the OOD-dev benchmark:

| Schedule | OOD positive recall | OOD negative specificity |
|---|---:|---:|
| Staged → Pair focus | 59.7% | **52.1%** |
| Prerequisites mixed → Pair focus | 66.4% | 47.6% |
| Mixed → Pair focus | 75.7% | 38.3% |
| Fully Mixed | **77.2%** | 36.0% |

These shifts explain why very similar overall OOD accuracy—roughly 56–57%—can conceal different question-level behavior. A schedule may improve rejection of absent pairs while worsening recognition of present ones.

**Higher negative specificity alone is not evidence of more precise object Binding.** It could arise from a more conservative decision threshold, weaker evidence for positives, or a different visual association mechanism. Accuracy, positive recall and negative specificity must be interpreted together.

The negative Binding questions are deliberately hard in one useful sense: both queried attributes appear somewhere in the image, but not on the same object. A presence-only rule cannot answer them. However, these scenes are not completely controlled for attribute multiplicities or all other scene statistics. Thus, above-chance Binding on the original task does not, by itself, prove that the model compares attributes within individual objects.

### Why OOD accuracy cannot settle compositional transfer

The OOD set has an additional generator-level confound. Every OOD question asks for one of four held-out Color–Shape pairs, which never appear as queried Binding targets in training. More importantly, **OOD-positive scenes contain a held-out pair, whereas OOD-negative scenes are built without any held-out pairs**. The mere presence of *any* held-out combination in scene metadata consequently predicts the OOD label perfectly in this constructed benchmark.

That does not show the ViT exploited this shortcut. It means the evaluation fails to isolate whether a model recognizes the *specific queried combination* when that combination has never appeared during training.

OOD accuracy, including the small differences between these four schedules, should therefore be read as performance on this **confounded constructed distribution**, not clean evidence for or against compositional generalization.

## 5. What the comparisons rule out—and what they leave open

The experiment supports a fairly narrow set of conclusions:

**Separate prerequisite stages are not necessary for the observed final performance.** Jointly training Color, Shape and Conjunction before Binding performs at least as well in the five-seed mean, with matched Binding introduction and total exposure. This weakens the case for a special advantage from the proposed prerequisite sequence.

**A late Binding focus is not sufficient to produce a durable winning curriculum.** Mixed → Pair has a slightly higher mean final ID Binding score than Fully Mixed (+1.56 pp), but the paired interval **[−1.50, +4.63] pp** spans zero. Its final OOD contrast is +0.40 pp [−3.55, +4.35]. Concentrating examples late changes the learning trajectory without demonstrating stable final superiority.

**The training schedules shape the model's decision behavior.** The differences in positive recall, negative rejection, short-term forgetting and restoration of earlier-task performance are scientifically meaningful even when final accuracy differences are small. These effects are not reducible to a ranking of curricula by one number.

**The study does not identify a hierarchy of internal concepts.** The Shape task is still weak when Binding training begins. Conjunction accuracy is compatible with a Color-dominant shortcut. The final outcome cannot tell whether the model acquired reusable Shape representations, how it aggregates object-local evidence, or why the shared predictor changes after task transitions.

**Failure to detect superiority is not evidence that ordering has no effect.** Five seeds, one architecture, one generator and a fixed total budget leave a meaningful range of possible effects unresolved. The question answered is whether *these specific schedules* show an advantage under these controls, not whether every possible visual-learning curriculum is equivalent.

## 6. How this changed the research direction

Before this control, a plausible story was that staged learning succeeds because the model first constructs attributes, combines them logically, and then uses those concepts for object-level Binding.

The matched comparison forces a weaker and more useful interpretation:

1. **Order alone has not earned explanatory priority.** When the timing and amount of Binding training are controlled, separating the prerequisite tasks does not improve the final outcome reliably.
2. **The prerequisite skills themselves require scrutiny.** A scheduled Color/Shape phase is not evidence that Shape is learned well; an approximately 80% Conjunction score may conceal an easier decision rule.
3. **Final accuracy hides behaviorally important instability.** The temporary Binding peak and sharp disruption when Mixed training resumes demand an account of retention and decision changes, not only acquisition.
4. **Association-sensitive behavior must be tested directly.** A balanced hard-negative benchmark helps, but does not exclude all count-based or visual-context shortcuts. The model must be challenged by counterfactuals that preserve marginal attributes and vary *which object possesses which attribute*.
5. **Representation and readout must be investigated separately.** Performance trade-offs across phases do not establish whether Shape information was lost from the visual encoder or merely used differently by query scoring and the final classifier.

These consequences explain why later investigations focused on matched continuation policies, count-controlled Binding, object-preserving swaps, frozen visual probes and task-head analysis. They do **not** retroactively turn those later results into findings of this particular comparison.

## Interpretation

The strongest conclusion is **not that learning order never matters**. It is that the originally proposed prerequisite sequence **does not provide a demonstrated lasting advantage once we compare it with plausible matched-exposure alternatives**.

The model can acquire Binding behavior late, especially under focused practice. But the gain is partly fragile under a return to mixed learning, and different schedules with nearly identical final accuracy can favor very different positive and negative decisions. Combined with weak Shape mastery and a shortcut-prone Conjunction task, this makes it premature to call the learned behavior a successful compositional progression.

The experiment therefore changes the scientific target: before asking *which curriculum teaches compositional reasoning best*, we must establish **what the current model actually learns, which evidence its predictions depend on, and whether those capabilities survive subsequent training**.

---

The numerical results above come from the executed [training notebook](../notebooks/medium_world/mixed_pair_control.ipynb), [analysis notebook](../notebooks/medium_world/mixed_pair_control_analysis.ipynb), and their saved [five-seed outputs](../outputs/medium_world/experiments/mixed_pair_control/five_seeds/). All inferential intervals use five paired training seeds and unadjusted 95% Student-$t$ intervals. Results concern repeatedly inspected development splits; they are not untouched final-test estimates.
