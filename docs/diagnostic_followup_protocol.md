# Diagnostic protocol — count-controlled Binding, swaps, and frozen readouts

## 0. Research boundary

We are not yet redesigning `AttributeBindingViT`. The present goal is to locate the most plausible bottleneck among (1) Shape-feature accessibility, (2) the eight-scalar shared readout, (3) same-object association, and (4) optimization timing / interference. Experiments answer **behavioral and decoding questions**, not a literal mechanistic partition of failure. The original model's four-query-type training is not modified.

The evaluation inputs are development data repeatedly used during past research; all outcomes are **exploratory**. No retrospective ranking should be presented as an untouched test.

## 1. Count-controlled learning curves — no checkpoints

A Pair query $(I,c,s)$ is count-controlled if

$$
\sum_o \mathbf{1}[\mathrm{Color}(o)=c]=1,\qquad
\sum_o \mathbf{1}[\mathrm{Shape}(o)=s]=1.
$$

Thus detecting how *many* target-Color or target-Shape objects are present cannot alone tell whether the two attributes belong to the same object. This is a negative control against **multiplicity**, not a fully shortcut-free distribution. Scene size, distractors, background, object geometry, and label-dependent sample construction may still differ. Record sample counts and label prevalence separately for ID-dev and OOD-dev.

For each training run, load the original `prediction_tracking.py` NPZ schema. Join saved question IDs and labels **exactly** with `id_dev.jsonl` or `ood_dev.jsonl`; abort for missing or inconsistent Pair IDs. Compute

$$
\mathrm{BA}(t)=\frac{1}{2}\left(\frac{TP_t}{TP_t+FN_t}+\frac{TN_t}{TN_t+FP_t}\right).
$$

Also report support, Yes/No counts, accuracy, positive recall, specificity, and predicted-Yes rate; a 50% BA baseline is appropriate when both labels exist. The broader original Pair set is a comparator, not the primary outcome. Use all logged steps (normally every 100 updates) and keep phase boundaries visible when interpreting figures. Contrast Constant LR with Global Cosine, Reset Cosine, Global Plateau and Reset Plateau within a fixed curriculum, not by pooled across-curriculum differences alone.

**Primary phase effects for Prereq-Mixed → Pair:** $5{,}400\to7{,}200$ (focused Binding acquisition), $7{,}200\to9{,}000$ (common consolidation). Compare trajectories with identical query IDs across steps. Calculate five **seed-wise differences** first, then summarize their mean and an exploratory paired Student-$t$ interval. Small $n$ and repeated outcome inspection warrant cautious language.

**Interpretation matrix:**

| Observed finding | Supports | Does not prove |
|---|---|---|
| Count-free BA rises after Binding starts | Association-sensitive predictions improve beyond simple counts | Internal object slots or causal prerequisite benefits |
| Constant LR rises, Global Cosine does not | Optimization timing influences the learned behavior in this experiment | Every schedule with LR decay fails in general |
| BA falls when all four tasks resume | Prediction-level interference/retention problem | Learned features were erased |
| All policies converge at final | End-state may be insensitive to the tested schedule differences | Equal representations or equal sample efficiency |
| OOD count-free BA near chance | No demonstrated performance on that filtered distribution | Clean proof of failure to compose; OOD sample construction is confounded |

## 2. Binding swaps with both-correct and directional metrics — checkpoints required

Take the **same count-controlled ID-dev Pair question**. Determine which object has the unique target Color and which has the unique target Shape. Swap Color assignments so that the queried Binding answer flips, while the object count, Color multiset, Shape multiset, radius, positions and rotation remain unchanged.

For positive Pair questions, swap the target Color with an object of a different Shape and Color. For negative Pair questions, swap the Color of the sole target-Color object with the sole target-Shape object. Verify the oracle before and after using metadata. Both images are rendered afresh with the **same renderer**, avoiding different image paths or image-processing code.

For each edited pair save: question/scene IDs; seed/schedule/step; queried attributes; original/new labels; predicted labels and $P(\mathrm{Yes})$; both-correct; direction-correct; and before/after accuracy. Direction-correct means the signed probability change agrees with the oracle label flip; it can succeed when neither discrete prediction is correct, so report it **alongside** both-correct, not as its replacement. Split positive→negative and negative→positive edits to reveal Yes/No bias. If several edits per original scene are later supported, group uncertainty by scene rather than treating edits as independent.

**Interpretation matrix:**

| Curve + swap outcome | Most plausible explanation to investigate |
|---|---|
| BA and swap both rise during Pair focus | Association-sensitive behavior is acquired; capacity exists at least for part of the task |
| BA rises, swap stays flat | Natural-scene shortcut or poor edit robustness may explain BA |
| Swap rises, then falls after consolidation | Predictions using pair association are hard to retain under mixed-task optimization |
| BA and swap remain low despite good Shape decoding | Association/readout/localization more suspicious than raw information availability |

The swap does not identify which internal layer stores Binding. A complex model may respond to side effects of the edit rather than instantiate a symbolic same-object variable.

## 3. Frozen Shape probes — checkpoints required

### 3.1 Exact feature definition

The production model forms `patches = Conv2d(image, kernel=4,stride=4)` with 256 positions and dimension 48. Its head consumes

$$
V=\operatorname{score\_norm}\!\left(\operatorname{pre\_norm}(P)+0.5\operatorname{post\_norm}(\mathrm{Transformer}(P+\mathrm{pos}))\right).
$$

The object-level probe uses a **ground-truth center** to index a 5×5 grid of these frozen features. This is privileged localization that does not exist in the original model. Do not compare absolute object-level four-way Shape accuracies directly to scene-level Yes/No Shape accuracy.

### 3.2 Capacity / geometry checks

Train on objects from the train split; evaluate on objects from ID-dev. Fit decoders on the identical labeled objects:

- Mean pooled 48-dim ROI + Ridge classifier.
- Flattened 5×5×48 = 1,200 features + Ridge classifier.
- Randomly position-shuffled 25 patch vectors (still 1,200 features) + Ridge classifier.
- Shared per-patch 48→4 score, max/mean aggregation, 8→32→4 MLP, trained with object labels.

The 1,200- vs 48-dimensional contrast is **confounded by capacity**. The 1,200-vs-1,200 position-shuffle contrast is more directly informative about *stable spatial arrangement* (though distribution shift and optimization remain possible explanations). The randomly initialized ViT is a pixel-derived random feature map, **not a label-free Shape classifier**. Only one random network per seed should be treated as a distinct random baseline; never treat copies across checkpoints as extra independent replicates.

Trained and random encoders are separately frozen. All external decoders are separately trained with the *same Shape labels*; their performance measures information accessibility under that decoder, not what the original VQA head currently uses.

### 3.3 Same-task scene-level readouts

To reduce the object-vs-scene mismatch, compare on the **original scene-level binary Shape questions** from train and ID-dev:

- Unmodified frozen model's original Yes/No predictions.
- Logistic regression trained solely on the **exact eight original scalar features** with the encoder and query machinery frozen.
- A learned 3×3 spatial-convolutional Shape readout over 16×16×48 frozen features, query-selected by the requested Shape, followed by max/mean pooling and an MLP.

Here the sample labels are the *same* Yes/No Shape task. Both external readouts are newly supervised. The Conv readout has more capacity and a new inductive bias, so an advantage localizes a **decoding opportunity** rather than proving which component of the original model was broken. Do not tune training epochs or hyperparameters using ID-dev.

### 3.4 Shared-head competition

Freeze the model through its eight scalars, then compare the pretrained original MLP with (a) a newly fitted shared MLP for all four tasks and (b) four newly fitted separate MLPs. This reuses identical features and train/dev splits. It does **not** match parameter counts: the second model has roughly four times as many head parameters. A performance gain can be due to specialization, extra capacity, or optimization; this is a preliminary diagnostic only.

## 4. Predictions this package can falsify

1. **Optimization-limited Binding:** If under identical constant LR and count-controlled queries Binding never exceeds 50% BA, then earlier accuracy jumps were mostly distribution shortcuts; if it rises in the focused phase, an absolute capacity claim is unsupported.
2. **Spatial readout bottleneck:** If a more expressive scene-level readout reliably decodes Shape from the frozen encoder while the original eight-feature head fails, then the encoder contains accessible but underused Shape information in the tested sense.
3. **Early feature bottleneck:** If all heads, including oracle-centered spatial probes, struggle and performance varies strongly with size, investigate better low-level geometry features. Null results cannot establish information absence in an absolute sense.
4. **Head interference hypothesis:** If separate task-specific heads outperform a shared one on frozen eight-feature inputs, follow with a parameter-matched ablation before attributing the advantage to gradient interference.
5. **Attribute association hypothesis:** If Binding-swap both-correct rises in the focused stage and falls in consolidation, the representation is capable of association-sensitive predictions that are not robustly retained under multitask training; this does not isolate the layer responsible.

## 5. Quality and reproducibility conditions

- Verify checkpoint SHA-256 against the original manifest when possible. **No automatic checkpoints reconstruction.** Earlier historical replays were not bitwise identical in all cases.
- Verify exact ID joins and binary labels. Never rely on matching row order between CSV and NPZ.
- Keep snapshots of package code, dataset hashes, environment versions, and settings in the eventual findings report.
- Use seed as the independent unit of analysis. Question-level CIs require respecting scene clustering and repeated measurements.
- No comparisons of distinct sample populations as though they were an absolute improvement.
- Do not interpret an unknown result as evidence for the preferred redesign. Negative controls matter.
- If a pretrained checkpoint is missing, report that the corresponding part **could not be executed**, rather than silently restoring an approximate training trajectory.

## 6. Follow-up decisions (only after observing results)

| Result | Candidate architectural direction to *consider* later |
|---|---|
| Readout probe high; original head weak | Spatially structured or task-specific readouts |
| Geometry weak even under controlled probes | Multi-scale convolutional preprocessing |
| Shape good; count-free swaps poor | Localized cross-attribute association |
| Focused Binding improves, consolidation undoes gain | Task interference and objective/optimizer scheduling |

These are **hypotheses for future ablations**, not changes included in this package.
