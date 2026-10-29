# AIO Conquer Module 5 — Experimental Research Design & Dataset Protocol v0.1

**Working theme:** From concepts to expertise: developmental curricula, compositional generalization, and adaptation in Vision Transformers  
**Status:** Design ready for generator implementation; **not preregistered, not empirically validated, and not final title**  
**Prepared:** 2026-10-25  
**Predecessor:** `AIO_M5_Research_Charter_v0.1.md`

## 0. Executive decision

**Provisional scientific question:** When identical small Vision Transformers learn elementary visual concepts, object relations, and composed relations in different orders, do their learned representations differ in (i) generalization to unseen attribute combinations and (ii) acquisition of a new geometric relation?

**Three competing hypotheses**

- **H1 — Reusable abstraction:** A prerequisite-first curriculum transfers better to held-out combinations and/or new tasks than matched mixed and reverse curricula.
- **H2 — Specialization trade-off:** The schedule leading to strongest familiar-task mastery can exhibit weaker transfer or slower adaptation.
- **H3 — No persistent order effect:** Any initial differences shrink after an identical final mixed-training phase; quantity, diversity, optimization, or recency explains most of the apparent curriculum advantage.

These hypotheses need not be mutually exclusive at every checkpoint. The paper may report null or contradictory effects. **Do not label this a study of actual brain development, human expertise, biological critical periods, or demonstrated loss of plasticity.** A short sequence of three task families does not reproduce long-run continual learning.

### What is new *enough to investigate*, not yet claimed novel

Progressive VQA training already exists (Akl et al., BMVC 2025). Held-out visual attribute binding and diversity effects already exist (Berlot-Attwell et al., EMNLP 2024). Loss of neural-network plasticity and transferability differences across training stages also exist (Dohare et al., Nature 2024; Lee et al., ICLR 2026). Our *candidate combination* is a controlled, fixed-architecture experiment that evaluates **both** held-out concept composition **and** subsequent acquisition of an unseen relation after matched task exposure. Novelty must be reassessed after inspecting full related methods.

## 1. Variables and estimands

### Independent variable (primary causal treatment)

Training order `schedule`:

1. `progressive`: Level 1 → Level 2 → Level 3.
2. `mixed`: Level 1/2/3 interleaved at matched counts.
3. `reverse`: Level 3 → Level 2 → Level 1.

A **common mixed consolidation phase** follows all three. This is a training-history study, not an architecture-comparison study.

### Outcomes

- **Primary outcome for RQ1:** Level-3 compositional-OOD accuracy after the common consolidation phase, reported together with Level-3 ID accuracy.
- **Primary outcome for RQ2:** Accuracy-versus-update learning curve for a new `CLOSER` relation when fitting a new task head to a **frozen visual encoder**, using the same adaptation examples and updates across curricula. Report normalized area under that curve (AULC).
- **Secondary outcome for RQ3:** Familiar-task accuracy versus novel-task adaptation across curricula and checkpoints; when fully fine-tuning, also measure old-task retention.
- **Supporting diagnostics:** Level-1 attribute binding, Level-2 spatial accuracy, paired counterfactual behavior, per-truth-pattern Level-3 accuracy, seed-level uncertainty.

The term *plasticity* here means **operational task adaptability under a specified optimization procedure**. Frozen-encoder transfer measures representation usefulness, *not* weight-update plasticity; full fine-tuning is the complementary experiment. Neither alone proves intrinsic biological plasticity.

## 2. Core visual world

### 2.1 Scene design

| Factor | v0.1 specification | Rationale |
|---|---|---|
| Image | RGB, `64 × 64`, fixed off-white/neutral background | Small enough for from-scratch ViT training |
| Objects | **Exactly three**, non-overlapping | Same visual complexity across task levels; difficulty comes from the query |
| Colors | `red`, `blue`, `green`, `yellow` | Four primitives, twelve possible color–shape pairs |
| Shapes | `circle`, `square`, `triangle` | Three primitives |
| Size | `small`, `large` independent of target | Nuisance variable / capacity for diverse scenes |
| Location | Three distinct x-columns and three distinct y-rows selected from four of each | Unambiguous left/above comparisons and ample arrangements |
| Grid centers | `x,y ∈ {11,25,39,53}` pixels | Enough margin for objects; use small, bounded jitter |
| Position jitter | Independently `−1,0,+1` pixels, seeded | Increases appearances without changing ordering |
| Rendering | Antialiased shapes; standardized fill, outlines, background and resize | Avoid unintended visual artifacts distinguishing train and test |

**Role construction:** Sample two distinct colors `c0 ≠ c1` and two distinct shapes `s0 ≠ s1`. Create:

- `A = (c0, s0)` — the **anchor**.
- `B = (c0, s1)` — shares the anchor's color.
- `C = (c1, s0)` — shares the anchor's shape.

The fourth possible pair, `(c1, s1)`, is **absent**.

Therefore, the anchor's color alone cannot distinguish it from B, and its shape alone cannot distinguish it from C. A query must bind *both* properties to uniquely identify A. **Role IDs A/B/C are metadata, never model inputs.** Object order and file naming must not expose them.

Example: `A = red circle`, `B = red square`, `C = blue circle`; `blue square` is the absent pair. A model must not answer `EXISTS_PAIR(blue,square)` by independently checking that *blue* and *square* exist somewhere in the image.

**Pre-implementation geometry checks:** Because all three objects have distinct x-columns and y-rows, pairwise minimum center separation stays above the maximum combined object radii (nominal cap `≤7px` per object with bounded jitter). Enumerate every size and position configuration, inspect bounding boxes and masks, and reject collisions/occlusion. Position labels use numeric object centers from the scene graph, not guessed pixel features.

### 2.2 Exact structural capacity (before jitter)

There are `4P2 × 3P2 = 12 × 6 = 72` ordered anchor/color-shape configurations before applying train/OOD restrictions, `4P3 = 24` choices for distinct x-columns, `4P3 = 24` choices for distinct y-rows, and `2^3 = 8` size combinations. This is sufficient for many distinct graph-level scenes without depending on image augmentation.

All attributes, spatial positions, backgrounds and jitter must be sampled independently of the target label *unless intentionally conditioned to create balanced label cases*. Log the sampling/rejection mechanism and raw acceptance rates.

## 3. A machine-checkable task grammar

Use **symbolic program tokens** as questions for the primary study. No pretrained language model and no unconstrained natural-language templates. The model sees the image and the serialized symbolic query, *not* the oracle scene graph.

### Level 1 — Atomic properties and attribute binding

- `EXISTS_COLOR(c)` → `1` iff some object has color `c`.
- `EXISTS_SHAPE(s)` → `1` iff some object has shape `s`.
- `EXISTS_PAIR(c,s)` → `1` iff a **single** object simultaneously has color `c` and shape `s`.

Sample each subfamily and each binary answer in balanced proportions. For `EXISTS_PAIR` the priority negative is **hard**: both queried color and queried shape appear in the scene, but not on the same object. Neither color-only nor shape-only rules should solve the pairing task.

### Level 2 — Spatial relation grounded in a conjunctive object reference

An object reference is `REF(c,s)`, resolving to a **unique** object with that specific color–shape pair.

- `LEFT(REF(c0,s0), REF(c0,s1))` → `1` iff `x(A) < x(B)`.
- `ABOVE(REF(c0,s0), REF(c1,s0))` → `1` iff `y(A) < y(C)` (image y-axis grows downward).

For Level 2, both referenced objects must exist exactly once. Do **not** create an unstated missing-reference case; that would mix relation evaluation with existence detection. Balance `True/False` within each predicate and query family.

### Level 3 — Composition of two independently learnable relations

Let:

- `p = LEFT(REF(A), REF(B))`.
- `q = ABOVE(REF(A), REF(C))`.

Here `REF(A)` denotes the generated expression `REF(c0,s0)`, and analogously for B and C; the symbolic A/B/C role names are never presented to the model.

**Primary composition task:** `XOR(p,q)`; oracle label `p != q`. Equal representation of truth combinations `(p,q) ∈ {(0,0),(0,1),(1,0),(1,1)}` yields 50% positive answers, and either predicate alone cannot exceed chance under this balanced construction.

**Secondary / interpretability task:** `AND(p,q)`; oracle label `p && q`. Binary-label balancing for AND necessarily skews the relative frequencies of the four truth states, allowing partial-predicate heuristics to beat chance. **Never interpret AND accuracy alone as proof of conjunctive reasoning.** Report truth-table-conditioned accuracy and an explicit single-predicate heuristic baseline. Defer AND until the XOR preliminary works.

**Why the Level-3 task is meaningful:** The model must ground multiple attributes to objects, evaluate two spatial relations, then combine the truth values. More complex evaluation questions are not simply larger images.

### Query tokenization (proposed, not code-frozen)

- Vocabulary includes color, shape, `REF`, `EXISTS_COLOR`, `EXISTS_SHAPE`, `EXISTS_PAIR`, `LEFT`, `ABOVE`, `XOR`, `AND`, `CLOSER`, structural delimiters and padding.
- Queries are serialized in a canonical prefix form, e.g. `XOR LEFT REF red circle REF red square ABOVE REF red circle REF blue circle`.
- A single shared binary head handles Levels 1–3; query-conditioned embeddings interact with ViT image patches.
- Reserve `CLOSER` as an unseen operator token during initial training, or add it during adaptation with controlled initialization. **It must not receive supervised training before the adaptation phase.**
- The query encoding must not include role IDs, intended answer, sequence rank, split name, or a task-specific answer prior.

## 4. Split policy: ID, held-out concept pairs and new-task adaptation

### 4.1 Primary held-out color–shape combinations

Declare the fixed set:

```text
H = {
  (red, triangle),
  (blue, square),
  (green, circle)
}
```

Each primitive color and shape appears in allowed training pairs; the **conjunction** is novel.

- **Train / ID validation / ID test:** No object **and no question-reference pair** from H, including negative questions.
- **OOD validation / OOD test:** Include novel pairs from H as target referents or queried pair candidates, while preserving the original color/shape inventory and the distribution of scene sizes, positions, rendering, and task difficulty.
- Never tune using OOD *test*. OOD *development* is for data-audit and feasibility assessment only; once full comparisons begin, do not repeatedly adjust methods to OOD test results.

**Important:** A held-out pair merely placed in the background is not an adequate compositional generalization evaluation. The queried **anchor** in Levels 2 and 3 must be the held-out pair for the OOD sample. In Level 1, the held-out pair must itself be the query target.

### 4.2 Feasibility enumeration, checked against the role construction

Using the four-color/three-shape grammar and H above:

- **24** role configurations have all three visible pairs outside H.
- **12** configurations additionally have the missing fourth pair outside H; using these for the primary training pool permits clean hard-negative `EXISTS_PAIR` questions without ever displaying H in a training query.
- Those 12 configurations cover **all 9 allowed color–shape pairs**.
- **18** role configurations place a held-out pair at the OOD anchor A while keeping B and C in the allowed pair set (**6 per held-out pair**).
- **12** role configurations have no held-out object in the image but have a held-out *missing fourth pair*, useful for Level-1 **OOD negative** questions (**4 per held-out pair**).
- Across all 36 permutations of three ordered x/y slots (or all 576 permutations when drawn from four columns/rows), all four truth patterns for `(LEFT(A,B), ABOVE(A,C))` are possible.

These are **structural feasibility counts**, not an audited final dataset or result. The renderer and sampler must reproduce and unit-test them.

### 4.3 How to split scenes without leakage

- Generate candidate **symbolic scene graphs first**; sort object tuples canonically and hash the attributes, precise locations and sizes to obtain `scene_graph_id`.
- Partition canonical graph IDs into train, ID-dev and ID-test *before* rendering or generating multiple questions per scene. No graph ID crosses these partitions.
- Keep a separate set of OOD graphs (with H at the anchor) and split it independently into OOD-dev and OOD-test.
- Distinct jittered photographs of the **same symbolic graph** belong to the **same partition** to avoid variants of the same scene on both sides.
- Questions with different truth operators about the same scene also remain in its partition.
- Split manifests, random seeds, IDs and SHA-256 digests of generated artifacts are frozen before full training.
- **ID and OOD tests** should match task type, query length, target prevalence (50% per level/template), coordinate/size marginals and rendering settings as closely as sampling permits. Preserve any remaining differences in a balancing report.

### 4.4 Counterfactual evaluation pairs

For a fixed query, construct a minimal matched change that flips exactly the intended predicate or property:

- `EXISTS_PAIR`: change the shape/color binding of one object while keeping constituent colors/shapes elsewhere represented, if feasible.
- `LEFT`: move or swap relevant objects to reverse the left/right relation while preserving their attributes.
- `XOR`: change one underlying relation's truth value while holding the other constant, retaining valid nonoverlapping geometry.

Validate generated pairs with the symbolic oracle. Some pairs may be impossible with fixed nuisance factors; do not silently substitute unmatched scenes. Save an explicit `pair_id` and changed factors.

## 5. New-relation adaptation (separate from composition OOD)

### Operator

`CLOSER(REF(A),REF(B),REF(C))` is `1` iff `dist(A,B) < dist(A,C)` using Euclidean distance between object centers. Filter nearly tied cases using a fixed minimum separation between the two distances (e.g., `>4 pixels`), with **balanced positive and negative answers**.

Why CLOSER? It is a geometric relation not explicitly supervised during Levels 1–3, yet visual and coordinate features from earlier tasks may transfer. It is therefore useful for measuring acquisition of a **new labeled function**, not magical recognition of a previously known language token.

### Adaptation dataset

- Adaptation examples use **allowed color–shape pairs** initially, to separate *novel operator* from *novel object binding*. Optional cross-OOD transfer is secondary.
- Fixed `128` labeled adaptation examples in the core setting; new scene IDs disjoint from all original training, validation and test scene IDs. Provide a shared held-out new-relation evaluation set (e.g., `2000` questions) for every condition.
- Balance both answers and distance-margin bins; keep renderer, concept inventory and query arguments consistent across schedules.
- **Same ordered adaptation examples, same initial new-head parameters and same evaluation set** for the three histories at a given seed.

### Two complementary adaptation modes

1. **Frozen visual encoder (primary transferability test):** Freeze the pretrained ViT. Train a newly initialized task-specific readout and, if necessary, a newly introduced operator embedding/query interface on `CLOSER`. Clearly log exactly what is frozen. This tests availability of transferable visual information; it is not proof of gradient plasticity in frozen layers.
2. **End-to-end fine-tuning (secondary plasticity/forgetting test):** Initialize a new relation head identically across conditions, reset adaptation optimizer for everyone, and update the ViT+query components under an identical adaptation protocol. Keep the original Levels 1–3 head and measure its performance before and after adaptation. Report both new-task acquisition and forgetting.

Suggested checkpoints are `0, 10, 25, 50, 100, 200` gradient updates (batch size 32; repeat from the fixed 128-example pool according to a seeded stream). Report accuracy and **normalized trapezoidal AULC** over updates. This measures *learning speed given the specified 128 examples*, **not** learning from 200 different examples. An optional `32/128/512`-example budget ablation measures sample efficiency more directly.

**Baselines:** Same task head on a randomly initialized frozen ViT; optionally on a fully trained mixed model; and end-to-end learning from scratch with matched adaptation steps. Separate transfer *benefit* from complete new-task learning capacity.

## 6. Model baseline and train schedules

### Primary model (held fixed)

A **small ViT image encoder trained from scratch**, with candidate inputs/configuration:

```yaml
image_size: 64
patch_size: 8
embed_dim: 128
vit_depth: 4
vit_heads: 4
query_encoder: small_trainable_token_encoder
fusion: query_to_image_cross_attention  # or a single jointly encoded sequence
output: shared_binary_logit
loss: binary_cross_entropy
optimizer: AdamW
learning_rate: 0.0003       # preliminary tune once on ID-dev, then freeze
weight_decay: 0.05         # preliminary default, not proven optimal
batch_size: 64
initial_training_steps: 2400
common_mixed_steps: 600
seeds_minimum: 3
seeds_preferred: 5
```

Pretraining on outside vision corpora is disallowed in the primary experiment: it would import a visual developmental history we cannot control. Keep the architecture, optimizer policy, initialization and total steps identical within paired seeds. A simple CNN+question baseline can serve as a sanity check, not a second research axis.

### Example-exposure-matched training streams

Let each level have a deterministic **ordered** stream of `64,000` scene/question exposures, which can repeat examples from that level's fixed training pool. Give *every condition the exact same stream from each level* (including repeat/augmentation IDs) and differ **only in when each stream is consumed**.

- Schedule-specific phase = `2400` optimizer steps = `800` batches from each level at batch size `64` = `51,200` exposures per level.
- Common final phase = `600` steps = `200` batches per level = `12,800` additional exposures per level.
- Total = `3000` steps; `64,000` exposures per level; `192,000` total.

| Step range | Progressive | Mixed | Reverse |
|---|---|---|---|
| `1–800` | L1 only | L1/L2/L3, interleaved | L3 only |
| `801–1600` | L2 only | L1/L2/L3, interleaved | L2 only |
| `1601–2400` | L3 only | L1/L2/L3, interleaved | L1 only |
| `2401–3000` | Same mixed stream for all | Same mixed stream for all | Same mixed stream for all |

For mixed training, use homogeneous level-specific minibatches in a seeded interleaving, to avoid changing loss normalization or batch composition. Within each level, maintain the same order of examples in all schedules. If batch order or augmentations differ across schedules, the treatment is no longer isolated.

**Optimizer state & learning rate:** Keep optimizer state across phases and use a fixed global-step learning-rate policy, initially a constant LR for transparent timing comparisons. **Never reset the optimizer only for progressive/reverse conditions.** At adaptation, reset the *new-task* optimizer for all models. If later adding warm-up, decay or optimizer resets, treat them as separate explicit ablations. Log gradient statistics and actual examples processed.

**Checkpoints:** Save at initialization, after each 800-step block (800/1600/2400) and after the common tail (3000), for each curriculum and seed. Compare at matched global steps, but note that task coverage differs before step 2400. The primary causal comparison is at step 3000.

## 7. Dataset sizes and balance targets (proposals, to verify in preliminary)

| Partition | Preliminary target | Remarks |
|---|---:|---|
| Training scene/question pool | 12,000 per level (`36,000` examples) | May reuse base symbolic scenes with diverse queries; record unique graph counts |
| ID development | 1,500–2,000 per level | Choose one exact size before training |
| ID test | 2,000 per level | Held-out graph IDs, fixed throughout |
| OOD development | 1,000–2,000 per level | Feasibility/data-audit only before freeze |
| OOD test | 2,000 per level | At least one H pair is the query-relevant anchor |
| Adaptation train | 128 labeled `CLOSER` examples | Repeated under fixed 200-step optimization budget |
| Adaptation test | 2,000 `CLOSER` examples | Independent graph IDs and balanced labels |

Across partitions, maintain parity by label **within each template**, balanced color/shape query frequencies where possible, and balanced `(p,q)` truth cases for XOR. Report counts before making any performance claim. For `EXISTS_PAIR` OOD, balance **positive held-out pair exists** with **hard-negative held-out pair absent while its component attributes are present separately**.

## 8. Label oracle and split audits: required pass/fail checks

The dataset generator should fail loudly if any check below fails:

1. **Scene validity:** Exactly three shapes, legal colors/shapes/sizes, non-overlapping masks, bounded coordinates, distinct x and y columns, unique `(color,shape)` pairs, and a uniquely identifiable anchor A.
2. **Reference resolution:** Every L2/L3/adaptation `REF(c,s)` resolves to exactly one object; `EXISTS_PAIR` may intentionally resolve to zero.
3. **Semantic oracle:** Independently recompute `EXISTS_*`, `LEFT`, `ABOVE`, `XOR`, `AND`, `CLOSER` labels directly from metadata. Test boundary cases for LEFT/ABOVE and distance margins.
4. **No held-out-pair contamination:** H never appears in **any training/ID image** nor in any **training/ID query**, including negative queries. Every OOD test item explicitly identifies which held-out pair drives it.
5. **Primitive coverage:** All 4 colors, all 3 shapes, all 9 permitted pairs and both elementary spatial predicates occur in train.
6. **Graph split integrity:** No canonical graph hash overlaps train, ID-dev, ID-test, OOD-dev/test, or new-task adaptation splits as applicable.
7. **Label balance:** Within each task template and split, positive fraction is close to 0.5 (set tolerance of ±0.01); XOR truth patterns each close to 0.25 (±0.02). Ensure query-only class priors are not systematically predictive.
8. **Counterfactual pairs:** Every prescribed change has the intended effect, with all unmodified factors preserved. Reject invalid changes explicitly.
9. **Shortcut tests:** Train a query-only baseline, an image-only baseline, and simple single-predicate heuristics for composed tasks. Report their accuracy; for a balanced binary query task, the expected uninformed baseline is ≈50%.
10. **Metadata privacy for model input:** No graph IDs, roles, RNG seed, split names, target labels, held-out flags or template counters in image pixels or query tokens.
11. **Reproducibility:** Rerunning generator with same configuration yields identical IDs, labels and file hashes. Dataset manifests record generator version and configuration hash.
12. **Pair feasibility:** The independent feasibility enumerator reproduces the structural counts in §4.2; failures block training.

### Minimal hand-checked oracle vectors

Use the example `A=red circle at (11,11)`, `B=red square at (39,39)`, `C=blue circle at (25,53)`:

| Query | Expected |
|---|---:|
| `EXISTS_COLOR(red)` | 1 |
| `EXISTS_SHAPE(triangle)` | 0 |
| `EXISTS_PAIR(red,circle)` | 1 |
| `EXISTS_PAIR(blue,square)` | 0 (both primitives appear, on different objects) |
| `LEFT(REF(red,circle),REF(red,square))` | 1 |
| `ABOVE(REF(red,circle),REF(blue,circle))` | 1 |
| `XOR(LEFT(A,B), ABOVE(A,C))` | 0 |
| `AND(LEFT(A,B), ABOVE(A,C))` | 1 |
| `CLOSER(A,B,C)` | 1 (`dist(A,B) ≈ 39.6 < dist(A,C) ≈ 44.3`) |

**Implementation rule:** Replace these hand-checked examples with verified executable test vectors before training. The deterministic oracle—not a manually typed table—is authoritative.

## 9. Evaluation, uncertainty and statistical interpretation

### Primary reporting table

Rows: `schedule × seed × checkpoint`; columns: L1 attribute accuracy, L1 pair-binding accuracy, L2 relation accuracy, L3 ID XOR accuracy, L3 OOD XOR accuracy, OOD–ID gap, new-relation AULC, final adaptation accuracy, old-task retention (full fine-tuning only).

### Measures

- Accuracy by level and subtask (never aggregate all levels into one misleading average).
- `Gap = Acc_ID − Acc_OOD` alongside **absolute OOD accuracy**.
- `AULC = 1/(Kmax−Kmin) * Σ_i (k[i+1]−k[i]) * (a[i]+a[i+1])/2` for checkpointed adaptation accuracy `a[k]`.
- Forgetting after full fine-tuning: `F = Acc_old_before − Acc_old_after`. Negative F means retention improved, not forgetting.
- Pre/post-tail comparisons to measure persistence of curriculum effects.
- Paired counterfactual consistency where the target label changes as intended; report both paired examples rather than hiding one.

Use **paired seeds** to estimate differences: initialize all three curricula identically per seed, use matched level-specific example streams, and compare the three final outcomes by seed. Show each seed point and mean/median with uncertainty. Bootstrap or permutation procedures should resample paired **runs/seeds** (and optionally item clusters within runs), not treat 2000 test items as independent model-retraining replicates. Three seeds yield weak uncertainty estimates; target five if compute allows. Don't claim statistical significance from a single seed.

### Diagnostic caution

A gain on L3 OOD could stem from better reference grounding, better spatial encoding, better logical composition, or their combination. Disentangle with L1 `EXISTS_PAIR` performance, L2 relation performance and conditioned error audits. Calling the gain “abstract reasoning” without these checks would overstate the result.

Frozen-encoder adaptation gains may reflect feature quality rather than increased gradient-update plasticity. Full-model learning curves are required before making claims about specialization-driven loss of plasticity. Short-course failures are not sufficient evidence of a biological critical period.

## 10. preliminary gate and fallback rules

### dataset generation — Generator and oracle (must pass before training)

- Enumerate structural combinations and run the 12 dataset audits.
- Display at least 24 example scenes and 8 counterfactual pairs, with oracle explanations.
- Check positive/negative balance, held-out-pair isolation, and case distribution of the spatial predicates.
- Freeze a preliminary manifest and count unique structural graphs.

### model feasibility study — Learnability and genuine difficulty

Train **only a mixed schedule** for ~600–1,200 steps as an inexpensive dry run. Evaluate all task families using ID-dev and OOD-dev (not final test). Requirements are behavioral, not arbitrary SOTA targets:

- L1 attributes and pair binding clearly above their audited trivial baselines.
- L2 spatial tasks clearly above chance.
- L3 XOR is learnable and nontrivial; if near chance indefinitely, first inspect query conditioning and oracle. If saturated immediately, increase compositional difficulty or visual diversity *before* the final protocol freeze.
- New-relation adaptation on a small dry-run split demonstrably improves above chance when fine-tuned.

If the preliminary fails: fix generator/reference ambiguity before touching ViT capacity; test a simple CNN+question baseline and/or structured-feature oracle baseline; simplify the query tokenization if necessary. **Do not** infer anything about human-like cognitive development from a failed model implementation.

### Freeze gate

After dataset and model development, issue `dataset_protocol_v1.0.md` with exact dataset counts, final hyperparameters, seeds, training budget, outcome definitions, and checksumed test manifests. **Only then** run all curricula and consider finalizing the formal project title.

## 11. Implementation order and proposed repository structure

```text
research/
  research_charter.md
  literature_matrix.md
  docs/dataset_design.md
  decision_log.md
configs/
  world.yaml
  model.yaml
  experiment.yaml
src/
  world.py           # scene factor sampling
  render.py          # deterministic 64x64 rendering
  grammar.py         # symbolic query AST + serializer
  oracle.py          # executable answers
  splits.py          # held-out pairs + graph-level partition
  dataset.py         # image/query tensors and manifests
  model.py           # fixed ViT + query interaction
  train.py           # curriculum ordering with matched streams
  evaluate.py        # ID/OOD, adaptations, errors
  adapt.py           # frozen/readout and full-finetuning modes
tests/
  test_world.py
  test_oracle.py
  test_splits.py
  test_balance.py
  test_reproducibility.py
  test_exposure_match.py
artifacts/
  manifests/
  checkpoints/
  metrics/
  figures/
```

**Recommended first engineering task:** Implement **only** `world.py`, `grammar.py`, `oracle.py`, `splits.py`, and the corresponding tests plus a small scene renderer. Do not create training loops yet. First deliverable is a manifest and evidence that the dataset answers the intended research questions.

## 12. Title selection gate

| Evidence after dataset generationnd initial comparative runs | Defensible title direction |
|---|---|
| Reliable training + well-audited OOD, but adaptation is weak/inconclusive | **From Attributes to Relations: Curriculum Order and Compositional Generalization in Vision Transformers** |
| Strong controlled new-relation adaptation differences | **The Price of Specialization: Learning History and Adaptability in Vision Transformers** |
| Reliable analyses on both compositionality and new-task adaptation, independent of positive/negative sign | **From Concepts to Expertise: Curriculum Order, Generalization, and Adaptation in Vision Transformers** |
| Implementation remains conceptually interesting but findings are null | **Does Learning Order Matter? A Controlled Study of Visual Concept Learning in Transformers** |

**Current working title:** *Learning to Learn, or Learning to Specialize?*  
A scientific title should describe the experiment, not presume that developmental training creates intelligence.

## 13. Literature and dataset references (verified landing pages)

1. Akl et al. (BMVC 2025), **Task Progressive Curriculum Learning for Robust Visual Question Answering** — https://bmvc2025.bmva.org/proceedings/294/
2. Berlot-Attwell et al. (EMNLP 2024), **Attribute Diversity Determines the Systematicity Gap in VQA** — https://aclanthology.org/2024.emnlp-main.537/
3. Johnson et al. (CVPR 2017), **CLEVR** — https://cs.stanford.edu/people/jcjohns/clevr/
4. Dohare et al. (Nature 2024), **Loss of Plasticity in Deep Continual Learning** — https://www.nature.com/articles/s41586-024-07711-7
5. Lee et al. (ICLR 2026), **Understanding the Learning Phases in Self-Supervised Learning via Critical Periods** — https://proceedings.iclr.cc/paper_files/paper/2026/hash/86ddf3543ad437d71c37e510f41b1a53-Abstract-Conference.html
6. Rajaraman et al. (June 2026), **Learning to Reason with Curriculum II: Compositional Generalization** — https://www.microsoft.com/en-us/research/publication/learning-to-reason-with-curriculum-ii-compositional-generalization/

---

### Open decisions (explicitly *not* final)

- Does 64×64 contain sufficient information for learning all intended predicates with a small ViT? preliminary will decide.
- Can the fixed 3-object scene world avoid task-template or query-length shortcuts? Require explicit audits.
- Is the primary OOD task challenging but learnable? Adjust *before* final freezing.
- Does new `CLOSER` adaptation test feature reuse, learning dynamics, or both? Separate readout and full-tuning analyses.
- Are three seeds enough? Prefer five or more before any strong statistical claim.
- Can results replicate on a second generator or CLEVR-like environment? Treat as extension, not a condition for initial AIO project completion.
