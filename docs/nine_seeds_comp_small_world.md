# Small World: Nine-Seed Learning-Order Replication

*13 November 2026*

An initial single-seed result suggested that Staged training might improve Color–Shape Binding. Replicating the comparison across seeds tested whether that difference was a stable effect or a favorable run.

## Comparison

- **27 runs:** Mixed, Staged and Reverse schedules, paired across seeds **42–50**.
- **Same exposure:** 600 Level-1 training examples, batch size 32, learning rate 0.002, D4 image augmentation and **450 updates** per run. Every example appeared **24 times**, including a common **75-update Mixed rehearsal** suffix.
- **Main measure:** fixed-final ID-dev accuracy, with **40 questions per task**. These are exploratory development results, not held-out transfer tests.

## Final accuracy

| Task | Mixed | Staged | Reverse |
|---|---:|---:|---:|
| Color | **98.61%** | 96.11% | 96.11% |
| Shape | 71.39% | 71.39% | **72.50%** |
| Binding | 69.72% | **71.94%** | 65.28% |
| Overall | **79.91%** | 79.81% | 77.96% |

Binding standard deviations across seeds were **5.07**, **9.17**, and **9.14 percentage points**, respectively. The paired final Binding contrasts were:

| Contrast | Mean difference | Approximate paired 95% interval |
|---|---:|---:|
| Staged − Mixed | **+2.22 pp** | **[−6.12, +10.57]** |
| Staged − Reverse | +6.67 pp | [−1.98, +15.31] |
| Mixed − Reverse | +4.44 pp | [−4.91, +13.80] |

Every interval includes zero. In particular, the +2.22 pp mean Staged advantage is much smaller and less certain than the motivating single-seed result. This is **not evidence of equivalence**, but it does not demonstrate a dependable final benefit either.

## Learning dynamics

| Training interval | Earlier Binding | Later Binding | Change |
|---|---:|---:|---:|
| Staged: end of attributes (250) → end of Binding (375) | 50.56% | 76.11% | **+25.56 pp** |
| Staged: Binding (375) → Mixed consolidation (450) | 76.11% | 71.94% | **−4.17 pp** |
| Reverse: Binding (125) → end of attributes (375) | 59.72% | 51.67% | **−8.06 pp** |
| Reverse: attributes (375) → Mixed consolidation (450) | 51.67% | 65.28% | **+13.61 pp** |

The trajectories show that a task's score can rise under focused training and fall when the training distribution changes. Common Mixed training sometimes restores performance. These are prediction-level effects; they do not identify whether the underlying features have been erased, how rehearsal works, or which component of the schedule caused the change.

## Interpretation

The nine-seed replication **weakened the initial claim of Staged superiority**. It also highlighted the difference between a high intermediate Binding peak and a smaller fixed-final difference after consolidation.

The original world had only 40 ID-dev questions per task, strong structural constraints and nonuniform Color exposure. Those limitations motivated the later Medium World with 3–5 objects, more attributes, and a separate independent-Conjunction task. The later dataset is not a simple larger sample of the same task grammar, so its results cannot be pooled with this replication.

The earlier single-seed trial is part of the motivation, **not a tenth independent seed** in this analysis.

Evidence: the original nine-seed saved schedule comparison and Small World learning notebooks; see also [Small World EDA](small_world_eda_findings.md) and [feasibility findings](small_world_feasibility_findings.md).
