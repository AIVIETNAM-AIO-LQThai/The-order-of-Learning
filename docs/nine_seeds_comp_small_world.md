# What happens on nine seeds instead of one?

## Purpose

Evaluate whether the order of learning color, shape, and color–shape binding changes final performance and task retention in the original synthetic visual world. Check consistency of the 3 mode performance.

## Protocol

- Seeds: 42–50 (nine seeds); schedules: `mixed`, `staged`, `reverse` (27 runs).
- Same 600 Level-1 training examples, architecture, batch size 32, learning rate 0.002, D4 augmentation, and 450 updates per schedule within each seed.
- Each example is presented 24 times; 75 of the 450 updates are the **common mixed-rehearsal phase**, present in all three schedules.
- Primary comparison: accuracy at the fixed final update on the same ID-development set (40 examples per task).
- Source: `schedule_comparison.json`; environment recorded there as PyTorch 2.13.0+cu130 / NVIDIA RTX 5050 Laptop GPU.

## Final ID-development results

| Task                | Mixed mean | Staged mean | Reverse mean |
| ------------------- | ---------: | ----------: | -----------: |
| Color               |     98.61% |      96.11% |       96.11% |
| Shape               |     71.39% |      71.39% |       72.50% |
| Color–shape binding |     69.72% |      71.94% |       65.28% |
| Overall             |     79.91% |      79.81% |       77.96% |

Binding sample standard deviations across seeds: mixed 5.07 pp, staged 9.17 pp, reverse 9.14 pp.

Paired binding differences, with approximate two-sided 95% t intervals (df=8):

| Comparison       | Mean difference |             95% CI |
| ---------------- | --------------: | -----------------: |
| Staged − mixed   |        +2.22 pp | [−6.12, +10.57] pp |
| Staged − reverse |        +6.67 pp | [−1.98, +15.31] pp |
| Mixed − reverse  |        +4.44 pp | [−4.91, +13.80] pp |

All three intervals include zero. Staged is numerically best on average for binding, but these runs do not establish a reliable final-performance advantage. Mixed has marginally higher mean overall accuracy and lower binding variance.

## Learning dynamics

| Comparison                                                      | Mean pair accuracy (earlier) | Mean pair accuracy (later) |    Change |
| --------------------------------------------------------------- | ---------------------------: | -------------------------: | --------: |
| Staged: end of attributes (250) → end of binding (375)          |                       50.56% |                     76.11% | +25.56 pp |
| Staged: end of binding (375) → end of mixed rehearsal (450)     |                       76.11% |                     71.94% |  −4.17 pp |
| Reverse: end of binding (125) → end of attributes (375)         |                       59.72% |                     51.67% |  −8.06 pp |
| Reverse: end of attributes (375) → end of mixed rehearsal (450) |                       51.67% |                     65.28% | +13.61 pp |

These trajectories suggest interference and recovery; they do not isolate whether the causal driver is task order, phase duration, recency, or rehearsal amount.

## Limitations and next experiment

- Only 40 ID-development examples per task; evaluations are exploratory and are not held-out test/OOD evidence.
- The original strict-ID pair scene distribution has known attribute-distribution constraints, motivating a cleaner dataset design.
- Preserve this dataset and all 27 run records unchanged. Do not pool the earlier seed-42 single run as an independent tenth seed.
- Next: a versioned, harder synthetic visual world (purple, crescent, 3–5 objects, 5,000 unique training scenes) with independent-conjunction versus same-object-binding questions and scene-disjoint evaluation. Continue comparing mixed, staged, and reverse schedules with matched exposures.
- Future mechanism study: vary attribute-phase duration and final mixed-rehearsal duration **in separate controlled experiments** to distinguish prerequisite learning from recency/interference.
