# Las Vegas GP winner-model feature comparison

Each variant was scored on the same 52 races across nine rolling test blocks. The archived current, grid-only, and full-v2 predictions were reused and their metrics matched the original run. Each new variant adds one feature group to the current model and uses the same chronological penalty selection and temperature-fitting protocol.

## Raw probabilities

| Model | Win log loss | Multiclass Brier | Winners picked | Change in log loss |
| --- | ---: | ---: | ---: | ---: |
| Current baseline | 1.1564 | 0.5368 | 30/52 | +0.0000 |
| Baseline + practice long runs | 1.1652 | 0.5494 | 28/52 | +0.0088 |
| Baseline + recent race pace | 1.1699 | 0.5529 | 32/52 | +0.0136 |
| Baseline + DNF rate | 1.1557 | 0.5358 | 31/52 | -0.0007 |
| V2 with all three groups | 1.1729 | 0.5561 | 32/52 | +0.0165 |
| Grid only | 1.2745 | 0.5440 | 34/52 | +0.1181 |

Lower log loss and Brier are better. A negative change favors the variant over the current baseline.

## Temperature-scaled probabilities

| Model | Win log loss | Multiclass Brier |
| --- | ---: | ---: |
| Current baseline | 1.1791 | 0.5474 |
| Baseline + practice long runs | 1.1783 | 0.5556 |
| Baseline + recent race pace | 1.2012 | 0.5669 |
| Baseline + DNF rate | 1.1807 | 0.5477 |
| V2 with all three groups | 1.1956 | 0.5663 |
| Grid only | 1.2863 | 0.5513 |

## Uncertainty in the raw comparisons

- Baseline + practice long runs: paired log-loss difference +0.009, with a 95% fold-bootstrap interval [-0.027, +0.053].
- Baseline + recent race pace: paired log-loss difference +0.014, with a 95% fold-bootstrap interval [-0.033, +0.070].
- Baseline + DNF rate: paired log-loss difference -0.001, with a 95% fold-bootstrap interval [-0.013, +0.009].
- V2 with all three groups: paired log-loss difference +0.017, with a 95% fold-bootstrap interval [-0.031, +0.076].

Baseline + DNF rate had the lowest raw log loss of the three single-group variants. These are follow-up comparisons on an already evaluated race sample. Keep the current model as the published baseline and use fresh races to confirm any apparent lift.

## Las Vegas forecast comparison

These are pre-race estimates from refits on all 90 historical races. The table is ordered by the current baseline. See the CSV for the full field and all probability variants.

| Driver | Current raw | Long runs raw | Race pace raw | DNF raw | Full v2 calibrated |
| --- | ---: | ---: | ---: | ---: | ---: |
| Max Verstappen | 63.91% | 41.60% | 74.59% | 66.89% | 59.95% |
| George Russell | 10.32% | 16.56% | 6.88% | 9.60% | 10.96% |
| Kimi Antonelli | 9.97% | 17.18% | 8.60% | 9.09% | 14.33% |
| Lando Norris | 9.05% | 15.59% | 6.28% | 8.51% | 10.42% |
| Isack Hadjar | 3.41% | 2.28% | 1.38% | 2.78% | 0.70% |
