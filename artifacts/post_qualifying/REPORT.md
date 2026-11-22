# 2026 Las Vegas GP post-qualifying forecast

Forecast snapshot from 21 November 2026 after qualifying and before the race. The Vegas race result was excluded from training and feature construction.

## Forecast

| Driver | Starting grid | Win probability | Estimated finish |
| --- | ---: | ---: | ---: |
| Max Verstappen | 1 | 56.34% | 1.22 |
| George Russell | 2 | 15.66% | 3.86 |
| Kimi Antonelli | 6 | 15.46% | 3.79 |
| Lando Norris | 3 | 8.89% | 4.17 |
| Isack Hadjar | 4 | 2.22% | 4.97 |

The models estimate different outcomes. Russell has the second-highest win probability. Antonelli has the second-best estimated finish. Probabilities are model estimates conditional on the 22-driver field. The 12-race holdout is too small to establish probability calibration.

## Chronological backtest

Regularization was chosen using six earlier validation races within the 32-race development sample. The final 12 races were scored once for each model variant. The models used for Vegas were then refitted on all 44 available races from 2025 and 2026.

| Features | Correct winner picks | Winner log loss | Finish MAE | Finish RMSE |
| --- | ---: | ---: | ---: | ---: |
| Full | 8/12 | 1.238 | 2.365 | 3.285 |
| Grid only | 8/12 | 1.512 | 2.977 | 3.849 |
| Full without practice | 8/12 | 1.230 | 2.369 | 3.289 |
| Full without constructor form | 8/12 | 1.238 | 2.396 | 3.307 |

Lower log loss, MAE, and RMSE are better. Finish errors are in positions and cover 222 eligible classified driver entries. The raw starting-grid baseline had a 3.108-position MAE. The pole sitter won 8 of 12 holdout races. Practice and constructor form made small differences in this sample. A longer chronological backtest would help establish whether those effects repeat.

## Models and inputs

- **L2-regularized conditional logistic regression** scores drivers against the other entrants and normalizes the scores into win probabilities.
- **Ridge regression** predicts normalized classified finishing positions and rescales them to the size of the field.
- Qualifying used session 11416. Practice used FP2 and FP3, sessions 11414 and 11415. OpenF1 did not return an FP1 classification. The OpenF1 grid agreed with the published Formula 1 grid snapshot.
- Constructor form uses the team’s average normalized finish over its last five Grands Prix that season. Results from the race being forecast are excluded from the form features.

[OpenF1 API](https://openf1.org/docs/) · [Formula 1 qualifying classification](https://www.formula1.com/en/results/2026/races/1300/las-vegas/qualifying) · [Formula 1 starting grid](https://www.formula1.com/en/results/2026/races/1300/las-vegas/starting-grid)
