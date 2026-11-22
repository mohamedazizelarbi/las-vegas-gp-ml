# Las Vegas 2026 winner model v2

Post-qualifying forecast using the published starting grid and OpenF1 data available before the race. The Vegas race outcome and race timing were excluded.

## Forecast

| Driver | Grid | V2 calibrated win chance | V2 raw win chance |
| --- | ---: | ---: | ---: |
| Max Verstappen | P1 | 60.0% | 56.1% |
| Kimi Antonelli | P6 | 14.3% | 15.1% |
| George Russell | P2 | 11.0% | 11.8% |
| Lando Norris | P3 | 10.4% | 11.3% |
| Charles Leclerc | P10 | 1.8% | 2.2% |

Usable matched practice runs were available for 19 of 22 Vegas entrants. FP2 contributed 57 matched laps and FP3 contributed 131. FP1 contributed none. Missing runs have a separate model feature. The probability estimates sum to 100% across the field.

## Rolling backtest

The dataset contains 90 historical races across 2023 (22), 2024 (24), 2025 (24), 2026 (20). The scored test covers 52 races from Baku 2024 through São Paulo 2026. Models were refitted on earlier races before each six-race block. The first 12 out-of-sample predictions initialize calibration and are excluded from scoring.

| Model | Probability treatment | Log loss | Brier score | Top picks |
| --- | --- | ---: | ---: | ---: |
| Starting grid only | Raw | 1.274 | 0.544 | 34/52 |
| Starting grid only | Calibrated | 1.286 | 0.551 | 34/52 |
| Current feature set | Raw | 1.156 | 0.537 | 30/52 |
| Current feature set | Calibrated | 1.179 | 0.547 | 30/52 |
| Enriched v2 | Raw | 1.173 | 0.556 | 32/52 |
| Enriched v2 | Calibrated | 1.196 | 0.566 | 32/52 |

Lower log loss and multiclass Brier score are better. Temperature scaling preserves the ordering of drivers, so raw and calibrated variants have identical top picks.

### What the test supports

- Calibrated v2 had 0.016 higher log loss than the calibrated current feature set. The paired block-bootstrap interval for v2 minus the benchmark is [-0.032, +0.077] across 9 scored blocks.
- Calibrated v2 had 0.091 lower log loss than the calibrated grid-only model. The paired block-bootstrap interval for v2 minus the benchmark is [-0.239, +0.055] across 9 scored blocks.
- The best pooled log loss was Current feature set (raw) and the best pooled Brier score was Current feature set (raw). These choices are based on this sample. Inspect the season breakdown and compare prospective forecasts before replacing the existing model.
- Lap matching yielded at least one usable driver race-pace estimate in 90 historical races. DNF rates combine incidents and mechanical failures because the OpenF1 classification flags lack a cause label. Caution filtering uses race-control messages and stint boundaries. Residual differences in traffic, fuel load, and run plans remain.

[OpenF1 data definitions](https://openf1.org/docs/) · [Temperature scaling](https://proceedings.mlr.press/v70/guo17a.html)
