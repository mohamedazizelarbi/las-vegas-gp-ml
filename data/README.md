# Data used in the Las Vegas study

| File | Role |
| --- | --- |
| `model-inputs-2023-2026.json.gz` | Versioned input snapshot. Each historical item contains its meeting key, location, season, classification labels, engineered pre-race field and race winner. The 22-driver Vegas field is stored under `target` and contains only input features. Timing diagnostics are under `coverage`. |
| `vegas_2026_official_grid.csv` | Published Vegas starting grid used as a pre-race feature. |
| `vegas_2026_final_classification.csv` | FIA document 65 transcribed for post-race scoring. Used by the audit command. |

The historical sample ends before the 2026 Las Vegas race. For field definitions and missing-data rules, see [`training.py`](../lasvegas_ml/v2/training.py) and [`lap_features.py`](../lasvegas_ml/v2/lap_features.py).
