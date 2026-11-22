"""Compare three feature groups using the exact v2 rolling backtest."""
import csv
import json
import math
from collections import defaultdict

from .training import (FEATURES, VARIANTS, calibrate,
                   choose_penalty, choose_scale, metrics, paired_interval, predict,
                   rolling, train, write_csv)
from ..paths import ARTIFACTS, RUNS
from ..snapshot import load_frozen


ADDITIONS = {
    "long_run": ("practice_long_run_gap", "practice_long_run_missing"),
    "race_pace": (
        "driver_recent_race_pace", "driver_recent_race_pace_missing",
        "constructor_recent_race_pace", "constructor_recent_race_pace_missing",
    ),
    "dnf_rate": ("driver_dnf_rate", "constructor_dnf_rate"),
}
CANDIDATES = {name: FEATURES + columns for name, columns in ADDITIONS.items()}
COMPARISON = {"current": VARIANTS["current"], **CANDIDATES,
              "enriched": VARIANTS["enriched"], "grid_only": VARIANTS["grid_only"]}
LABEL = {
    "current": "Current baseline",
    "long_run": "Baseline + practice long runs",
    "race_pace": "Baseline + recent race pace",
    "dnf_rate": "Baseline + DNF rate",
    "enriched": "V2 with all three groups",
    "grid_only": "Grid only",
}
OUT = ARTIFACTS / "win_v2"
RESULTS = RUNS / "feature_comparison"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def reuse_benchmarks(scored):
    """Use the archived out-of-sample predictions for the original models."""
    saved = defaultdict(dict)
    for row in read_csv(OUT / "rolling_predictions.csv"):
        key = int(row["meeting_key"]), int(row["fold_start"]), row["variant"]
        driver = int(row["driver_number"])
        if driver in saved[key]:
            raise ValueError(f"Duplicate archived driver {driver} in {key}.")
        saved[key][driver] = row
    for race in scored:
        roster = [r["driver_number"] for r in race["field"]]
        for variant in VARIANTS:
            key = race["meeting_key"], race["block"], variant
            previous = saved[key]
            if set(previous) != set(roster):
                raise ValueError(f"Roster mismatch against the archived backtest at {key}.")
            winner = [driver for driver in roster if previous[driver]["winner"] == "1"]
            if winner != [race["winner"]]:
                raise ValueError(f"Winner mismatch against the archived backtest at {key}.")
            race["models"][variant] = {
                "raw": [float(previous[driver]["probability_raw"]) for driver in roster],
                "calibrated": [float(previous[driver]["probability_calibrated"]) for driver in roster],
                "i": roster.index(race["winner"]),
                "temperature": float(previous[roster[0]]["temperature"]),
            }
    expected = {r["meeting_key"] for r in scored}
    if expected != {key[0] for key in saved}:
        raise ValueError("Comparison does not cover the same races as the archived backtest.")


def output_metrics(scored, totals, bins):
    archived = json.loads((OUT / "rolling_metrics.json").read_text(encoding="utf-8"))
    for variant in VARIANTS:
        for kind in ("raw", "calibrated"):
            key = f"{variant}_{kind}"
            for metric, value in totals[key].items():
                if not math.isclose(value, archived["overall"][key][metric], abs_tol=1e-12):
                    raise ValueError(f"Archived {key} {metric} differs.")
    contrasts = {
        variant: {
            kind: paired_interval(scored, "current", variant=variant, kind=kind)
            for kind in ("raw", "calibrated")
        }
        for variant in (*CANDIDATES, "enriched")
    }
    by_season = {
        str(year): metrics([r for r in scored if r["year"] == year], COMPARISON)[0]
        for year in sorted({r["year"] for r in scored})
    }
    report = {"test_races": len(scored), "total_history": sum(archived["history_by_year"].values()),
              "meeting_keys": [r["meeting_key"] for r in scored],
              "feature_groups": ADDITIONS, "overall": totals, "by_season": by_season,
              "paired_log_loss_vs_current": contrasts}
    (RESULTS / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(RESULTS / "comparison.csv",
              ("model", "probability_treatment", "test_races", "winner_picks", "log_loss",
               "brier", "delta_log_loss_vs_current", "delta_brier_vs_current"),
              [(name, kind, m["races"], m["top_pick_hits"], m["log_loss"], m["brier"],
                m["log_loss"] - totals[f"current_{kind}"]["log_loss"],
                m["brier"] - totals[f"current_{kind}"]["brier"])
               for name in COMPARISON for kind in ("raw", "calibrated")
               for m in [totals[f"{name}_{kind}"]]])
    write_csv(RESULTS / "calibration_bins.csv",
              ("model", "probability_low", "probability_high", "favorite_count",
               "mean_probability", "observed_win_rate"), bins)
    write_csv(RESULTS / "rolling_predictions.csv",
              ("meeting_key", "year", "race", "fold_start", "driver_number", "winner",
               "variant", "probability_raw", "probability_calibrated", "temperature"),
              [(r["meeting_key"], r["year"], r["location"], r["block"], d["driver_number"],
                int(d["driver_number"] == r["winner"]), variant, m["raw"][i],
                m["calibrated"][i], m["temperature"])
               for r in scored for variant, m in r["models"].items() for i, d in enumerate(r["field"])])
    return contrasts


def output_forecast(history, field, prior):
    old_models = json.loads((OUT / "trained_models.json").read_text(encoding="utf-8"))
    models = {}
    forecasts = {}
    for name, columns in COMPARISON.items():
        if name in old_models:
            model = old_models[name]
        else:
            model = train(history, columns, choose_penalty(history, columns), "winner")
            scale = choose_scale(prior[name])
            model.update(temperature=1 / scale, inverse_temperature=scale,
                         calibration_races=len(prior[name]), training_races=len(history))
        raw = predict(model, field, len(field))
        fitted = calibrate(raw, model["inverse_temperature"])
        for values in (raw, fitted):
            if not math.isclose(sum(values), 1):
                raise ValueError(f"{name} forecast probabilities do not sum to one.")
        models[name] = model
        forecasts[name] = (raw, fitted)
    (RESULTS / "trained_models.json").write_text(json.dumps(models, indent=2), encoding="utf-8")
    order = sorted(range(len(field)), key=forecasts["current"][0].__getitem__, reverse=True)
    write_csv(RESULTS / "vegas_2026_forecast_comparison.csv",
              ("driver", "driver_number", "team", "grid",
               *(f"{name}_{kind}" for name in COMPARISON for kind in ("raw", "calibrated"))),
              [(field[i]["driver"], field[i]["driver_number"], field[i]["constructor"],
                field[i]["starting_grid"], *(x[i] for name in COMPARISON for x in forecasts[name]))
               for i in order])
    return order, forecasts


def write_report(totals, contrasts, field, order, forecasts):
    count = totals["current_raw"]["races"]
    lines = [
        "# Las Vegas GP winner-model feature comparison",
        "",
        f"Each variant was scored on the same {count} races across nine rolling test blocks. "
        "The archived current, grid-only, and full-v2 predictions were reused and their metrics matched the original run. "
        "Each new variant adds one feature group to the current model and uses the same "
        "chronological penalty selection and temperature-fitting protocol.",
        "",
        "## Raw probabilities",
        "",
        "| Model | Win log loss | Multiclass Brier | Winners picked | Change in log loss |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in COMPARISON:
        m = totals[f"{name}_raw"]
        delta = m["log_loss"] - totals["current_raw"]["log_loss"]
        lines.append(f"| {LABEL[name]} | {m['log_loss']:.4f} | {m['brier']:.4f} | "
                     f"{m['top_pick_hits']}/{count} | {delta:+.4f} |")
    lines += [
        "",
        "Lower log loss and Brier are better. A negative change favors the variant over the current baseline.",
        "",
        "## Temperature-scaled probabilities",
        "",
        "| Model | Win log loss | Multiclass Brier |",
        "| --- | ---: | ---: |",
    ]
    for name in COMPARISON:
        m = totals[f"{name}_calibrated"]
        lines.append(f"| {LABEL[name]} | {m['log_loss']:.4f} | {m['brier']:.4f} |")
    lines += ["", "## Uncertainty in the raw comparisons", ""]
    for name, contrast in contrasts.items():
        m = contrast["raw"]
        lo, hi = m["bootstrap_95"]
        lines.append(f"- {LABEL[name]}: paired log-loss difference {m['mean_difference']:+.3f}, "
                     f"with a 95% fold-bootstrap interval [{lo:+.3f}, {hi:+.3f}].")
    best = min(CANDIDATES, key=lambda name: totals[f"{name}_raw"]["log_loss"])
    lines += [
        "",
        f"{LABEL[best]} had the lowest raw log loss of the three single-group variants. "
        "These are follow-up comparisons on an already evaluated race sample. Keep the current "
        "model as the published baseline and use fresh races to confirm any apparent lift.",
        "",
        "## Las Vegas forecast comparison",
        "",
        "These are pre-race estimates from refits on all 90 historical races. The table is ordered "
        "by the current baseline. See the CSV for the full field and all probability variants.",
        "",
        "| Driver | Current raw | Long runs raw | Race pace raw | DNF raw | Full v2 calibrated |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i in order[:5]:
        values = [forecasts[k][j][i] for k, j in (("current", 0), ("long_run", 0),
                  ("race_pace", 0), ("dnf_rate", 0), ("enriched", 1))]
        lines.append(f"| {field[i]['driver'].title()} | " + " | ".join(f"{p:.2%}" for p in values) + " |")
    (RESULTS / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    history, field, _, _ = load_frozen()
    scored, prior = rolling(history, CANDIDATES)
    reuse_benchmarks(scored)
    totals, bins = metrics(scored, COMPARISON)
    RESULTS.mkdir(parents=True, exist_ok=True)
    contrasts = output_metrics(scored, totals, bins)
    order, forecasts = output_forecast(history, field, prior)
    write_report(totals, contrasts, field, order, forecasts)


if __name__ == "__main__":
    main()
