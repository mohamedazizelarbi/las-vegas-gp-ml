"""Fit a calibrated post-qualifying F1 winner model with rolling race-level tests."""
import csv
import json
import math
import random
from collections import Counter, defaultdict, deque
from statistics import fmean

from .download_data import BASE, weeks
from .lap_features import meeting_pace
from ..baselines.train_post_qualifying import FEATURES, OpenF1, clean_finish, predict, train, weekend, write_csv
from ..paths import RUNS
from ..snapshot import load_frozen


ENRICHED = FEATURES + (
    "practice_long_run_gap", "practice_long_run_missing",
    "driver_recent_race_pace", "driver_recent_race_pace_missing",
    "constructor_recent_race_pace", "constructor_recent_race_pace_missing",
    "driver_dnf_rate", "constructor_dnf_rate",
)
VARIANTS = {"grid_only": FEATURES[:1], "current": FEATURES, "enriched": ENRICHED}
PENALTIES = (0.02, 0.08, 0.25)
INITIAL_TRAIN = 26
BLOCK = 6
MIN_CAL_RACES = 12
OUT = RUNS / "win_v2"
EPS = 1e-15


class Form:
    def __init__(self):
        self.driver_finish = defaultdict(lambda: deque(maxlen=5))
        self.team_finish = defaultdict(lambda: deque(maxlen=5))
        self.driver_pace = defaultdict(lambda: deque(maxlen=5))
        self.team_pace = defaultdict(lambda: deque(maxlen=5))
        self.driver_dnfs = defaultdict(lambda: deque(maxlen=10))
        self.team_dnfs = defaultdict(lambda: deque(maxlen=10))

    def enrich(self, field, practice):
        for driver in field:
            n, team = driver["driver_number"], driver["constructor"]
            long_gap, count = practice.get(n, (0, 0))
            d_pace = [p for p in self.driver_pace[n] if p is not None]
            t_pace = [p for p in self.team_pace[team] if p is not None]
            driver.update(
                practice_long_run_gap=long_gap, practice_long_run_missing=int(not count),
                driver_recent_race_pace=sum(d_pace) / (len(d_pace) + 2),
                driver_recent_race_pace_missing=int(not d_pace),
                constructor_recent_race_pace=sum(t_pace) / (len(t_pace) + 2),
                constructor_recent_race_pace_missing=int(not t_pace),
                driver_dnf_rate=(sum(self.driver_dnfs[n]) + 0.48) / (len(self.driver_dnfs[n]) + 4),
                constructor_dnf_rate=(sum(self.team_dnfs[team]) + 0.48) / (len(self.team_dnfs[team]) + 4),
                matched_practice_laps=count,
            )

    def observe(self, results, roster, race_pace):
        teams = {r["driver_number"]: r["constructor"] for r in roster if r["constructor"]}
        team_laps = defaultdict(list)
        team_dnfs = defaultdict(list)
        for r in results:
            if r.get("dns") or r.get("dsq"):
                continue
            number = r["driver_number"]
            team = teams.get(number)
            gap, count = race_pace.get(number, (None, 0))
            self.driver_pace[number].append(gap)
            if isinstance(r.get("dnf"), bool):
                self.driver_dnfs[number].append(int(r["dnf"]))
                if team:
                    team_dnfs[team].append(int(r["dnf"]))
            if team:
                team_laps[team].append(gap)
        for team, gaps in team_laps.items():
            observed = [gap for gap in gaps if gap is not None]
            self.team_pace[team].append(fmean(observed) if observed else None)
        for team, dnfs in team_dnfs.items():
            self.team_dnfs[team].append(fmean(dnfs))


def build_dataset():
    api = OpenF1(BASE)
    seasons = {}
    historical = []
    audit_rows = []
    skipped = []
    for year, race, sessions, target in weeks(api):
        # Form is season-specific so identity and regulation changes do not silently carry over.
        if year not in seasons:
            print(f"Cleaning laps and constructing {year} features...", flush=True)
            seasons[year] = Form()
        form = seasons[year]
        key = race["meeting_key"]
        field, results = weekend(sessions, api, True, form.driver_finish, form.team_finish, target=target)
        practice, race_pace, diagnostics = meeting_pace(sessions, key, target=target)
        for session, audit in diagnostics.items():
            audit_rows.append({"year": year, "meeting_key": key, "session": session, **audit})
        form.enrich(field, practice)
        if target:
            return historical, field, audit_rows, skipped
        winners = [r["driver_number"] for r in results if clean_finish(r, len(results)) == 1]
        if field and len(winners) == 1 and any(r["driver_number"] == winners[0] for r in field):
            historical.append((key, race["location"], year, results, field, winners[0]))
        else:
            skipped.append({"year": year, "race": race["location"], "meeting_key": key})
        # Pace and reliability for this race become features only for later races.
        form.observe(results, field, race_pace)
    raise ValueError("Target race did not appear.")


def calibrate(probs, scale):
    scores = [scale * math.log(max(EPS, p)) for p in probs]
    max_score = max(scores)
    weights = [math.exp(s - max_score) for s in scores]
    total = sum(weights)
    return [w / total for w in weights]


def choose_scale(records):
    """Fit inverse temperature to earlier OOF races with mild shrinkage toward 1."""
    if len(records) < MIN_CAL_RACES:
        return 1.0
    def grad(scale):
        gradients = []
        for probs, winner in records:
            logs = [math.log(max(EPS, p)) for p in probs]
            shifted = calibrate(probs, scale)
            gradients.append(sum(p * z for p, z in zip(shifted, logs)) - logs[winner])
        return fmean(gradients) + 0.03 * (scale - 1)
    low, high = 0.4, 2.5
    for _ in range(35):
        mid = (low + high) / 2
        if grad(mid) > 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2


def raw_predictions(races, model):
    for _, _, _, _, field, winner in races:
        index = next(i for i, driver in enumerate(field) if driver["driver_number"] == winner)
        yield predict(model, field, len(field)), index


def choose_penalty(history, columns):
    dev, validation = history[:-BLOCK], history[-BLOCK:]
    best = (math.inf, None)
    for penalty in PENALTIES:
        model = train(dev, columns, penalty, "winner")
        loss = fmean(-math.log(max(EPS, p[i])) for p, i in raw_predictions(validation, model))
        if loss < best[0]:
            best = loss, penalty
    return best[1]


def rolling(history, variants=VARIANTS):
    prior = defaultdict(list)
    scored = []
    for start in range(INITIAL_TRAIN, len(history), BLOCK):
        prefix, test = history[:start], history[start:start + BLOCK]
        include = start >= INITIAL_TRAIN + MIN_CAL_RACES
        print(f"Rolling block {start+1}–{start+len(test)} of {len(history)}"
              f" ({'scored' if include else 'calibration warm-up'})", flush=True)
        block = [{"meeting_key": r[0], "location": r[1], "year": r[2], "block": start,
                  "scored": include, "field": r[4], "winner": r[5], "models": {}} for r in test]
        for name, columns in variants.items():
            scale = choose_scale(prior[name])
            penalty = choose_penalty(prefix, columns)
            model = train(prefix, columns, penalty, "winner")
            batch = list(raw_predictions(test, model))
            for row, (p, i) in zip(block, batch):
                row["models"][name] = {"raw": p, "calibrated": calibrate(p, scale), "i": i,
                                       "temperature": 1 / scale, "penalty": penalty}
            prior[name].extend(batch)
        if include:
            scored.extend(block)
    return scored, prior


def metrics(scored, variants=VARIANTS):
    totals, bins = {}, []
    for name in variants:
        for kind in ("raw", "calibrated"):
            label = f"{name}_{kind}"
            races = [(r["models"][name][kind], r["models"][name]["i"]) for r in scored]
            totals[label] = {
                "races": len(races),
                "top_pick_hits": sum(max(range(len(p)), key=p.__getitem__) == i for p, i in races),
                "log_loss": fmean(-math.log(max(EPS, p[i])) for p, i in races),
                "brier": fmean(sum((q - (j == i)) ** 2 for j, q in enumerate(p)) for p, i in races),
            }
            favorites = [(max(p), max(range(len(p)), key=p.__getitem__) == i) for p, i in races]
            for low, high in ((0, .25), (.25, .4), (.4, .55), (.55, .7), (.7, 1.01)):
                group = [(p, win) for p, win in favorites if low <= p < high]
                if group:
                    bins.append((label, low, min(high, 1), len(group),
                                 fmean(p for p, _ in group), fmean(win for _, win in group)))
    return totals, bins


def paired_interval(scored, reference, variant="enriched", kind="calibrated"):
    """Bootstrap contiguous test folds and report variant minus reference log loss."""
    groups = defaultdict(list)
    for row in scored:
        v2 = row["models"][variant]
        base = row["models"][reference]
        delta = -math.log(max(EPS, v2[kind][v2["i"]])) + math.log(
            max(EPS, base[kind][base["i"]])
        )
        groups[row["block"]].append(delta)
    blocks = list(groups.values())
    rng = random.Random(34)
    samples = sorted(fmean(x for group in rng.choices(blocks, k=len(blocks)) for x in group)
                     for _ in range(3000))
    return {"folds": len(blocks), "mean_difference": fmean(x for group in blocks for x in group),
            "bootstrap_95": [samples[75], samples[2924]]}


def save(history, field, audit, skipped, scored, prior):
    OUT.mkdir(exist_ok=True, parents=True)
    totals, bins = metrics(scored)
    contrasts = {k: paired_interval(scored, k) for k in ("current", "grid_only")}
    split_metrics = {str(y): metrics([r for r in scored if r["year"] == y])[0]
                     for y in sorted({r["year"] for r in scored})}
    data = {"history_by_year": dict(sorted(Counter(r[2] for r in history).items())),
            "skipped_races": skipped, "test_races": len(scored),
            "first_test": [scored[0]["year"], scored[0]["location"]],
            "last_test": [scored[-1]["year"], scored[-1]["location"]],
            "overall": totals, "by_season": split_metrics, "paired_log_loss": contrasts}
    (OUT / "rolling_metrics.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    write_csv(OUT / "calibration_bins.csv", ("model", "probability_low", "probability_high",
                                            "favorite_count", "mean_probability", "observed_win_rate"), bins)
    write_csv(OUT / "timing_coverage.csv",
              ("year", "meeting_key", "session", "raw_laps", "clean_laps", "matched_laps",
               "drivers_with_pace", "control_windows", "identified_deleted_laps", "sessions_without_required_data"),
              [tuple(row.get(k, 0) for k in ("year", "meeting_key", "session", "raw_laps", "clean_laps",
                                            "matched_laps", "drivers_with_pace", "control_windows",
                                            "sessions_without_required_data")) for row in audit])
    write_csv(OUT / "training_features.csv",
              ("meeting_key", "year", "race", "driver_number", "constructor", "winner",
               "matched_practice_laps", *ENRICHED),
              [(key, year, loc, row["driver_number"], row["constructor"], int(row["driver_number"] == winner),
                row["matched_practice_laps"], *(row[f] for f in ENRICHED))
               for key, loc, year, _, rows, winner in history for row in rows])
    write_csv(OUT / "rolling_predictions.csv",
              ("meeting_key", "year", "race", "fold_start", "driver_number", "winner",
               "variant", "probability_raw", "probability_calibrated", "temperature"),
              [(row["meeting_key"], row["year"], row["location"], row["block"], d["driver_number"],
                int(d["driver_number"] == row["winner"]), name, entry["raw"][i], entry["calibrated"][i],
                entry["temperature"])
               for row in scored for name, entry in row["models"].items() for i, d in enumerate(row["field"])])
    models = {}
    probabilities = {}
    uncalibrated = {}
    for name, columns in VARIANTS.items():
        model = train(history, columns, choose_penalty(history, columns), "winner")
        scale = choose_scale(prior[name])
        model.update(temperature=1 / scale, inverse_temperature=scale,
                     calibration_races=len(prior[name]), training_races=len(history))
        models[name] = model
        uncalibrated[name] = predict(model, field, len(field))
        probabilities[name] = calibrate(uncalibrated[name], scale)
    (OUT / "trained_models.json").write_text(json.dumps(models, indent=2), encoding="utf-8")
    order = sorted(range(len(field)), key=probabilities["enriched"].__getitem__, reverse=True)
    write_csv(OUT / "vegas_2026_v2_forecast.csv",
              ("driver", "driver_number", "constructor", "starting_grid", "matched_practice_laps",
               *(f"{name}_calibrated" for name in VARIANTS), *(f"{name}_raw" for name in VARIANTS), *ENRICHED[len(FEATURES):]),
              [(field[i]["driver"], field[i]["driver_number"], field[i]["constructor"],
                field[i]["starting_grid"], field[i]["matched_practice_laps"],
                *(probabilities[name][i] for name in VARIANTS),
                *(uncalibrated[name][i] for name in VARIANTS),
                *(field[i][f] for f in ENRICHED[len(FEATURES):])) for i in order])
    write_csv(OUT / "coefficient_table.csv",
              ("variant", "feature", "relative_log_odds_per_sd", "calibrated_relative_log_odds_per_sd"),
              [(name, feature, coefficient, coefficient * model["inverse_temperature"])
               for name, model in models.items() for feature, coefficient in zip(model["features"], model["weights"])])
    print(json.dumps(data, indent=2))
    for i in order[:5]:
        print(f"{field[i]['driver']}: {probabilities['enriched'][i]:.2%}")


def main():
    history, target, audit, skipped = load_frozen()
    if len(history) < INITIAL_TRAIN + MIN_CAL_RACES + BLOCK:
        raise ValueError("Too few complete races for chronological calibration and testing.")
    scored, prior = rolling(history)
    save(history, target, audit, skipped, scored, prior)


if __name__ == "__main__":
    main()
