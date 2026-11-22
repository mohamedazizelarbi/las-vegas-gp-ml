"""Train and audit the 2026 Las Vegas post-qualifying forecast (Python 3.10+)."""
import argparse
import csv
import json
import math
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from urllib.error import HTTPError, URLError

from .vegas_finish import fit as ridge_fit
from .vegas_finish import write_csv
from .vegas_predictor import OpenF1, probabilities, timestamp

from ..paths import CACHE, DATA, RUNS


FEATURES = ("grid_log_position", "qualifying_rank_fraction", "qualifying_gap_pct",
            "practice_gap_pct", "practice_missing", "driver_form", "constructor_form")
PENALTIES = (0.01, 0.05, 0.2, 1.0)
LAS_VEGAS_GRID = "https://www.formula1.com/en/results/2026/races/1300/las-vegas/starting-grid"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def fetch(api, offline, endpoint, **filters):
    key = "_".join(f"{k}={v}" for k, v in filters.items())
    filename = api.cache / f"{endpoint}_{key}.json"
    if offline and not filename.is_file():
        raise ValueError(f"Missing cached input {filename}. Run without --offline to download it.")
    try:
        return api.get(endpoint, **filters)
    except HTTPError as err:
        if err.code != 404 or endpoint == "sessions":
            raise
        return []


def positive(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def index(rows):
    return {row["driver_number"]: row for row in rows}


def clean_finish(row, size):
    pos = row.get("position")
    if not row.get("dns") and not row.get("dsq") and positive(pos) and pos <= size:
        return pos
    return None


def smoothed(values):
    return (sum(values) + 1) / (len(values) + 2)


def features_for(quali, practice, grid, drivers, driver_history, team_history):
    phase_times = {}
    for row in quali:
        duration = row.get("duration")
        values = duration if isinstance(duration, list) else [duration]
        phase_times[row["driver_number"]] = [v if positive(v) else None for v in values]
    leaders = {i: min(slots[i] for slots in phase_times.values() if len(slots) > i and slots[i])
               for i in range(3) if any(len(slots) > i and slots[i] for slots in phase_times.values())}
    practice_by_driver = defaultdict(list)
    practice_raw = defaultdict(dict)
    for name, rows in practice.items():
        laps = {r["driver_number"]: r["duration"] for r in rows if positive(r.get("duration"))}
        if len(laps) < 10:
            continue
        best = min(laps.values())
        for number, duration in laps.items():
            practice_raw[number][name] = duration
            practice_by_driver[number].append(min(5, (duration / best - 1) * 100))
    grid_by_driver = index(grid)
    driver_details = index(drivers)
    field = []
    size = len(quali)
    for row in quali:
        number = row["driver_number"]
        driver = driver_details.get(number, {})
        team = driver.get("team_name") or ""
        raw_grid = grid_by_driver.get(number, {}).get("position")
        start = raw_grid if positive(raw_grid) else size + 1
        rank = row.get("position")
        times = phase_times[number]
        phase = next((i for i in reversed(range(len(times))) if times[i] and i in leaders), None)
        gap = min(5, (times[phase] / leaders[phase] - 1) * 100) if phase is not None else 5
        practice_gaps = practice_by_driver[number]
        data = {
            "driver_number": number, "driver": driver.get("full_name") or f"Driver #{number}",
            "constructor": team, "starting_grid": start, "qualifying_position": rank,
            "q1_seconds": times[0] if times else None,
            "q2_seconds": times[1] if len(times) > 1 else None,
            "q3_seconds": times[2] if len(times) > 2 else None,
            "fp1_seconds": practice_raw[number].get("Practice 1"),
            "fp2_seconds": practice_raw[number].get("Practice 2"),
            "fp3_seconds": practice_raw[number].get("Practice 3"),
            "grid_log_position": math.log(start),
            "qualifying_rank_fraction": (rank - 1) / (size - 1) if positive(rank) else 1,
            "qualifying_gap_pct": gap,
            "practice_gap_pct": fmean(practice_gaps) if practice_gaps else 0,
            "practice_missing": float(not practice_gaps),
            "driver_form": smoothed(driver_history[number]),
            "constructor_form": smoothed(team_history[team]) if team else 0.5,
        }
        field.append(data)
    return field


def weekend(sessions, api, offline, driver_history, team_history, *, target=False):
    names = {s["session_name"]: s for s in sessions if not s.get("is_cancelled")}
    race, quali = names.get("Race"), names.get("Qualifying")
    if not race or not quali:
        return [], []
    q_key = quali["session_key"]
    races = []
    practice_sessions = [name for name in ("Practice 1", "Practice 2", "Practice 3")
                         if name in names and timestamp(names[name], "date_end") <= timestamp(race, "date_start")]
    if target:
        def get(endpoint, key):
            return fetch(api, offline, endpoint, session_key=key)
        qualifying = get("session_result", q_key)
        drivers = get("drivers", q_key)
        practice = {name: get("session_result", names[name]["session_key"]) for name in practice_sessions}
        if not qualifying or not drivers:
            raise ValueError("Vegas qualifying classification or driver metadata is missing.")
        # Cross-check the feed against the published grid snapshot.
        official_grid = [{"driver_number": int(row["driver_number"]), "position": int(row["position"])}
                for row in read_csv(DATA / "vegas_2026_official_grid.csv")]
        openf1_grid = get("starting_grid", q_key)
        official_order = {row["driver_number"]: row["position"] for row in official_grid}
        provider_order = {row["driver_number"]: row["position"] for row in openf1_grid}
        if openf1_grid and official_order != provider_order:
            print("OpenF1 grid differs from Formula 1 snapshot. Using the official grid.", flush=True)
        grid = official_grid
        if len(grid) != len(qualifying) or set(index(grid)) != set(index(qualifying)):
            raise ValueError("Official grid driver numbers do not match the qualifying field.")
    else:
        key = race["meeting_key"]
        results = defaultdict(list)
        for row in fetch(api, offline, "session_result", meeting_key=key):
            results[row["session_key"]].append(row)
        qualifying = results[q_key]
        races = results[race["session_key"]]
        drivers = [r for r in fetch(api, offline, "drivers", meeting_key=key)
                   if r["session_key"] == q_key]
        grid = [r for r in fetch(api, offline, "starting_grid", meeting_key=key)
                if r["session_key"] == q_key]
        practice = {name: results[names[name]["session_key"]] for name in practice_sessions}
    if len(qualifying) < 15 or len(grid) < len(qualifying) - 3 or len(drivers) < len(qualifying) - 2:
        if target:
            raise ValueError("Vegas feature coverage is incomplete.")
        field = []
    else:
        field = features_for(qualifying, practice, grid, drivers, driver_history, team_history)
    detail = index(drivers)
    if races:
        actual = {r["driver_number"]: clean_finish(r, len(races)) for r in races}
        for data in field:
            data["actual_finish"] = actual.get(data["driver_number"])
        # Advance form only after the forecast features for this race are built.
        team_race = defaultdict(list)
        for number, finish in actual.items():
            score = (finish - 1) / (len(races) - 1) if finish else 1
            driver_history[number].append(score)
            team = detail.get(number, {}).get("team_name")
            if team:
                team_race[team].append(score)
        for team, scores in team_race.items():
            team_history[team].append(fmean(scores))
    return field, races


def dataset(api, offline, first_year):
    history = []
    skipped = []
    for season in range(first_year, 2027):
        by_meeting = defaultdict(list)
        for s in fetch(api, offline, "sessions", year=season):
            by_meeting[s["meeting_key"]].append(s)
        ordered = sorted((timestamp(s, "date_start"), s)
                         for sessions in by_meeting.values() for s in sessions
                         if s["session_name"] == "Race" and not s.get("is_cancelled"))
        driver_history = defaultdict(lambda: deque(maxlen=5))
        team_history = defaultdict(lambda: deque(maxlen=5))
        print(f"Preparing {season} features...", flush=True)
        for start, race in ordered:
            target = season == 2026 and race["location"] == "Las Vegas"
            sessions = by_meeting[race["meeting_key"]]
            if target:
                now = datetime.now(timezone.utc)
                if now < max(timestamp(s, "date_end") for s in sessions if s["session_name"] == "Qualifying"):
                    raise ValueError("Vegas qualifying is still pending.")
                field, _ = weekend(sessions, api, offline, driver_history, team_history, target=True)
                return history, field, skipped
            field, results = weekend(sessions, api, offline, driver_history, team_history)
            winners = [r["driver_number"] for r in results if clean_finish(r, len(results)) == 1]
            if field and len(winners) == 1 and winners[0] in {r["driver_number"] for r in field}:
                history.append((race["meeting_key"], race["location"], season, results, field, winners[0]))
            else:
                skipped.append(f"{season} {race['location']}")
    raise ValueError("Las Vegas was missing from the 2026 schedule.")


def design(races, names):
    return [[[row[name] for name in names] for row in race[4]] for race in races]


def train(races, names, penalty, task):
    design_rows = design(races, names)
    columns = list(zip(*(row for field in design_rows for row in field)))
    means = [fmean(c) for c in columns]
    scales = [pstdev(c) or 1.0 for c in columns]
    if task == "finish":
        targets = [
            ([(tuple(x), (r["actual_finish"] - 1) / (len(race[3]) - 1), len(race[3]))
              for x, r in zip(xs, race[4]) if r["actual_finish"] is not None], len(race[4]))
            for xs, race in zip(design_rows, races)
        ]
        intercept, means, scales, weights = ridge_fit(targets, penalty)
    else:
        fields = [
            ([(r["driver_number"], r["starting_grid"],
               [(x - m) / s for x, m, s in zip(values, means, scales)])
              for r, values in zip(race[4], xs)], race[5])
            for race, xs in zip(races, design_rows)
        ]
        weights = [0.0] * len(names)
        intercept = 0.0
        for _ in range(400):
            gradient = [penalty * w for w in weights]
            for field, winner in fields:
                for (number, _, values), prob in zip(field, probabilities(field, weights)):
                    err = (prob - (number == winner)) / len(fields)
                    for i, x in enumerate(values):
                        gradient[i] += err * x
            weights = [w - 0.1 * g for w, g in zip(weights, gradient)]
            if max(map(abs, gradient)) < 1e-6:
                break
    name = "L2-regularized conditional logistic regression" if task == "winner" else "Ridge regression"
    return {"name": name, "task": task, "features": names, "penalty": penalty,
            "means": means, "scales": scales, "weights": weights, "intercept": intercept}


def predict(model, field, size):
    transformed = [
        (r["driver_number"], r["starting_grid"],
         [(r[k] - m) / s for k, m, s in zip(model["features"], model["means"], model["scales"])])
        for r in field
    ]
    if model["task"] == "winner":
        return probabilities(transformed, model["weights"])
    return [
        1 + (size - 1) * min(1, max(0, model["intercept"] +
                                    sum(w * x for w, x in zip(model["weights"], values))))
        for _, _, values in transformed
    ]


def score(races, model):
    loss, absolute, squared, hits, total = 0, 0, 0, 0, 0
    for _, _, _, results, field, winner in races:
        size = len(results)
        if model is None:
            predicted = [min(size, r["starting_grid"]) for r in field]
        else:
            predicted = predict(model, field, size)
        if model and model["task"] == "winner":
            top = max(range(len(field)), key=predicted.__getitem__)
            hits += field[top]["driver_number"] == winner
            win_index = next(i for i, r in enumerate(field) if r["driver_number"] == winner)
            loss -= math.log(predicted[win_index])
        else:
            for r, value in zip(field, predicted):
                if r["actual_finish"] is not None:
                    err = value - r["actual_finish"]
                    absolute += abs(err)
                    squared += err * err
                    total += 1
    if model and model["task"] == "winner":
        return {"top_pick_correct": hits, "races": len(races), "winner_log_loss": loss / len(races)}
    return {"classified_entries": total, "mae": absolute / total, "rmse": math.sqrt(squared / total)}


def model_suite(history):
    # Hold out the last 12 races. Choose regularization on a preceding six-race block.
    train_set, test_set = history[:-12], history[-12:]
    development, validation = train_set[:-6], train_set[-6:]
    variants = {"full": FEATURES, "grid_only": FEATURES[:1],
                "without_practice": FEATURES[:3] + FEATURES[5:],
                "without_constructor": FEATURES[:-1]}
    evaluation, fitted = {}, {}
    for task, metric in (("winner", "winner_log_loss"), ("finish", "mae")):
        evaluation[task] = {}
        for variant, columns in variants.items():
            print(f"Fitting {task} / {variant}...", flush=True)
            best = min(PENALTIES, key=lambda p: score(validation, train(development, columns, p, task))[metric])
            evaluation[task][variant] = score(test_set, train(train_set, columns, best, task))
            if variant == "full":
                fitted[task] = train(history, columns, best, task)
    evaluation["finish"]["raw_grid"] = score(test_set, None)
    evaluation["pole_sitter_correct"] = sum(
        min(race[4], key=lambda r: r["starting_grid"])["driver_number"] == race[5] for race in test_set
    )
    evaluation["training_races"] = len(train_set)
    evaluation["holdout_races"] = [{"season": r[2], "race": r[1], "meeting_key": r[0]} for r in test_set]
    return fitted, evaluation


def write_outputs(directory, history, field, models, metrics, skipped):
    directory.mkdir(exist_ok=True, parents=True)
    probs = predict(models["winner"], field, len(field))
    finishes = predict(models["finish"], field, len(field))
    order = sorted(range(len(field)), key=lambda i: finishes[i])
    forecast_columns = ("projected_order", "estimated_finish", "win_probability", *field[0])
    write_csv(directory / "vegas_2026_post_qualifying.csv", forecast_columns,
              [(rank, finishes[i], probs[i], *field[i].values()) for rank, i in enumerate(order, 1)])
    training_columns = ("meeting_key", "location", "season", "split", "winner", *history[0][4][0])
    write_csv(directory / "training_features.csv", training_columns,
              [(key, loc, year, "holdout" if i >= len(history) - 12 else "train",
                int(row["driver_number"] == winner), *row.values())
               for i, (key, loc, year, _, entries, winner) in enumerate(history) for row in entries])
    write_csv(directory / "coefficients.csv", ("task", "feature", "coefficient_per_sd", "unit"),
              [(task, name, weight * (len(field) - 1) if task == "finish" else weight,
                "finishing positions" if task == "finish" else "relative log odds")
               for task, model in models.items() for name, weight in zip(model["features"], model["weights"])])
    (directory / "trained_models.json").write_text(json.dumps(models, indent=2), encoding="utf-8")
    metrics["refit_races"] = len(history)
    metrics["skipped_races"] = skipped
    metrics["target_grid_source"] = LAS_VEGAS_GRID
    metrics["openf1_target_sessions"] = {"fp2": 11414, "fp3": 11415, "qualifying": 11416}
    (directory / "backtest.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for i in sorted(range(len(field)), key=lambda i: -probs[i])[:5]:
        print(f"{field[i]['driver']:25} win {probs[i]:6.2%}  expected finish {finishes[i]:5.2f}")
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output-dir", type=Path, default=RUNS / "post_qualifying")
    parser.add_argument("--first-year", type=int, choices=range(2023, 2027), default=2025)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    try:
        history, target, skipped = dataset(OpenF1(args.cache), args.offline, args.first_year)
        if len(history) < 28:
            raise ValueError(f"Only {len(history)} usable races. Need 28 or more.")
        models, metrics = model_suite(history)
        write_outputs(args.output_dir, history, target, models, metrics, skipped)
    except (OSError, ValueError, URLError) as err:
        parser.exit(1, f"Error: {err}\n")


if __name__ == "__main__":
    main()
