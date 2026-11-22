"""Replay a DNF intercept recalibration using only earlier out-of-sample races."""
import json
import math
from collections import defaultdict, deque

from .model import (BACKTEST_DRAWS, BLOCK, INITIAL_TRAIN, MIN_CAL_RACES,
                   eligible, event_values, fit_pace, fit_retirement, hit_labels, logistic,
                   pace_scores, record_residuals, residual_pools, retirement_probs, score,
                   simulate, write_csv)

from ..paths import ARTIFACTS, RUNS
from ..snapshot import load_frozen

OUT = RUNS / "dnf_calibration"
RECENT_RACES = 18
ANCHOR = 80
MAX_SHIFT = 0.7


def calibrate_intercept(earlier):
    """Fit one log-odds offset with shrinkage toward the existing probabilities."""
    data = [r for race in earlier for r in race]
    def gradient(offset):
        return sum(logistic(logit + offset) - hit for logit, hit in data) + ANCHOR * offset * .12 * .88
    low, high = -MAX_SHIFT, MAX_SHIFT
    for _ in range(35):
        mid = (low + high) / 2
        if gradient(mid) > 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2


def log_odds(p):
    return math.log(p / (1 - p))


def main():
    historical, _, _, _ = load_frozen()
    archive = json.loads((ARTIFACTS / "simulation" / "rolling_metrics.json").read_text(encoding="utf8"))
    earlier = deque(maxlen=RECENT_RACES)
    residuals = defaultdict(list)
    compared, ids, shifts = [], [], []
    for start in range(INITIAL_TRAIN, len(historical), BLOCK):
        races = historical[start:start + BLOCK]
        train_set = historical[:start]
        pace = fit_pace(train_set)
        retirement = fit_retirement(train_set)
        scored = start >= INITIAL_TRAIN + MIN_CAL_RACES
        offset = calibrate_intercept(earlier) if scored else 0
        pools = residual_pools(residuals) if scored else None
        updates = []
        warm = []
        for race in races:
            key, location, year, _, field, _ = race
            if key == 1300:
                raise ValueError("Vegas cannot enter the challenger backtest.")
            ids_to_index = {r["driver_number"]: i for i, r in enumerate(field)}
            centers = pace_scores(pace, field)
            dnf = retirement_probs(retirement, field)
            logits = [log_odds(p) for p in dnf]
            pairs = [(r, hit_labels(result, size)) for r, result, size in eligible(race)]
            pairs = [(r, y) for r, y in pairs if y is not None]
            if scored:
                ids.append(key)
                revised = [logistic(z + offset) for z in logits]
                seed = 20260000 + key
                original = event_values(simulate(field, centers, dnf, pools, BACKTEST_DRAWS, seed), dnf)
                challenger = event_values(simulate(field, centers, revised, pools, BACKTEST_DRAWS, seed), revised)
                for row, y in pairs:
                    num = row["driver_number"]
                    i = ids_to_index[num]
                    compared.append((key, location, year, start, num, y, challenger[i], original[i]))
            updates.append((race, ids_to_index, centers))
            warm.append([(logits[ids_to_index[r["driver_number"]]], y[2]) for r, y in pairs])
        for race, indices, centers in updates:
            record_residuals(residuals, race, indices, centers)
        earlier.extend(warm)
        if scored:
            shifts.append({"first_test_meeting":races[0][0], "test_races":len(races), "log_odds_offset":offset})
        print(f"Checked block at historical race {start + 1} with DNF offset {offset:+.3f}", flush=True)
    if ids != archive["meeting_keys"] or len(compared) != archive["test_drivers"]:
        raise ValueError("Challenger and original evaluation cohorts differ.")
    aggregate, _, by_race = score(compared)
    for event, result in aggregate.items():
        for name, value in result["baseline"].items():
            if not math.isclose(value, archive["metrics"][event]["model"][name], abs_tol=1e-12):
                raise ValueError(f"Original score did not reproduce: {event}, {name}.")
    truth = {"test_races":len(ids), "test_drivers":len(compared), "lookback_races":RECENT_RACES,
             "anchor_entries":ANCHOR, "maximum_log_odds_shift":MAX_SHIFT, "test_offsets":shifts,
             "events":{e:{"challenger":v["model"],"frozen_model":v["baseline"]} for e,v in aggregate.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"dnf_recalibration_experiment.json").write_text(json.dumps(truth, indent=2),encoding="utf8")
    write_csv(OUT/"dnf_recalibration_race_metrics.csv",
              ("meeting_key","race","year","event","variant","scored_drivers","brier","log_loss"),
              [(k,r,y,e,"challenger" if v=="model" else "frozen_model",n,b,l)
               for k,r,y,e,v,n,b,l in by_race])
    print(json.dumps(truth["events"], indent=2))


if __name__ == "__main__":
    main()
