"""Fit a retirement model and simulate a coherent joint finishing order."""
import json
import math
import random
from collections import defaultdict
from statistics import fmean, pstdev

from ..v2.training import BLOCK, ENRICHED, INITIAL_TRAIN, MIN_CAL_RACES
from ..baselines.train_post_qualifying import clean_finish, write_csv
from ..baselines.vegas_finish import fit as ridge_fit

from ..paths import ARTIFACTS, RUNS
from ..snapshot import load_frozen

OUT = RUNS / "simulation"
PACE = ENRICHED[:-2]
RETIRE = ("driver_dnf_rate", "constructor_dnf_rate", "grid_log_position")
EVENTS = ("podium", "points", "dnf")
BACKTEST_DRAWS = 5000
FORECAST_DRAWS = 50000
MODEL_VERSION = "1.0"


def logistic(x):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def eligible(race):
    size = len(race[3])
    entries = {r["driver_number"]: r for r in race[3] if not (r.get("dns") or r.get("dsq"))}
    return [(row, entries[row["driver_number"]], size)
            for row in race[4] if row["driver_number"] in entries]


def fit_retirement(history):
    records = [(r, int(outcome["dnf"])) for race in history
               for r, outcome, _ in eligible(race) if isinstance(outcome.get("dnf"), bool)]
    inputs = [[r[key] for key in RETIRE] for r, _ in records]
    means = [fmean(col) for col in zip(*inputs)]
    scales = [pstdev(col) or 1.0 for col in zip(*inputs)]
    normalized = [[(x - m) / s for x, m, s in zip(row, means, scales)] for row in inputs]
    targets = [y for _, y in records]
    pooled = (sum(targets) + 1) / (len(targets) + 2)
    intercept, weights = math.log(pooled / (1 - pooled)), [0.0] * len(RETIRE)
    for _ in range(200):
        delta = [0.0, *[0.08 * weight for weight in weights]]
        for row, label in zip(normalized, targets):
            err = (logistic(intercept + sum(w*x for w, x in zip(weights, row))) - label) / len(targets)
            delta[0] += err
            for i, value in enumerate(row, 1):
                delta[i] += err * value
        intercept -= 0.6 * delta[0]
        weights = [w - 0.6 * g for w, g in zip(weights, delta[1:])]
        if max(map(abs, delta)) < 1e-6:
            break
    return {"intercept": intercept, "means": means, "scales": scales, "weights": weights,
            "features": RETIRE, "pooled_rate": pooled, "starter_observations": len(targets),
            "retirements": sum(targets), "l2_penalty": 0.08}


def retirement_probs(model, field):
    return [logistic(model["intercept"] + sum(w * (r[k] - m) / s for k, w, m, s
            in zip(RETIRE, model["weights"], model["means"], model["scales"]))) for r in field]


def fit_pace(history):
    batches = []
    for race in history:
        scored = []
        for row, outcome, size in eligible(race):
            pos = clean_finish(outcome, size)
            if pos is not None and outcome.get("dnf") is False:
                scored.append((tuple(row[k] for k in PACE), (pos - 1) / (size - 1), size))
        batches.append((scored, len(race[4])))
    intercept, means, scales, weights = ridge_fit(batches, 0.08)
    return {"intercept": intercept, "means": means, "scales": scales, "weights": weights,
            "features": PACE, "l2_penalty": 0.08}


def pace_scores(model, field):
    return [model["intercept"] + sum(w * (r[k] - m) / s for k, w, m, s in
            zip(PACE, model["weights"], model["means"], model["scales"])) for r in field]


def grid_scores(field):
    return [min(1.0, (row["starting_grid"]-1)/(len(field)-1)) for row in field]


def grid_band(row):
    return min(2, (row["starting_grid"] - 1) // 7)


def record_residuals(bank, race, field, means):
    for (row, outcome, size) in eligible(race):
        pos = clean_finish(outcome, size)
        if pos is not None and outcome.get("dnf") is False:
            index = field[row["driver_number"]]
            bank[grid_band(row)].append((pos - 1) / (size - 1) - means[index])


def residual_pools(bank):
    pooled = [v for block in bank.values() for v in block]
    if not pooled:
        raise ValueError("Need earlier out-of-sample residuals for simulation.")
    return [bank[k] if len(bank[k]) >= 24 else pooled for k in range(3)]


def simulate(field, pace, retirement, pools, draws, seed):
    """Draw DNF indicators, then sample race-pace residuals and rank all finishers."""
    rng = random.Random(seed)
    uniform = rng.random
    size = len(field)
    counts = [[0] * (size + 1) for row in field]
    by_driver = [pools[grid_band(row)] for row in field]
    for _ in range(draws):
        finishers = []
        for i, (score, dnf, residuals) in enumerate(zip(pace, retirement, by_driver)):
            retired, u, tie_break = uniform(), uniform(), uniform()
            if retired < dnf:
                counts[i][-1] += 1
            else:
                finishers.append((score + residuals[int(u * len(residuals))], tie_break, i))
        finishers.sort()
        for rank, (_, _, i) in enumerate(finishers):
            counts[i][rank] += 1
    if not all(sum(row) == draws for row in counts):
        raise ValueError("Each driver must have exactly one result in each simulated race.")
    if any(sum(row[i] for row in counts) > draws for i in range(size)):
        raise ValueError("A finishing position was awarded twice within a simulation.")
    return [[value / draws for value in row] for row in counts]


def hit_labels(outcome, size):
    pos = clean_finish(outcome, size)
    if not isinstance(outcome.get("dnf"), bool):
        return None
    return int(pos is not None and pos <= 3), int(pos is not None and pos <= 10), int(outcome["dnf"])


def event_values(matrix, dnf_probability):
    return [(sum(row[:3]), sum(row[:10]), dnf) for row, dnf in zip(matrix, dnf_probability)]


def rolling(history):
    banks = {name: defaultdict(list) for name in ("model", "grid")}
    records = []
    scored_keys = []
    for start in range(INITIAL_TRAIN, len(history), BLOCK):
        past = history[:start]
        test = history[start:start+BLOCK]
        scored = start >= INITIAL_TRAIN + MIN_CAL_RACES
        print(f"Rolling simulation {start+1}–{start+len(test)} / {len(history)} "
              f"({'scored' if scored else 'residual warm-up'})", flush=True)
        pace_model = fit_pace(past)
        dnf_model = fit_retirement(past)
        # Freeze both residual banks before the entire block is predicted.
        pools = {name: residual_pools(bank) for name, bank in banks.items()} if scored else {}
        updates = []
        for race in test:
            key, name, season, _, field, _ = race
            identifiers = {r["driver_number"]: i for i, r in enumerate(field)}
            pace = pace_scores(pace_model, field)
            grid = grid_scores(field)
            if scored:
                scored_keys.append(key)
                dnf = retirement_probs(dnf_model, field)
                pooled = [dnf_model["pooled_rate"]] * len(field)
                seed = 20260000 + key
                full = event_values(simulate(field, pace, dnf, pools["model"], BACKTEST_DRAWS, seed), dnf)
                baseline = event_values(simulate(field, grid, pooled, pools["grid"], BACKTEST_DRAWS, seed), pooled)
                for row, outcome, size in eligible(race):
                    labels = hit_labels(outcome, size)
                    if labels is not None:
                        i = identifiers[row["driver_number"]]
                        records.append((key, name, season, start, row["driver_number"], labels, full[i], baseline[i]))
            updates.append((race, identifiers, pace, grid))
        for race, identifiers, pace, grid in updates:
            record_residuals(banks["model"], race, identifiers, pace)
            record_residuals(banks["grid"], race, identifiers, grid)
    return records, banks, scored_keys


def score(records):
    groups = defaultdict(list)
    for row in records:
        groups[row[0]].append(row)
    summary = {}
    calibration = []
    race_scores = []
    for i, event in enumerate(EVENTS):
        summary[event] = {}
        for column, name in ((6,"model"),(7,"baseline")):
            errors = []
            pool = []
            for key, rows in groups.items():
                target = [(r[5][i], r[column][i]) for r in rows]
                pool.extend(target)
                brier = fmean((p-y)**2 for y,p in target)
                def penalty(y,p):
                    probability = p if y else 1-p
                    return -math.log(max(1/(BACKTEST_DRAWS+2), probability))
                loss = fmean(penalty(y,p) for y,p in target)
                errors.append((brier,loss))
                race_scores.append((key,rows[0][1],rows[0][2],event,name,len(rows),brier,loss))
            summary[event][name] = {"brier": fmean(a for a,_ in errors), "log_loss": fmean(b for _,b in errors),
                                    "observed_rate": fmean(y for y,_ in pool), "mean_probability": fmean(p for _,p in pool)}
            bins = (0, .05, .10, .20, .35, .50, .70, .90, 1.01) if event=="dnf" else (0,.10,.25,.50,.75,.90,1.01)
            for a,b in zip(bins,bins[1:]):
                bucket = [(y,p) for y,p in pool if a<=p<b]
                if bucket:
                    calibration.append((event,name,a,min(b,1),len(bucket),fmean(p for _,p in bucket),fmean(y for y,_ in bucket)))
        summary[event]["brier_delta"] = summary[event]["model"]["brier"] - summary[event]["baseline"]["brier"]
    return summary, calibration, race_scores


def main():
    history, field, _, _ = load_frozen()
    scored, banks, keys = rolling(history)
    expected = json.loads((ARTIFACTS / "feature_comparison" / "comparison.json").read_text(encoding="utf-8"))["meeting_keys"]
    if keys != expected:
        raise ValueError("Simulation backtest differs from the archived 52-race test set.")
    stats, calibration, race_scores = score(scored)
    pace = fit_pace(history)
    retirement = fit_retirement(history)
    pools = residual_pools(banks["model"])
    dns = retirement_probs(retirement, field)
    matrix = simulate(field, pace_scores(pace,field), dns, pools, FORECAST_DRAWS, 217051)
    if not math.isclose(sum(row[0] for row in matrix),1):
        raise ValueError("Vegas forecast must assign exactly one winner in all draws.")
    size = len(field)
    ordered = sorted(range(size), key=lambda i: sum((j+1)*p for j,p in enumerate(matrix[i][:-1]))/(1-matrix[i][-1]))
    OUT.mkdir(parents=True, exist_ok=True)
    rows=[]
    for i in ordered:
        d=field[i]
        p=matrix[i]
        avg=sum((j+1)*value for j,value in enumerate(p[:-1]))/(1-p[-1])
        rows.append((d["driver_number"],d["driver"],d["constructor"],d["starting_grid"],avg,
                     sum(p[:3]),sum(p[:10]),dns[i],*p))
    write_csv(OUT/"vegas_2026_finish_distribution.csv",
              ("driver_number","driver","constructor","grid","expected_finish_if_no_dnf","podium","points_top_10",
               "dnf_model",*(f"P{i}" for i in range(1,size+1)),"DNF"), rows)
    write_csv(OUT/"rolling_event_predictions.csv",
              ("meeting_key","race","year","fold_start","driver_number",
               *(f"actual_{e}" for e in EVENTS),*(f"model_{e}" for e in EVENTS),*(f"baseline_{e}" for e in EVENTS)),
              [(key,name,season,start,num,*labels,*m,*b) for key,name,season,start,num,labels,m,b in scored])
    write_csv(OUT/"race_metrics.csv",("meeting_key","race","year","event","variant","scored_drivers","brier","log_loss"),race_scores)
    write_csv(OUT/"calibration_bins.csv",("event","variant","bin_low","bin_high","observations","mean_probability","observed_rate"),calibration)
    (OUT/"trained_parameters.json").write_text(json.dumps({"version":MODEL_VERSION,"pace":pace,"retirement":retirement,
                 "residual_pools":pools,"training_races":len(history)},indent=2),encoding="utf-8")
    report={"version":MODEL_VERSION,"training_races":len(history),"test_races":len(keys),"test_drivers":len(scored),
            "backtest_draws":BACKTEST_DRAWS,"forecast_draws":FORECAST_DRAWS,"metrics":stats,"meeting_keys":keys,
            "labels":{"podium":"P1–P3 on official numeric race classification","points":"P1–P10 points-finish proxy",
                      "dnf":"OpenF1 DNF flag for eligible starters"}}
    (OUT/"rolling_metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
