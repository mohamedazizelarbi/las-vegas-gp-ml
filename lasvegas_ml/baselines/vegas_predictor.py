"""Post-qualifying F1 winner model. Python 3.10+, standard library only."""
import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


class OpenF1:
    def __init__(self, cache):
        self.cache = cache
        self.next_request = 0.0

    def get(self, endpoint, *, refresh=False, **filters):
        query = urlencode(filters)
        filename = self.cache / (endpoint + "_" + query.replace("&", "_") + ".json")
        if filename.exists() and not refresh:
            return json.loads(filename.read_text(encoding="utf-8"))
        url = f"https://api.openf1.org/v1/{endpoint}?{query}"
        for attempt in range(4):
            time.sleep(max(0, self.next_request - time.monotonic()))
            self.next_request = time.monotonic() + 2.2
            try:
                with urlopen(url, timeout=20) as response:
                    rows = json.load(response)
                break
            except HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise
                delay = error.headers.get("Retry-After", "")
                time.sleep(int(delay) if delay.isdigit() else 2 ** (attempt + 2))
        if not isinstance(rows, list):
            raise ValueError(f"Unexpected response from {endpoint}.")
        if rows:
            self.cache.mkdir(parents=True, exist_ok=True)
            filename.write_text(json.dumps(rows), encoding="utf-8")
        return rows


def timestamp(session, field):
    return datetime.fromisoformat(session[field].replace("Z", "+00:00"))


def fetch_rows(api, year, circuit, first_year):
    now = datetime.now(timezone.utc)
    target_sessions = api.get("sessions", refresh=True, year=year)
    target = next(
        (s for s in target_sessions if s["session_name"] == "Race"
         and circuit.casefold() in f'{s["location"]} {s["circuit_short_name"]}'.casefold()
         and not s.get("is_cancelled")), None
    )
    if not target:
        raise ValueError(f"No race matching {circuit!r} in {year}.")
    qualifying = next(
        (s for s in target_sessions if s["meeting_key"] == target["meeting_key"]
         and s["session_name"] == "Qualifying" and not s.get("is_cancelled")), None
    )
    if not qualifying or now < timestamp(qualifying, "date_end") + timedelta(minutes=30):
        raise ValueError("Target qualifying is pending. Run after qualifying data becomes historical.")
    target_results = api.get("session_result", session_key=qualifying["session_key"])
    if len(target_results) < 10:
        raise ValueError("Target qualifying classification is missing or incomplete.")

    history = []
    form = defaultdict(lambda: deque(maxlen=5))
    cutoff = timestamp(qualifying, "date_start")
    for season in range(first_year, year + 1):
        print(f"Loading {season} race history...", file=sys.stderr, flush=True)
        sessions = target_sessions if season == year else api.get("sessions", year=season)
        qualifying_by_meeting = {
            s["meeting_key"]: s for s in sessions
            if s["session_name"] == "Qualifying" and not s.get("is_cancelled")
        }
        races = sorted(
            (s for s in sessions if s["session_name"] == "Race" and not s.get("is_cancelled")
             and timestamp(s, "date_end") < cutoff
             and timestamp(s, "date_end") < now - timedelta(minutes=30)),
            key=lambda s: timestamp(s, "date_start"),
        )
        for race in races:
            quali = qualifying_by_meeting.get(race["meeting_key"])
            if not quali:
                print(f'Missing qualifying for race {race["session_key"]}.', file=sys.stderr)
                continue
            qualifying_results = api.get("session_result", session_key=quali["session_key"])
            results = api.get("session_result", session_key=race["session_key"])
            winners = [r["driver_number"] for r in results if r["position"] == 1]
            if len(qualifying_results) >= 10 and len(results) >= 10 and len(winners) == 1:
                rows = make_field(qualifying_results, form, season)
                if any(driver == winners[0] for driver, _, _ in rows):
                    history.append((rows, winners[0], season))
                for result in results:
                    position = result["position"]
                    if position is None or result.get("dsq") or result.get("dns"):
                        position = len(results) + 1
                    form[season, result["driver_number"]].append(
                        1.0 - min(position, len(results) + 1) / (len(results) + 1)
                    )
            else:
                print(f'Skipped incomplete race {race["session_key"]}.', file=sys.stderr)

    names = api.get("drivers", session_key=qualifying["session_key"])
    names = {r["driver_number"]: r["full_name"] for r in names}
    return history, make_field(target_results, form, year), names


def make_field(results, form, year):
    rows = []
    for result in results:
        driver = result["driver_number"]
        position = result["position"] or len(results) + 1
        recent = form[year, driver]
        # Two neutral observations shrink estimates for drivers with little history.
        strength = (sum(recent) + 1) / (len(recent) + 2)
        rows.append((driver, position, (-math.log(position), 4 * (strength - 0.5))))
    if len({driver for driver, _, _ in rows}) != len(rows):
        raise ValueError("Qualifying classification contains duplicate driver numbers.")
    return rows


def probabilities(field, weights):
    scores = [sum(weight * value for weight, value in zip(weights, features))
              for _, _, features in field]
    peak = max(scores)
    values = [math.exp(score - peak) for score in scores]
    total = sum(values)
    return [value / total for value in values]


def fit(history):
    weights = [0.0, 0.0]
    latest_year = history[-1][2]
    weighted = [(field, winner, 0.7 ** (latest_year - year))
                for field, winner, year in history]
    total = sum(importance for _, _, importance in weighted)
    for _ in range(350):
        gradient = [0.02 * weight for weight in weights]
        for field, winner, importance in weighted:
            for (driver, _, features), probability in zip(field, probabilities(field, weights)):
                error = importance * (probability - (driver == winner)) / total
                for i, feature in enumerate(features):
                    gradient[i] += error * feature
        weights = [weight - 0.3 * change for weight, change in zip(weights, gradient)]
    return weights


def evaluate(history):
    if len(history) < 25:
        raise ValueError("Need at least 25 complete historical race weekends for training and evaluation.")
    split = max(20, int(len(history) * 0.8))
    train, holdout = history[:split], history[split:]
    weights = fit(train)
    pole_weights = fit([(without_form(field), winner, year)
                        for field, winner, year in train])
    hits = pole_hits = 0
    loss = baseline_loss = 0.0
    for field, winner, _ in holdout:
        predictions = probabilities(field, weights)
        baseline = probabilities(without_form(field), pole_weights)
        winner_index = next(i for i, (driver, _, _) in enumerate(field) if driver == winner)
        hits += max(range(len(field)), key=predictions.__getitem__) == winner_index
        pole_hits += field[winner_index][1] == 1
        loss -= math.log(predictions[winner_index])
        baseline_loss -= math.log(baseline[winner_index])
    count = len(holdout)
    print(f"Chronological holdout ({count} races, model fit on {split} earlier races)")
    print(f"Top pick accuracy: model {hits / count:.1%}, pole sitter {pole_hits / count:.1%}")
    print(f"Winner log loss: model {loss / count:.3f}, qualifying-only {baseline_loss / count:.3f}")


def without_form(field):
    return [(driver, position, (features[0], 0.0)) for driver, position, features in field]


def demo_data():
    from random import Random

    random = Random(42)
    form = defaultdict(lambda: deque(maxlen=5))
    history = []
    for index in range(81):
        order = sorted(range(1, 21), key=lambda driver: driver / 3 + random.gauss(0, 2))
        results = [{"driver_number": driver, "position": position}
                   for position, driver in enumerate(order, 1)]
        field = make_field(results, form, 2026)
        if index == 80:
            return history, field, {}
        winner = random.choices(order, weights=[1 / (position + 1) for position in range(20)])[0]
        history.append((field, winner, 2026))
        finish = [winner] + [driver for driver in order if driver != winner]
        for position, driver in enumerate(finish, 1):
            form[2026, driver].append(1 - position / 21)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--first-year", type=int, default=2023)
    parser.add_argument("--circuit", default="Las Vegas")
    parser.add_argument("--cache", type=Path, default=Path(".openf1-cache"))
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--demo", action="store_true", help="Run offline with synthetic races.")
    args = parser.parse_args()
    if not 2023 <= args.first_year <= args.year:
        parser.error("Choose 2023 <= first-year <= year.")

    try:
        if args.demo:
            print("SYNTHETIC DEMO")
            history, field, names = demo_data()
        else:
            history, field, names = fetch_rows(OpenF1(args.cache), args.year, args.circuit, args.first_year)
        evaluate(history)
        predictions = probabilities(field, fit(history))
        ranked = sorted(zip(field, predictions), key=lambda row: -row[1])
        rows = [
            (rank, names.get(driver, f"Driver #{driver}"), driver, position, probability)
            for rank, ((driver, position, _), probability) in enumerate(ranked, 1)
        ]
        print("\nSynthetic race" if args.demo else f"\n{args.year} {args.circuit} winner forecast")
        for rank, name, _, position, probability in rows:
            print(f"{rank:2}. {name:26} {probability:6.2%}  Q{position}")
        if args.csv:
            with args.csv.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["rank", "driver", "driver_number", "qualifying_position", "win_probability"])
                writer.writerows(rows)
    except (ValueError, OSError, URLError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
