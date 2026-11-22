"""Forecast an F1 winner before the race weekend. Python 3.10+."""
import argparse
import csv
import math
import sys
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import URLError

from .vegas_predictor import OpenF1, demo_data, fit, probabilities, timestamp


def field_from_roster(roster, form):
    field = []
    for number in roster:
        recent = form[number]
        mean_finish = (sum(score for score, _ in recent) + 1) / (len(recent) + 2)
        win_rate = (sum(won for _, won in recent) + 0.1) / (len(recent) + 2)
        field.append((number, 0, (4 * (mean_finish - 0.5), 4 * win_rate)))
    return field


def update_form(form, results):
    for row in results:
        position = row["position"]
        if position is None or row.get("dsq") or row.get("dns"):
            position = len(results) + 1
        score = 1 - min(position, len(results) + 1) / (len(results) + 1)
        form[row["driver_number"]].append((score, position == 1))


def historical_fields(api, year, first_year, circuit, observe_race=None):
    now = datetime.now(timezone.utc)
    current = api.get("sessions", refresh=True, year=year)
    matches = [
        row for row in current
        if row["session_name"] == "Race" and not row.get("is_cancelled")
        and circuit.casefold() in f'{row["location"]} {row["circuit_short_name"]}'.casefold()
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one race matching {circuit!r} in {year}, found {len(matches)}.")
    meeting = matches[0]["meeting_key"]
    cutoff = min(timestamp(row, "date_start") for row in current
                 if row["meeting_key"] == meeting and not row.get("is_cancelled"))
    cutoff = min(cutoff, now)

    history = []
    for season in range(first_year, year + 1):
        print(f"Loading {season} race history...", file=sys.stderr, flush=True)
        sessions = current if season == year else api.get("sessions", year=season)
        races = sorted(
            (row for row in sessions if row["session_name"] == "Race"
             and not row.get("is_cancelled") and timestamp(row, "date_end") < cutoff
             and timestamp(row, "date_end") < now - timedelta(minutes=30)),
            key=lambda row: timestamp(row, "date_start"),
        )
        form = defaultdict(lambda: deque(maxlen=5))
        previous_roster = []
        last_race = None
        for race in races:
            results = api.get("session_result", session_key=race["session_key"])
            winners = [row["driver_number"] for row in results if row["position"] == 1]
            numbers = [row["driver_number"] for row in results]
            if len(results) < 10 or len(winners) != 1 or len(set(numbers)) != len(numbers):
                print(f'Skipped incomplete race {race["session_key"]}.', file=sys.stderr)
                continue
            if previous_roster:
                history.append((field_from_roster(previous_roster, form), winners[0], season))
            if observe_race:
                observe_race(season, previous_roster, results)
            # Only use this race's classification after forming its forecast.
            update_form(form, results)
            previous_roster = sorted(numbers)
            last_race = race

    names = {}
    if last_race:
        drivers = api.get("drivers", session_key=last_race["session_key"])
        names = {row["driver_number"]: row["full_name"] for row in drivers
                 if row.get("full_name")}
        print(f'Expected lineup from {last_race["location"]} (previous completed GP).')
    return history, form, previous_roster, names


def backtest(history):
    if len(history) < 25:
        raise ValueError("Need at least 25 historical race weekends with a previous-race roster.")
    split = int(len(history) * 0.8)
    train, holdout = history[:split], history[split:]
    eligible = [race for race in train if any(row[0] == race[1] for row in race[0])]
    if not eligible:
        raise ValueError("No training race winners appeared in their expected lineups.")
    weights = fit(eligible)
    hits = baseline_hits = loss = 0
    covered = 0
    for field, winner, _ in holdout:
        predictions = probabilities(field, weights)
        hits += field[max(range(len(field)), key=predictions.__getitem__)][0] == winner
        baseline_hits += max(field, key=lambda row: row[2][0])[0] == winner
        for (driver, _, _), probability in zip(field, predictions):
            if driver == winner:
                covered += 1
                loss -= math.log(probability)
    count = len(holdout)
    print(f"Chronological holdout ({count} races)")
    print(f"Top pick accuracy: model {hits / count:.1%}, recent form {baseline_hits / count:.1%}")
    print(f"Expected lineup included the winner in {covered}/{count} races.")
    if covered:
        print(f"Winner log loss when lineup included winner: {loss / covered:.3f}")
    return fit([race for race in history if any(row[0] == race[1] for row in race[0])])


def synthetic_history():
    races, last_field, _ = demo_data()
    form = defaultdict(lambda: deque(maxlen=5))
    roster = sorted(row[0] for row in last_field)
    history = []
    for field, winner, season in races:
        if form:
            history.append((field_from_roster(roster, form), winner, season))
        finish = [winner] + [driver for driver, _, _ in field if driver != winner]
        update_form(form, [{"driver_number": driver, "position": position}
                           for position, driver in enumerate(finish, 1)])
    return history, form, roster, {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--first-year", type=int, default=2023)
    parser.add_argument("--circuit", default="Las Vegas")
    parser.add_argument("--cache", type=Path, default=Path(".openf1-cache"))
    parser.add_argument("--roster", type=Path, help="Optional expected lineup CSV: driver_number,driver")
    parser.add_argument("--csv", type=Path, help="Save the ranked forecast as CSV.")
    parser.add_argument("--demo", action="store_true", help="Run offline with synthetic races.")
    args = parser.parse_args()
    if not 2023 <= args.first_year <= args.year:
        parser.error("Choose 2023 <= first-year <= year.")

    try:
        if args.demo:
            print("SYNTHETIC DEMO")
            history, form, roster, names = synthetic_history()
        else:
            history, form, roster, names = historical_fields(
                OpenF1(args.cache), args.year, args.first_year, args.circuit
            )
        if args.roster:
            with args.roster.open(encoding="utf-8-sig", newline="") as stream:
                names = {int(row["driver_number"]): row["driver"]
                         for row in csv.DictReader(stream)}
            roster = sorted(names)
            print(f"Expected lineup supplied in {args.roster}.")
        if len(roster) < 10:
            raise ValueError("No complete expected lineup. Supply --roster with the full entry list.")
        weights = backtest(history)
        field = field_from_roster(roster, form)
        ranked = sorted(zip(field, probabilities(field, weights)), key=lambda row: -row[1])
        title = "Synthetic race" if args.demo else f"{args.year} {args.circuit} pre-weekend forecast"
        print(f"\n{title} (probabilities conditional on expected lineup)")
        rows = []
        for rank, ((driver, _, _), probability) in enumerate(ranked, 1):
            name = names.get(driver, f"Driver #{driver}")
            rows.append((rank, name, driver, probability))
            print(f"{rank:2}. {name:26} {probability:6.2%}")
        if args.csv:
            with args.csv.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["rank", "driver", "driver_number", "win_probability"])
                writer.writerows(rows)
    except (ValueError, KeyError, OSError, URLError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
