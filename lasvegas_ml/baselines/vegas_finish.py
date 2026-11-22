"""Pre-weekend finishing-position ridge regression. Python 3.10+."""
import argparse
import csv
import html
import math
from collections import defaultdict, deque
from datetime import date
from pathlib import Path
from statistics import fmean, pstdev

from .vegas_pre_weekend import historical_fields
from .vegas_predictor import OpenF1


FEATURES = ("Recent classified finish", "Recent DNF rate", "Recent win rate")


class RaceHistory:
    def __init__(self):
        self.races = []
        self.drivers = defaultdict(lambda: deque(maxlen=5))
        self.candidates = 0

    def features(self, year, driver):
        runs = self.drivers[year, driver]
        finishes = [position for position, _, _ in runs if position is not None]
        return (
            (sum(finishes) + 1) / (len(finishes) + 2),
            (sum(dnf for _, dnf, _ in runs) + 0.25) / (len(runs) + 2),
            (sum(won for _, _, won in runs) + 0.1) / (len(runs) + 2),
        )

    def observe(self, year, roster, results):
        size = len(results)
        eligible = set(roster)
        race = []
        for row in results:
            driver = row["driver_number"]
            position = row["position"]
            started = not row.get("dns")
            classified = started and not row.get("dsq") and isinstance(position, int) and 1 <= position <= size
            if driver in eligible and classified:
                race.append((self.features(year, driver), (position - 1) / (size - 1), size))
        if roster:
            self.candidates += len(roster)
            self.races.append((race, len(roster)))
        for row in results:
            if row.get("dns"):
                continue
            position = row["position"]
            ranked = not row.get("dsq") and isinstance(position, int) and 1 <= position <= size
            fraction = (position - 1) / (size - 1) if ranked else None
            self.drivers[year, row["driver_number"]].append(
                (fraction, bool(row.get("dnf")), ranked and position == 1)
            )


def fit(races, penalty):
    data = [record for race, _ in races for record in race]
    columns = list(zip(*(features for features, _, _ in data)))
    means = [fmean(column) for column in columns]
    scales = [pstdev(column) or 1.0 for column in columns]
    intercept = fmean(y for _, y, _ in data)
    normalized = [
        ([(x - center) / scale for x, center, scale in zip(features, means, scales)], y - intercept)
        for features, y, _ in data
    ]
    size = len(columns)
    # Solve the regularized normal equations by elimination with pivoting.
    matrix = []
    for i in range(size):
        row = [
            fmean(x[i] * x[j] for x, _ in normalized) + (penalty if i == j else 0)
            for j in range(size)
        ]
        matrix.append(row + [fmean(x[i] * y for x, y in normalized)])
    for i in range(size):
        pivot = max(range(i, size), key=lambda k: abs(matrix[k][i]))
        matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
        divisor = matrix[i][i]
        matrix[i] = [value / divisor for value in matrix[i]]
        for j in range(size):
            if j != i:
                factor = matrix[j][i]
                matrix[j] = [value - factor * ref for value, ref in zip(matrix[j], matrix[i])]
    return intercept, means, scales, [row[-1] for row in matrix]


def predict(model, features, field_size):
    intercept, means, scales, weights = model
    fraction = intercept + sum(w * (x - m) / s for x, m, s, w in zip(features, means, scales, weights))
    return 1 + min(1, max(0, fraction)) * (field_size - 1)


def evaluate(races, model=None):
    errors = []
    for race, _ in races:
        for features, target, size in race:
            estimate = predict(model, features, size) if model else 1 + features[0] * (size - 1)
            errors.append(estimate - (1 + target * (size - 1)))
    return fmean(map(abs, errors)), math.sqrt(fmean(e * e for e in errors)), len(errors)


def chart(path, coefficients, size):
    max_value = max(abs(value) for value in coefficients) or 1
    lines = []
    for name, value in zip(FEATURES, coefficients):
        width = 225 * abs(value) / max_value
        left = 530 - width if value < 0 else 530
        tx = left - 9 if value < 0 else left + width + 9
        align = "end" if value < 0 else "start"
        y = 161 + 75 * len(lines)
        color = "#2563a1" if value < 0 else "#bd482f"
        lines.append(f'<text x="28" y="{y+20}">{html.escape(name)}</text>'
                     f'<rect x="{left}" y="{y}" width="{width}" height="28" fill="{color}"/>'
                     f'<text x="{tx}" y="{y+20}" text-anchor="{align}">{value:+.2f}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="880" height="450" viewBox="0 0 880 450">
<rect width="880" height="450" fill="#f8fafc"/>
<g font-family="Arial,sans-serif" font-size="17" fill="#17283a">
<text x="28" y="55" font-size="28" font-weight="bold">Las Vegas finishing-position ridge coefficients</text>
<text x="28" y="88">Change in predicted positions per one feature standard deviation, with other features held fixed</text>
<text x="530" y="126" text-anchor="middle">0</text>
<line x1="530" y1="137" x2="530" y2="359" stroke="#708196"/>
{"".join(lines)}
<text x="28" y="408">Negative = better finish. Positive = worse finish. Effects scaled to a {size}-driver field.</text>
</g></svg>'''
    path.write_text(svg, encoding="utf-8")


class CacheOnly(OpenF1):
    def get(self, endpoint, *, refresh=False, **filters):
        from urllib.parse import urlencode

        filename = self.cache / (endpoint + "_" + urlencode(filters).replace("&", "_") + ".json")
        if not filename.exists():
            raise ValueError(f"Cache entry missing: {filename}. Run again without --offline.")
        return super().get(endpoint, **filters)


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--first-year", type=int, default=2023)
    parser.add_argument("--cache", type=Path, default=Path(".openf1-cache"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    if not 2023 <= args.first_year <= args.year:
        parser.error("Choose 2023 <= first-year <= year.")
    try:
        data = RaceHistory()
        api = (CacheOnly if args.offline else OpenF1)(args.cache)
        _, _, roster, names = historical_fields(
            api, args.year, args.first_year, "Las Vegas", observe_race=data.observe
        )
        if len(data.races) < 25 or len(roster) < 10:
            raise ValueError("Need 25 or more historical races and a complete expected lineup.")
        split = int(len(data.races) * 0.8)
        train, test = data.races[:split], data.races[split:]
        inner = int(len(train) * 0.8)
        penalty = min((0.01, 0.05, 0.2, 1.0), key=lambda p: evaluate(train[inner:], fit(train[:inner], p))[0])
        mae, rmse, scored = evaluate(test, fit(train, penalty))
        base_mae, base_rmse, _ = evaluate(test)
        print(f"Holdout: {len(test)} races, {scored}/{sum(n for _, n in test)} expected driver entries scored")
        print(f"Ridge MAE {mae:.3f}, RMSE {rmse:.3f} positions")
        print(f"Recent finish baseline MAE {base_mae:.3f}, RMSE {base_rmse:.3f} positions")
        print(f"Selected penalty: {penalty}")
        model = fit(data.races, penalty)
        size = len(roster)
        ranks = sorted((predict(model, data.features(args.year, driver), size), driver) for driver in roster)
        coefficients = [weight * (size - 1) for weight in model[-1]]
        stem = args.output_dir / f"vegas_{args.year}_finish"
        write_csv(stem.with_suffix(".csv"), ("projected_rank", "driver", "driver_number", "expected_position"),
                  [(index, names.get(driver, f"Driver #{driver}"), driver, f"{value:.4f}")
                   for index, (value, driver) in enumerate(ranks, 1)])
        write_csv(stem.with_name(stem.name + "_coefficients.csv"),
                  ("feature", "positions_per_standard_deviation", "training_mean", "training_standard_deviation"),
                  zip(FEATURES, coefficients, model[1], model[2]))
        chart(stem.with_name(stem.name + "_coefficients.svg"), coefficients, size)
        for index, (value, driver) in enumerate(ranks[:5], 1):
            print(f"{index}. {names.get(driver, driver)}: expected position {value:.2f}")
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
