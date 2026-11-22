"""Score the locked 2026 Vegas forecasts against FIA document 65."""
import csv
import json
import math
from statistics import fmean

from ..paths import ARTIFACTS, DATA, RUNS

OUT = RUNS / "audit"
FIA = "https://www.fia.com/system/files/decision-document/2026_las_vegas_grand_prix_-_final_race_classification.pdf"


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def indexed(path):
    entries = read(path)
    result = {r["driver_number"]: r for r in entries}
    if len(result) != len(entries):
        raise ValueError(f"Duplicate driver number in {path}")
    return result


def attach():
    actual = indexed(DATA / "vegas_2026_final_classification.csv")
    sim = indexed(ARTIFACTS / "simulation" / "vegas_2026_finish_distribution.csv")
    ridge = indexed(ARTIFACTS / "post_qualifying/vegas_2026_post_qualifying.csv")
    variants = indexed(ARTIFACTS / "feature_comparison/vegas_2026_forecast_comparison.csv")
    early_finish = indexed(ARTIFACTS / "pre_weekend" / "vegas_2026_finish.csv")
    early_win = indexed(ARTIFACTS / "pre_weekend" / "vegas_2026_pre_weekend.csv")
    if not len(actual) == 22 or any(set(actual) != set(v) for v in (sim,ridge,variants,early_finish,early_win)):
        raise ValueError("Official and predicted fields differ.")
    comparison = []
    for no, info in actual.items():
        s, r = sim[no], ridge[no]
        pos = int(info["position"]) if info["position"] else None
        status = info["status"]
        outcome = "DNF" if status == "DNF" else f"P{pos}" if status == "finished" else None
        comparison.append({
            "driver_number": int(no), "driver": s["driver"].title(), "team": s["constructor"],
            "forecast_grid": int(s["grid"]), "position": pos, "status": status,
            "laps": int(info["laps"]), "points": int(info["points"]),
            "expected": float(s["expected_finish_if_no_dnf"]), "ridge": float(r["estimated_finish"]),
            "early_ridge": float(early_finish[no]["expected_position"]),
            "win": float(s["P1"]), "podium":float(s["podium"]), "top10":float(s["points_top_10"]),
            "dnf":float(s["DNF"]), "dnf_model":float(s["dnf_model"]),
            "outcome": outcome, "outcome_probability":float(s[outcome]) if outcome else None
        })
    assert sum(r["points"] for r in comparison) == 101
    return comparison, early_win, ridge, variants


def calculate(rows, early_win, ridge, variants):
    finishers = [r for r in rows if r["status"] == "finished"]
    starters = [r for r in rows if r["status"] != "DNS"]
    winner = next(r for r in rows if r["position"] == 1)
    err = {metric:fmean(abs(row[col] - row["position"]) for row in finishers)
           for metric,col in (("simulation","expected"),("post_qualifying_ridge","ridge"),
                              ("pre_weekend_ridge","early_ridge"),("grid","forecast_grid"))}
    race = {}
    for key,c,is_hit in [
        ("podium","podium",lambda r: r["position"] is not None and r["position"] <= 3),
        ("top10","top10",lambda r: r["points"] > 0),
        ("dnf","dnf_model",lambda r: r["status"]=="DNF"),
    ]:
        race[key] = {"brier":fmean((row[c]-is_hit(row))**2 for row in starters),
                     "expected_count":sum(row[c] for row in starters),
                     "actual_count":sum(is_hit(row) for row in starters)}
        if key != "dnf":
            count = 3 if key=="podium" else 10
            likely = sorted(rows,key=lambda row: -row[c])[:count]
            race[key]["top_pick_overlap"] = sum(is_hit(row) for row in likely)
    odds = {
        "Pre-weekend winner model":float(early_win[str(winner["driver_number"])]["win_probability"]),
        "Post-qualifying winner model (44 races)":float(ridge[str(winner["driver_number"])]["win_probability"]),
        "Current winner model (90 races)":float(variants[str(winner["driver_number"])]["current_raw"]),
        "Full v2 winner model (calibrated)":float(variants[str(winner["driver_number"])]["enriched_calibrated"]),
        "Finishing-order simulator":winner["win"],
    }
    return {"source":FIA,"winner":winner["driver_number"],"podium":[r["driver_number"] for r in rows[:3]],
            "finishers":len(finishers),"starters":len(starters),"mae":err,"race_events":race,
            "winner_forecast":[{"model":m,"win_probability":p,"winner_log_loss":-math.log(p)} for m,p in odds.items()]}


def main():
    comparisons, early_win, ridge, variants = attach()
    results = calculate(comparisons, early_win, ridge, variants)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "vegas_2026_post_race_comparison.csv").open("w",encoding="utf8",newline="") as file:
        writer = csv.DictWriter(file, fieldnames=comparisons[0])
        writer.writeheader()
        writer.writerows(comparisons)
    (OUT / "vegas_2026_post_race_audit.json").write_text(
        json.dumps({"summary":results,"field":comparisons},indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
