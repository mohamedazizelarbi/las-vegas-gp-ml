"""Build matched clean-lap pace features from OpenF1 laps, stints and control messages."""
import gzip
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from statistics import median

from .download_data import PRACTICES, timing_file

COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
DELETED_LAP = re.compile(r"\bLAP\s+(\d+)\b")
DELETED_CAR = re.compile(r"\bCAR\s+(\d+)\b")


def time_of(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() if value else None


def data_for(endpoint, meeting, target):
    path = timing_file(endpoint, meeting, target)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        result = defaultdict(list)
        for row in json.load(stream):
            result[row["session_key"]].append(row)
        return result


def messages(rows):
    """Build conservative caution windows and exact deletions when the lap is identified."""
    blocked, deleted = [], set()
    active = {}
    for row in sorted(rows, key=lambda item: item.get("date") or ""):
        moment = time_of(row.get("date"))
        if moment is None:
            continue
        msg = (row.get("message") or "").upper()
        flag = (row.get("flag") or "").upper()
        scope = row.get("scope")
        sector = row.get("sector") if scope == "Sector" else None
        key = sector if sector is not None else "track"
        if "DELETED" in msg and "REINSTATED" not in msg:
            match = DELETED_LAP.search(msg)
            car = DELETED_CAR.search(msg)
            number = row.get("driver_number")
            if number is None and car:
                number = int(car[1])
            lap = int(match[1]) if match else row.get("lap_number")
            if number is not None and lap:
                deleted.add((number, lap))
        if flag in ("YELLOW", "DOUBLE YELLOW", "RED") and scope != "Driver":
            active.setdefault(key, moment)
        if "SAFETY CAR" in msg and ("DEPLOYED" in msg or "LIGHTS ON" in msg):
            active.setdefault("track", moment)
        green = flag in ("GREEN", "CLEAR") or bool(re.search(
            r"GREEN FLAG|TRACK CLEAR|VIRTUAL SAFETY CAR ENDED", msg
        ))
        if green:
            keys = [key] if key != "track" else list(active)
            for current in keys:
                if current in active:
                    blocked.append((active.pop(current), moment))
    # Some local alerts have no paired clear in the feed. Cover the immediate lap window.
    blocked.extend((start, start + 110) for start in active.values())
    return blocked, deleted


def acceptable_laps(laps, stints, controls, race, audit):
    if not laps or not stints or not controls:
        audit["sessions_without_required_data"] += 1
        return []
    blocked, deleted = messages(controls)
    audit["control_windows"] += len(blocked)
    audit["identified_deleted_laps"] += len(deleted)
    stint_map = {}
    for row in stints:
        first, last = row.get("lap_start"), row.get("lap_end")
        age = row.get("tyre_age_at_start")
        if not isinstance(first, int) or not isinstance(last, int) or not isinstance(age, int):
            continue
        if row.get("compound") not in COMPOUNDS or not 1 <= first <= last <= 120:
            continue
        for lap in range(first, last + 1):
            stint_map[row["driver_number"], lap] = row
    groups = defaultdict(list)
    duplicates = Counter((row["driver_number"], row["lap_number"]) for row in laps)
    for row in laps:
        audit["raw_laps"] += 1
        lap = row["lap_number"]
        driver = row["driver_number"]
        key = (driver, lap)
        stint = stint_map.get(key)
        if not stint or duplicates[key] != 1 or key in deleted:
            continue
        duration = row.get("lap_duration")
        sectors = [row.get(f"duration_sector_{i}") for i in (1, 2, 3)]
        moment = time_of(row.get("date_start"))
        if (not isinstance(duration, (int, float)) or not 45 < duration < 180
                or not all(isinstance(v, (int, float)) and v > 0 for v in sectors)
                or abs(sum(sectors) - duration) > 1 or moment is None):
            continue
        if row.get("is_pit_out_lap") or lap <= stint["lap_start"] or lap == stint["lap_end"]:
            continue
        if any(moment < end and moment + duration > start for start, end in blocked):
            continue
        age = stint["tyre_age_at_start"] + lap - stint["lap_start"]
        groups[driver, stint["stint_number"], stint["compound"]].append((lap, age, duration))
    accepted = []
    for (driver, _, compound), samples in groups.items():
        # Drop in-lap-like outliers using a within-stint threshold before checking continuity.
        center = median(item[2] for item in samples)
        clean = sorted(item for item in samples if abs(item[2] / center - 1) <= 0.04)
        runs = []
        for item in clean:
            if not runs or item[0] != runs[-1][-1][0] + 1:
                runs.append([])
            runs[-1].append(item)
        for run in runs:
            if len(run) >= (2 if race else 3):
                accepted.extend((driver, compound, *item) for item in run)
    audit["clean_laps"] += len(accepted)
    return accepted


def match_pace(accepted, race, audit):
    peers = defaultdict(lambda: defaultdict(list))
    for driver, compound, lap, age, seconds in accepted:
        key = (compound, age // 7, (lap - 1) // 15 if race else 0)
        peers[key][driver].append(seconds)
    gaps = defaultdict(list)
    for field in peers.values():
        if len(field) < 3:
            continue
        centers = {driver: median(times) for driver, times in field.items()}
        for driver, laps in field.items():
            standard = median(v for n, v in centers.items() if n != driver)
            gaps[driver].extend(max(-4, min(4, 100 * (seconds / standard - 1))) for seconds in laps)
    audit["matched_laps"] += sum(map(len, gaps.values()))
    minimum = 5 if race else 3
    return {n: (median(values), len(values)) for n, values in gaps.items() if len(values) >= minimum}


def meeting_pace(sessions, meeting, target=False):
    laps, stints, flags = [data_for(name, meeting, target) for name in ("laps", "stints", "race_control")]
    practice_pace = defaultdict(list)
    race_pace = {}
    diagnostics = {}
    for session in sessions:
        name = session["session_name"]
        if name not in PRACTICES and (target or name != "Race"):
            continue
        key = session["session_key"]
        audit = Counter()
        race = name == "Race"
        accepted = acceptable_laps(laps[key], stints[key], flags[key], race, audit)
        pace = match_pace(accepted, race, audit)
        diagnostics[name] = {**audit, "drivers_with_pace": len(pace)}
        if race:
            race_pace = pace
        else:
            for number, (gap, count) in pace.items():
                practice_pace[number].append((gap, count))
    practices = {
        number: (median(gap for gap, _ in sessions), sum(count for _, count in sessions))
        for number, sessions in practice_pace.items()
    }
    return practices, race_pace, diagnostics
