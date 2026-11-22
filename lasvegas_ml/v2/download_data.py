"""Download and cache OpenF1 data for the rolling winner backtest."""
import gzip
import json
import time
from collections import defaultdict
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from ..baselines.train_post_qualifying import OpenF1, timestamp
from ..paths import CACHE as BASE, TIMING_CACHE as TELEMETRY
LIMITS = {"laps": 5, "stints": 12, "race_control": 12,
          "session_result": 12, "starting_grid": 12, "drivers": 12}
PRACTICES = {"Practice 1", "Practice 2", "Practice 3"}


def weeks(api):
    for year in range(2023, 2027):
        meetings = defaultdict(list)
        for row in api.get("sessions", year=year):
            if not row.get("is_cancelled"):
                meetings[row["meeting_key"]].append(row)
        ordered = sorted(
            ((timestamp(s, "date_start"), s) for block in meetings.values()
             for s in block if s["session_name"] == "Race"),
            key=lambda item: item[0],
        )
        for _, race in ordered:
            target = year == 2026 and race["location"] == "Las Vegas"
            yield year, race, meetings[race["meeting_key"]], target
            if target:
                return


def get_rows(api, endpoint, query):
    url = f"https://api.openf1.org/v1/{endpoint}?{query}"
    for attempt in range(4):
        time.sleep(max(0, api.next_request - time.monotonic()))
        api.next_request = time.monotonic() + 2.25
        try:
            with urlopen(url, timeout=65) as response:
                rows = json.load(response)
            if not isinstance(rows, list):
                raise ValueError(f"Unexpected JSON at {url}.")
            return rows
        except HTTPError as error:
            if error.code == 404:
                return []
            if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            delay = error.headers.get("Retry-After", "")
            time.sleep(int(delay) if delay.isdigit() else 2 ** (attempt + 2))
        except (TimeoutError, URLError):
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 2))


def timing_file(endpoint, key, target=False):
    tag = "vegas_practice" if target else str(key)
    return TELEMETRY / f"{endpoint}_{tag}.json.gz"


def dump(rows, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix(destination.suffix + ".partial")
    if destination.suffix == ".gz":
        with gzip.open(pending, "wt", encoding="utf-8", compresslevel=3) as stream:
            json.dump(rows, stream, separators=(",", ":"))
    else:
        pending.write_text(json.dumps(rows), encoding="utf-8")
    pending.replace(destination)


def fetch_history(api, endpoint, groups):
    timing = endpoint in ("laps", "stints", "race_control")
    limit = LIMITS[endpoint]
    for year in sorted(groups):
        paths = {}
        selections = {}
        for race, sessions in groups[year]:
            key = race["meeting_key"]
            path = timing_file(endpoint, key) if timing else BASE / f"{endpoint}_meeting_key={key}.json"
            if not path.exists():
                paths[key] = path
                selections[key] = {s["session_key"] for s in sessions
                                   if s["session_name"] in PRACTICES | {"Race"}}
        keys = sorted(paths)
        for start in range(0, len(keys), limit):
            block = keys[start:start+limit]
            print(f"{endpoint} {year} meetings {start+1}–{start+len(block)} of {len(keys)}", flush=True)
            query = f"meeting_key%3E={block[0]}&meeting_key%3C={block[-1]}"
            matched = defaultdict(list)
            for row in get_rows(api, endpoint, query):
                key = row["meeting_key"]
                if key in block and (not timing or row["session_key"] in selections[key]):
                    matched[key].append(row)
            for key in block:
                dump(matched[key], paths[key])


def main():
    api = OpenF1(BASE)
    history = defaultdict(list)
    target = None
    for year, race, sessions, is_target in weeks(api):
        if is_target:
            target = sessions
        else:
            history[year].append((race, sessions))
    if not target:
        raise ValueError("2026 Las Vegas schedule is missing.")
    for endpoint in LIMITS:
        fetch_history(api, endpoint, history)
        if endpoint in ("laps", "stints", "race_control"):
            path = timing_file(endpoint, None, target=True)
            if not path.exists():
                rows = []
                for item in target:
                    if item["session_name"] in PRACTICES:
                        rows.extend(get_rows(api, endpoint, f"session_key={item['session_key']}"))
                dump(rows, path)
    for session in target:
        name = session["session_name"]
        if name == "Qualifying":
            endpoints = ("session_result", "drivers", "starting_grid")
        elif name in PRACTICES:
            endpoints = ("session_result",)
        else:
            continue
        query = f"session_key={session['session_key']}"
        for endpoint in endpoints:
            path = BASE / f"{endpoint}_{query}.json"
            if not path.exists():
                dump(get_rows(api, endpoint, query), path)
    print("Download complete. Target inputs are limited to pre-race sessions.", flush=True)


if __name__ == "__main__":
    main()
