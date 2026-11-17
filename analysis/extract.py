import json, math
from datetime import datetime, timezone, timedelta

def parse(t): return datetime.fromisoformat(t.replace("Z","+00:00"))

def release_day(comp):
    """Return list of 24 hourly viewer counts for release day (index 0 = hour 1), None if missing."""
    ds = parse(comp["day_start"])
    by = {}
    for row in comp["hourly"]:
        t = parse(row["from"])
        if ds <= t < ds + timedelta(hours=24):
            by[int((t-ds).total_seconds()//3600)] = row["viewers"]
    return [by.get(i) for i in range(24)], ds

for pid in ["p00","p01","p02","p03"]:
    c = json.load(open(f"/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit/input/{pid}.in"))
    print("="*100)
    print(pid, c["game"]["name"])
    for comp in c["comparables"]:
        day, ds = release_day(comp)
        have = sum(1 for v in day if v is not None)
        h1 = day[0]
        print(f"{comp['name'][:42]:42s} day {ds.date()} have {have:2d}/24 h1={h1}")
