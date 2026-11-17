import json, math, statistics
from datetime import datetime, timedelta

def parse(t): return datetime.fromisoformat(t.replace("Z","+00:00"))

def release_day(comp):
    ds = parse(comp["day_start"])
    by = {}
    for row in comp["hourly"]:
        t = parse(row["from"])
        if ds <= t < ds + timedelta(hours=24):
            by[int((t-ds).total_seconds()//3600)] = row["viewers"]
    return [by.get(i) for i in range(24)]

c = json.load(open("/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit/input/p00.in"))
comps = []
for comp in c["comparables"]:
    if "Grand Theft Auto VI" in comp["name"]: continue
    day = release_day(comp)
    if day[0] is None or any(v is None for v in day): continue
    comps.append((comp["name"], comp.get("platforms",""), comp.get("unlock",""), day))
comps.sort(key=lambda x: -x[3][0])

print("Top 10 by h1 — log-ratio profile (ratio to h1) at hours 2,3,6,9,12,15,18,21,24:")
print(f"{'name':38s} {'h1':>8s} | " + " ".join(f"h{h:>2d}" for h in [2,3,6,9,12,15,18,21,24]))
for name, plat, unlock, day in comps[:10]:
    r = [day[h-1]/day[0] for h in [2,3,6,9,12,15,18,21,24]]
    print(f"{name[:38]:38s} {day[0]:8d} | " + " ".join(f"{x:5.2f}" for x in r))

print("\nSmall-h1 launches:")
for name, plat, unlock, day in comps[-8:]:
    r = [day[h-1]/day[0] for h in [2,3,6,9,12,15,18,21,24]]
    print(f"{name[:38]:38s} {day[0]:8d} | " + " ".join(f"{x:5.2f}" for x in r))

print("\nUnlock strings for top-12:")
for name, plat, unlock, day in comps[:12]:
    print(f"  {name}: platforms={plat} | unlock={unlock[:150]}")

print("\nPlatform strings (unique):")
seen=set()
for name, plat, unlock, day in comps:
    if plat not in seen:
        seen.add(plat); print("  ", repr(plat), "->", name)
