import json, math, statistics, sys
sys.path.insert(0, "/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01")
from main import forecast_case, LEVELS

def pinball(y, q, tau):
    y = max(1.0, y)
    d = math.log(y) - math.log(max(1e-9, q))
    return 2*max(tau*d, (tau-1)*d)

KIT = "/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit"
c = json.load(open(f"{KIT}/input/p00.in"))
comps = [x for x in c["comparables"] if "Grand Theft Auto VI" not in x["name"]]

def release_day(comp):
    from datetime import datetime, timedelta
    ds = datetime.fromisoformat(comp["day_start"].replace("Z","+00:00"))
    by = {}
    for row in comp["hourly"]:
        t = datetime.fromisoformat(row["from"].replace("Z","+00:00"))
        if ds <= t < ds + timedelta(hours=24):
            by[int((t-ds).total_seconds()//3600)] = row["viewers"]
    return [by.get(i) for i in range(24)]

# Build pseudo-cases: each comparable as the "game", others as comparables
losses_all, losses_big = [], []
for target in comps:
    day = release_day(target)
    if day[0] is None or any(v is None for v in day): continue
    others = [x for x in comps if x is not target]
    case = {"known": [{"hour": 1, "viewers": day[0]}],
            "forecast": [{"hour": h} for h in range(2, 25)],
            "comparables": others}
    fc = forecast_case(case, target["name"])
    assert len(fc["hours"]) == 23
    tot, n = 0.0, 0
    for row in fc["hours"]:
        y = day[row["hour"]-1]
        for tau, q in zip(LEVELS, row["q"]):
            tot += pinball(y, q, tau); n += 1
    loss = tot/n
    losses_all.append(loss)
    if day[0] >= 50000: losses_big.append(loss)
    print(f"{target['name'][:40]:40s} h1={day[0]:7d} loss={loss:.4f}")

print(f"\nSHIPPED main.py LOO: all n={len(losses_all)} mean={statistics.mean(losses_all):.4f} median={statistics.median(losses_all):.4f}")
print(f"SHIPPED main.py LOO: big n={len(losses_big)} mean={statistics.mean(losses_big):.4f} median={statistics.median(losses_big):.4f}")

# House baseline on the same sets for comparison
def baseline_loss(day):
    p1 = max(1, day[0]); tot, n = 0.0, 0
    Z = [-1.645,-0.674,0,0.674,1.645]
    for h in range(2, 25):
        for tau, z in zip(LEVELS, Z):
            q = p1*math.exp(z*0.15*math.sqrt(h-1))
            tot += pinball(day[h-1], q, tau); n += 1
    return tot/n
bl_all, bl_big = [], []
for target in comps:
    day = release_day(target)
    if day[0] is None or any(v is None for v in day): continue
    l = baseline_loss(day)
    bl_all.append(l)
    if day[0] >= 50000: bl_big.append(l)
print(f"BASELINE      LOO: all mean={statistics.mean(bl_all):.4f} | big mean={statistics.mean(bl_big):.4f}")
