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

def quantile(sorted_vals, q):
    n = len(sorted_vals)
    if n == 0: return float("nan")
    pos = q*(n-1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi: return sorted_vals[lo]
    return sorted_vals[lo]*(hi-pos) + sorted_vals[hi]*(pos-lo)

c = json.load(open("/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit/input/p00.in"))
comps = []
for comp in c["comparables"]:
    if "Grand Theft Auto VI" in comp["name"]: continue
    day = release_day(comp)
    if day[0] is None or any(v is None for v in day): continue
    comps.append((comp["name"], day))
print("n usable comparables:", len(comps))

# log-ratio profiles
print("\nhour | median ratio | q05 q25 q75 q95 of log-ratio (as ratio) | n")
for h in range(1, 24):
    lrs = sorted(math.log(day[h]/day[0]) for _, day in comps)
    print(f"h{h+1:2d} | med {math.exp(statistics.median(lrs)):7.3f} | "
          f"q05 {math.exp(quantile(lrs,0.05)):7.3f} q25 {math.exp(quantile(lrs,0.25)):7.3f} "
          f"q75 {math.exp(quantile(lrs,0.75)):7.3f} q95 {math.exp(quantile(lrs,0.95)):7.3f}")

# correlation of log-ratio with log(h1): check per hour
print("\ncorr(log h1, log ratio) per hour:")
import statistics as st
for h in [1,2,4,7,11,15,19,23]:
    xs = [math.log(day[0]) for _, day in comps]
    ys = [math.log(day[h]/day[0]) for _, day in comps]
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    vx = sum((x-mx)**2 for x in xs); vy = sum((y-my)**2 for y in ys)
    print(f"h{h+1:2d}: corr = {cov/math.sqrt(vx*vy):+.3f}  (mean ratio {math.exp(my):.3f})")

# hour-0 (23:00 day before) vs hour 1
print("\nhour0/hour1 ratio distribution:")
def day_before(comp):
    ds = parse(comp["day_start"])
    by = {}
    for row in comp["hourly"]:
        t = parse(row["from"])
        if ds - timedelta(hours=24) <= t < ds + timedelta(hours=24):
            by[int((t-ds).total_seconds()//3600)] = row["viewers"]
    return by
ratios = []
for comp in c["comparables"]:
    if "Grand Theft Auto VI" in comp["name"]: continue
    by = day_before(comp)
    if -1 in by and 0 in by and by[0] and by[-1]:
        ratios.append(by[0]/by[-1])
ratios.sort()
print("n:", len(ratios), "median:", round(statistics.median(ratios),3),
      "q25:", round(quantile(ratios,0.25),3), "q75:", round(quantile(ratios,0.75),3))
