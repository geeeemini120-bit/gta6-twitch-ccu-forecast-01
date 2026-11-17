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

def pre_median(comp, hours=24):
    ds = parse(comp["day_start"])
    vals = []
    for row in comp["hourly"]:
        t = parse(row["from"])
        if ds - timedelta(hours=hours) <= t < ds:
            if row["viewers"] > 0: vals.append(row["viewers"])
    return statistics.median(vals) if vals else None

def quantile(sorted_vals, q):
    n = len(sorted_vals)
    if n == 0: return float("nan")
    pos = q*(n-1); lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi: return sorted_vals[lo]
    return sorted_vals[lo]*(hi-pos) + sorted_vals[hi]*(pos-lo)

def wquantile(pairs, q):
    # pairs: list of (value, weight); weighted quantile
    pairs = sorted(pairs)
    tot = sum(w for _, w in pairs)
    if tot <= 0: return float("nan")
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= q*tot: return v
    return pairs[-1][0]

LEVELS = [0.05,0.25,0.5,0.75,0.95]
Z = [-1.645,-0.674,0,0.674,1.645]

def pinball(y, q, tau):
    y = max(1.0, y); q = max(1e-9, q)
    d = math.log(y) - math.log(q)
    return 2*max(tau*d, (tau-1)*d)

c = json.load(open("/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit/input/p00.in"))
data = []
for comp in c["comparables"]:
    if "Grand Theft Auto VI" in comp["name"]: continue
    day = release_day(comp)
    if day[0] is None or any(v is None for v in day): continue
    pm = pre_median(comp)
    data.append({"name": comp["name"], "day": day, "h1": day[0], "pre": pm,
                 "logh1": math.log(day[0]), "logpre": math.log(pm) if pm else None})
print("n =", len(data))

def score_model(predict, label):
    losses, misses, cov = [], [], []
    per_case = {}
    for d in data:
        qs = predict(d, [x for x in data if x is not d])  # LOO: predict fn gets others
        tot = 0.0; n = 0
        for h in range(1, 24):
            y = d["day"][h]
            for tau, q in zip(LEVELS, qs[h-1]):
                tot += pinball(y, q, tau); n += 1
        losses.append(tot/n)
        misses.append(abs(math.log(d["day"][h]) - math.log(qs[h-1][2])) for h in range(1,24))
        cov.append(sum(1 for h in range(1,24) if qs[h-1][0] <= d["day"][h] <= qs[h-1][4])/23)
        per_case[d["name"]] = tot/n
    flat_miss = [m for ms in misses for m in ms]
    print(f"{label}: loss={statistics.mean(losses):.4f} median={statistics.median(losses):.4f} "
          f"typ_miss={100*(math.exp(statistics.mean(flat_miss))-1):.1f}% cov90={statistics.mean(cov):.3f}")
    return per_case

# M0: house baseline
def pred_baseline(d, others):
    p1 = max(1, d["h1"])
    return [[p1*math.exp(z*0.15*math.sqrt(h)) for z in Z] for h in range(1,24)]
score_model(pred_baseline, "M0 house baseline")

# M1: empirical quantiles, LOO
def pred_empirical(d, others):
    out = []
    for h in range(1,24):
        lrs = sorted(math.log(x["day"][h]/x["day"][0]) for x in others)
        out.append([d["h1"]*math.exp(quantile(lrs, tau)) for tau in LEVELS])
    return out
score_model(pred_empirical, "M1 empirical quantiles (LOO)")

# M3: per-hour OLS of log-ratio on log h1 (LOO), normal quantiles from residual sd
def pred_regression(d, others):
    xs = [x["logh1"] for x in others]; mx = statistics.mean(xs)
    out = []
    for h in range(1,24):
        ys = [math.log(x["day"][h]/x["day"][0]) for x in others]
        my = statistics.mean(ys)
        sxx = sum((x-mx)**2 for x in xs)
        b = sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sxx if sxx>0 else 0.0
        a = my - b*mx
        resid = [y-(a+b*x) for x,y in zip(xs,ys)]
        sd = math.sqrt(sum(r*r for r in resid)/max(1,len(resid)-2))
        mu = a + b*d["logh1"]
        out.append([d["h1"]*math.exp(mu + z*sd) for z in Z])
    return out
score_model(pred_regression, "M3 OLS on log h1 (LOO)")

# M4: kNN kernel-weighted quantiles on log h1
def make_knn(bw):
    def pred(d, others):
        out = []
        for h in range(1,24):
            pairs = []
            for x in others:
                w = math.exp(-0.5*((x["logh1"]-d["logh1"])/bw)**2)
                pairs.append((math.log(x["day"][h]/x["day"][0]), w))
            out.append([d["h1"]*math.exp(wquantile(pairs, tau)) for tau in LEVELS])
        return out
    return pred
for bw in [0.5, 1.0, 1.5]:
    score_model(make_knn(bw), f"M4 kNN bw={bw} (LOO)")

# M5: kNN with extra spread multiplier
def make_knn_spread(bw, mult):
    def pred(d, others):
        out = []
        for h in range(1,24):
            pairs = []
            for x in others:
                w = math.exp(-0.5*((x["logh1"]-d["logh1"])/bw)**2)
                pairs.append((math.log(x["day"][h]/x["day"][0]), w))
            med = wquantile(pairs, 0.5)
            qs = []
            for tau in LEVELS:
                qv = wquantile(pairs, tau)
                qs.append(d["h1"]*math.exp(med + mult*(qv-med)))
            out.append(qs)
        return out
    return pred
for bw, mult in [(1.0,1.2),(1.0,1.5),(0.7,1.3)]:
    score_model(make_knn_spread(bw, mult), f"M5 kNN bw={bw} spread x{mult} (LOO)")

# M6: regression + kNN median blend? and regression with clamped slope
def pred_regression_clamped(d, others):
    xs = [x["logh1"] for x in others]; mx = statistics.mean(xs)
    out = []
    for h in range(1,24):
        ys = [math.log(x["day"][h]/x["day"][0]) for x in others]
        my = statistics.mean(ys)
        sxx = sum((x-mx)**2 for x in xs)
        b = sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sxx if sxx>0 else 0.0
        b = max(-1.2, min(0.2, b))
        a = my - b*mx
        resid = [y-(a+b*x) for x,y in zip(xs,ys)]
        sd = math.sqrt(sum(r*r for r in resid)/max(1,len(resid)-2))
        # clamp prediction to range seen among top-half h1 comps
        mu = a + b*d["logh1"]
        out.append([d["h1"]*math.exp(mu + z*sd) for z in Z])
    return out
score_model(pred_regression_clamped, "M6 OLS clamped slope (LOO)")

# Also: only big launches (h1>=50k) for empirical quantiles
def pred_empirical_big(d, others):
    big = [x for x in others if x["h1"] >= 50000] or others
    out = []
    for h in range(1,24):
        lrs = sorted(math.log(x["day"][h]/x["day"][0]) for x in big)
        out.append([d["h1"]*math.exp(quantile(lrs, tau)) for tau in LEVELS])
    return out
score_model(pred_empirical_big, "M7 empirical quantiles, h1>=50k only (LOO)")
