import json, math, statistics
from datetime import datetime, timedelta
exec(open("eval_models.py").read().split('c = json.load')[0])

c = json.load(open("/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit/input/p00.in"))
data = []
for comp in c["comparables"]:
    if "Grand Theft Auto VI" in comp["name"]: continue
    day = release_day(comp)
    if day[0] is None or any(v is None for v in day): continue
    pm = pre_median(comp)
    ds = parse(comp["day_start"])
    # channels profile
    ch = {}
    for row in comp["hourly"]:
        t = parse(row["from"])
        if ds <= t < ds + timedelta(hours=24):
            ch[int((t-ds).total_seconds()//3600)] = row["channels"]
    data.append({"name": comp["name"], "day": day, "h1": day[0], "pre": pm,
                 "logh1": math.log(day[0]), "logpre": math.log(pm) if pm else None,
                 "logratio_pre": math.log(day[0]/pm) if pm else None,
                 "channels": [ch.get(i) for i in range(24)],
                 "unlock": comp.get("unlock","")})
big = [d for d in data if d["h1"] >= 50000]
print("big n =", len(big))

def score_on(subset, predict, label, others_pool=None):
    losses = []
    for d in subset:
        pool = others_pool if others_pool is not None else [x for x in data if x is not d]
        qs = predict(d, pool)
        tot = 0.0; n = 0
        for h in range(1, 24):
            y = d["day"][h]
            for tau, q in zip(LEVELS, qs[h-1]):
                tot += pinball(y, q, tau); n += 1
        losses.append(tot/n)
    print(f"{label}: mean={statistics.mean(losses):.4f} median={statistics.median(losses):.4f}")
    return statistics.mean(losses)

def make_knn(bw, spread=1.0, feat="logh1", feat2=None, bw2=None):
    def pred(d, others):
        out = []
        for h in range(1,24):
            pairs = []
            for x in others:
                z = (x[feat]-d[feat])/bw
                w = math.exp(-0.5*z*z)
                if feat2:
                    z2 = (x[feat2]-d[feat2])/bw2
                    w *= math.exp(-0.5*z2*z2)
                pairs.append((math.log(x["day"][h]/x["day"][0]), w))
            med = wquantile(pairs, 0.5)
            qs = []
            for tau in LEVELS:
                qv = wquantile(pairs, tau)
                qs.append(d["h1"]*math.exp(med + spread*(qv-med)))
            out.append(qs)
        return out
    return pred

print("--- bandwidth sweep (big only) ---")
for bw in [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]:
    score_on(big, make_knn(bw), f"kNN bw={bw}")

print("--- spread sweep bw=1.0 (big only) ---")
for s in [0.8, 0.9, 1.0, 1.1, 1.25]:
    score_on(big, make_knn(1.0, spread=s), f"kNN bw=1.0 spread={s}")

print("--- 2D kNN: logh1 + logratio_pre (big only) ---")
for bw2 in [0.5, 1.0, 1.5]:
    score_on(big, make_knn(1.0, feat2="logratio_pre", bw2=bw2), f"kNN 2D bw2={bw2}")

print("--- tier filter: drop early-access-contaminated / weak (big only) ---")
def contaminated(d):
    u = d["unlock"].lower()
    return ("early-access-contaminated" in u or "tier: weak" in u)
clean_pool = [x for x in data if not contaminated(x)]
print("clean pool n =", len(clean_pool), "| dropped:", [x["name"] for x in data if contaminated(x)])
def score_clean(subset, predict, label):
    losses = []
    for d in subset:
        pool = [x for x in clean_pool if x is not d]
        qs = predict(d, pool)
        tot = 0.0; n = 0
        for h in range(1, 24):
            y = d["day"][h]
            for tau, q in zip(LEVELS, qs[h-1]):
                tot += pinball(y, q, tau); n += 1
        losses.append(tot/n)
    print(f"{label}: mean={statistics.mean(losses):.4f} median={statistics.median(losses):.4f}")
score_clean(big, make_knn(1.0), "kNN bw=1.0 clean pool")
score_clean(big, make_knn(0.8), "kNN bw=0.8 clean pool")

print("--- all-37 check for the best candidates ---")
def score_all(predict, label):
    losses = []
    for d in data:
        qs = predict(d, [x for x in data if x is not d])
        tot = 0.0; n = 0
        for h in range(1, 24):
            y = d["day"][h]
            for tau, q in zip(LEVELS, qs[h-1]):
                tot += pinball(y, q, tau); n += 1
        losses.append(tot/n)
    print(f"{label}: mean={statistics.mean(losses):.4f} median={statistics.median(losses):.4f}")
score_all(make_knn(1.0), "kNN bw=1.0 (all)")
score_all(make_knn(0.8), "kNN bw=0.8 (all)")
score_all(make_knn(1.0, spread=1.1), "kNN bw=1.0 spread=1.1 (all)")
