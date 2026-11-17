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
    data.append({"name": comp["name"], "day": day, "h1": day[0],
                 "logh1": math.log(day[0])})
big = [d for d in data if d["h1"] >= 50000]

def score(subset, predict, label, detail=False):
    losses, covs, misses = [], [], []
    for d in subset:
        qs = predict(d, [x for x in data if x is not d])
        tot = 0.0; n = 0
        for h in range(1, 24):
            y = d["day"][h]
            for tau, q in zip(LEVELS, qs[h-1]):
                tot += pinball(y, q, tau); n += 1
        losses.append(tot/n)
        covs.append(sum(1 for h in range(1,24) if qs[h-1][0] <= d["day"][h] <= qs[h-1][4])/23)
        misses.extend(abs(math.log(d["day"][h]) - math.log(qs[h-1][2])) for h in range(1,24))
    print(f"{label}: mean={statistics.mean(losses):.4f} median={statistics.median(losses):.4f} "
          f"cov90={statistics.mean(covs):.3f} typ_miss={100*(math.exp(statistics.mean(misses))-1):.1f}%")
    return statistics.mean(losses)

def knn_raw(d, others, bw=1.0):
    """returns per-hour list of 5 weighted-quantile log-ratios"""
    out = []
    for h in range(1,24):
        pairs = [(math.log(x["day"][h]/x["day"][0]),
                  math.exp(-0.5*((x["logh1"]-d["logh1"])/bw)**2)) for x in others]
        out.append([wquantile(pairs, tau) for tau in LEVELS])
    return out

def make_knn(bw=1.0, smooth=0):
    def pred(d, others):
        lr = knn_raw(d, others, bw)
        if smooth:
            # smooth each quantile series across hours with a small kernel
            sm = []
            for i in range(23):
                row = []
                for j in range(5):
                    num = 0.0; den = 0.0
                    for k in range(max(0,i-smooth), min(23,i+smooth+1)):
                        w = 1.0 - abs(k-i)/(smooth+1)
                        num += w*lr[k][j]; den += w
                    row.append(num/den)
                sm.append(row)
            lr = sm
        return [[d["h1"]*math.exp(v) for v in row] for row in lr]
    return pred

print("--- smoothing sweep (big only) ---")
for s in [0, 1, 2]:
    score(big, make_knn(1.0, smooth=s), f"kNN bw=1.0 smooth={s}")

print("--- smoothing sweep (all 37) ---")
for s in [0, 1]:
    score(data, make_knn(1.0, smooth=s), f"kNN bw=1.0 smooth={s} (all)")

print("--- adaptive bw: widen until eff-n >= 8 (big only) ---")
def make_knn_adaptive(bw0=1.0, min_eff=8.0):
    def pred(d, others):
        bw = bw0
        while True:
            ws = [math.exp(-0.5*((x["logh1"]-d["logh1"])/bw)**2) for x in others]
            eff = sum(ws)**2 / sum(w*w for w in ws)
            if eff >= min_eff or bw > 4.0: break
            bw *= 1.25
        return make_knn(bw)(d, others)
    return pred
score(big, make_knn_adaptive(), "kNN adaptive (big only)")
score(data, make_knn_adaptive(), "kNN adaptive (all)")

# Print top-8 big-launch median-ratio profiles for shape sanity
print("\nTop-8 by h1 median-ratio profiles (h2,h3,h6,h9,h12,h15,h18,h21,h24):")
top = sorted(data, key=lambda x: -x["h1"])[:8]
for d in top:
    r = [d["day"][h-1]/d["day"][0] for h in [2,3,6,9,12,15,18,21,24]]
    print(f"{d['name'][:36]:36s} h1={d['h1']:7d} | " + " ".join(f"{x:5.2f}" for x in r))
# median profile of top-8
print("top-8 median: " + " ".join(f"{statistics.median([d['day'][h-1]/d['day'][0] for d in top]):5.2f}" for h in [2,3,6,9,12,15,18,21,24]))

# What does the chosen model predict for h1 = 500k, 1M, 2M?
print("\nChosen model (kNN bw=1.0, no smooth) median ratios for hypothetical h1:")
for h1 in [300000, 500000, 1000000, 2000000]:
    fake = {"name":"GTA6","h1":h1,"logh1":math.log(h1)}
    qs = make_knn(1.0)(fake, data)
    print(f"h1={h1:7d}: med " + " ".join(f"{qs[h][2]/h1:.2f}" for h in range(23)))
    print(f"          q05 " + " ".join(f"{qs[h][0]/h1:.2f}" for h in range(23)))
    print(f"          q95 " + " ".join(f"{qs[h][4]/h1:.2f}" for h in range(23)))
