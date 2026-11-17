import json, math, statistics
from datetime import datetime, timedelta
exec(open("eval_models.py").read().split('c = json.load')[0])  # reuse helpers

c = json.load(open("/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit/input/p00.in"))
data = []
for comp in c["comparables"]:
    if "Grand Theft Auto VI" in comp["name"]: continue
    day = release_day(comp)
    if day[0] is None or any(v is None for v in day): continue
    pm = pre_median(comp)
    # hour before midnight (index -1) and last-3h median pre-release
    ds = parse(comp["day_start"])
    pre24 = [r["viewers"] for r in comp["hourly"] if ds - timedelta(hours=24) <= parse(r["from"]) < ds and r["viewers"]>0]
    data.append({"name": comp["name"], "day": day, "h1": day[0], "pre": pm,
                 "logh1": math.log(day[0]), "logpre": math.log(pm) if pm else None,
                 "logratio_pre": math.log(day[0]/pm) if pm else None})
big = [d for d in data if d["h1"] >= 50000]
print("big launches n =", len(big), [d["name"] for d in big])

def score_on(subset, predict, label):
    losses = []
    for d in subset:
        qs = predict(d, [x for x in data if x is not d])
        tot = 0.0; n = 0
        for h in range(1, 24):
            y = d["day"][h]
            for tau, q in zip(LEVELS, qs[h-1]):
                tot += pinball(y, q, tau); n += 1
        losses.append((d["name"], tot/n))
    mean = statistics.mean(l for _, l in losses)
    print(f"{label}: mean={mean:.4f} median={statistics.median([l for _,l in losses]):.4f}")
    return dict(losses)

def pred_baseline(d, others):
    p1 = max(1, d["h1"])
    return [[p1*math.exp(z*0.15*math.sqrt(h)) for z in Z] for h in range(1,24)]

def pred_empirical(d, others):
    out = []
    for h in range(1,24):
        lrs = sorted(math.log(x["day"][h]/x["day"][0]) for x in others)
        out.append([d["h1"]*math.exp(quantile(lrs, tau)) for tau in LEVELS])
    return out

def make_knn(bw, feat="logh1"):
    def pred(d, others):
        out = []
        for h in range(1,24):
            pairs = []
            for x in others:
                w = math.exp(-0.5*((x[feat]-d[feat])/bw)**2)
                pairs.append((math.log(x["day"][h]/x["day"][0]), w))
            out.append([d["h1"]*math.exp(wquantile(pairs, tau)) for tau in LEVELS])
        return out
    return pred

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

# 2-feature regression: log-ratio ~ logh1 + logratio_pre
def pred_regression2(d, others):
    out = []
    # simple 2-var OLS via normal equations
    import itertools
    def ols(X, y):
        n = len(X); k = len(X[0])
        A = [[sum(X[i][a]*X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
        rhs = [sum(X[i][a]*y[i] for i in range(n)) for a in range(k)]
        # gaussian elimination
        M = [row[:] + [rhs[i]] for i, row in enumerate(A)]
        for col in range(k):
            piv = max(range(col,k), key=lambda r: abs(M[r][col]))
            M[col], M[piv] = M[piv], M[col]
            if abs(M[col][col]) < 1e-12: return None
            for r in range(k):
                if r != col:
                    f = M[r][col]/M[col][col]
                    for cc in range(col, k+1): M[r][cc] -= f*M[col][cc]
        return [M[i][k]/M[i][i] for i in range(k)]
    for h in range(1,24):
        ys = [math.log(x["day"][hh]/x["day"][0]) for x in others for hh in [h]]
        X = [[1.0, x["logh1"], x["logratio_pre"]] for x in others]
        beta = ols(X, ys)
        if beta is None:
            out.append([d["h1"]*math.exp(quantile(sorted(ys), tau)) for tau in LEVELS]); continue
        resid = [y - sum(b*xx for b, xx in zip(beta, row)) for y, row in zip(ys, X)]
        sd = math.sqrt(sum(r*r for r in resid)/max(1,len(resid)-3))
        mu = beta[0] + beta[1]*d["logh1"] + beta[2]*d["logratio_pre"]
        out.append([d["h1"]*math.exp(mu + z*sd) for z in Z])
    return out

res = {}
res["baseline"] = score_on(big, pred_baseline, "M0 baseline (big only)")
res["empirical"] = score_on(big, pred_empirical, "M1 empirical (big only)")
res["knn1.0"] = score_on(big, make_knn(1.0), "M4 kNN bw=1.0 (big only)")
res["knn1.5"] = score_on(big, make_knn(1.5), "M4 kNN bw=1.5 (big only)")
res["regr"] = score_on(big, pred_regression, "M3 OLS logh1 (big only)")
res["regr2"] = score_on(big, pred_regression2, "M8 OLS logh1+logpre (big only)")

print("\nPer-case losses (big launches):")
hdr = f"{'name':38s} " + " ".join(f"{k:>8s}" for k in res)
print(hdr)
for d in big:
    print(f"{d['name'][:38]:38s} " + " ".join(f"{res[k][d['name']]:8.3f}" for k in res))

# also check: what does kNN predict for a hypothetical GTA VI h1 = 1,000,000?
print("\nHypothetical GTA VI h1=1,000,000 — predicted median ratio by hour (kNN bw=1.0 vs OLS):")
fake = {"name":"GTA6","day":[1000000]*24,"h1":1000000,"logh1":math.log(1000000),"logratio_pre":math.log(1000000/30000)}
for label, fn in [("kNN1.0", make_knn(1.0)), ("OLS", pred_regression)]:
    qs = fn(fake, data)
    print(label, " ".join(f"h{h+1}:{qs[h][2]/1000000:.2f}" for h in range(23)))
