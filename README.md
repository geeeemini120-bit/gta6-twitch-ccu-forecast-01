[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/geeeemini120-bit/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/geeeemini120-bit/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch viewers, hour by hour

Forecast how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day, 2026-11-19 (UTC), from everything recorded up to
the end of the day's first hour. The entry is a single Python 3.13 program,
`main.py` (standard library only, no network, deterministic), which reads a
case file and writes five quantiles (5%, 25%, 50%, 75%, 95%) of the
concurrent-viewer count for each of hours 2–24.

## Approach

An empirical day-shape model learned at run time from the launch
comparables shipped inside the case file itself:

1. For every past big launch in the case's `comparables`, take its
   release-day hourly Twitch category counts and express each hour as a
   log-ratio to that launch's own hour-1 print.
2. Weight each comparable by how close its log hour-1 print is to this
   case's log hour-1 print (Gaussian kernel, bandwidth 1.0 in log space).
   Launch days differ by scale: small hour-1 launches explode later in the
   day as their big markets unlock, while huge hour-1 launches dip through
   the morning and recover by the evening.
3. Take the weighted quantiles of each hour's log-ratio at the five scored
   levels, lightly smooth each quantile series across neighbouring hours,
   and exponentiate back against this case's hour-1 print.

GTA VI unlocks at midnight local time rolling around the world (NZ 11:00
UTC 18 Nov, EU 23:00 UTC, UK 00:00 UTC, US East 05:00 UTC, US West 08:00
UTC), so hour 1 already holds New Zealand, Europe and the UK, the US joins
in hours 6–9, and the peak lands in the UTC afternoon/evening. The nearest
analogues by scale and unlock structure (Elden Ring, Cyberpunk 2077,
Diablo IV, Hogwarts Legacy, The Last of Us Part II) all show the same
pattern: a morning dip to ~0.5× hour 1, recovery to ~0.9–1.3× by hours
18–21.

## Validation

Leave-one-out over the 37 released comparables in the kit, each scored as
its own pseudo-case under the contest's pinball loss on the log scale:

| set | this entry | house baseline |
| --- | --- | --- |
| all 37 launches | **0.403** | 0.930 |
| 20 launches with hour-1 ≥ 50k (GTA VI's scale) | **0.207** | 0.251 |

90% coverage on the big launches is 0.92 (target 0.90). The kit's starter
(house baseline) scores ~0.387 on the four practice cases; this model
cannot be scored on the practice cases directly because their answers are
not shipped in the kit.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Runs in ~0.05 s per case, far inside the 8 s limit. Two runs on the same
case produce byte-identical output.

## Files

- `main.py` — the whole entry (single file, standard library only)
- `LICENSE` — MIT
- `.makefun/owner.json` — entry wallet (the make.fun account is not
  GitHub-linked, so this claim file names the payout wallet)
