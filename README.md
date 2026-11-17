# Comparables log-quantile GTA VI Twitch forecaster

[![make.fun](https://make.fun/badge.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Entry for [gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01):
forecast how many people watch Grand Theft Auto VI live on Twitch in each hour
of its release day, 2026-11-19 (UTC), as five quantiles (5%, 25%, 50%, 75%,
95%) per hour, from everything recorded up to the end of the day's first hour.

## How it works

Per forecast hour, an OLS line is fitted across past big game launches
("comparables") in log space:

```
log viewers(hour h) = a_h + b_h * log viewers(hour 1)
```

The median is the line evaluated at the case's own hour-1 print; the spread
is the fit's residual standard deviation (floored at 0.15 log units), giving
normal quantiles `q = exp(median + z_tau * sigma)`.

Three things the practice cases taught:

- **Early hours scale ~1:1 with hour 1** (fitted slope ≈ 1.05 at hour 2,
  ≈ 0.99 at hour 5), but **later hours regress toward the typical
  launch-day level** (slope ≈ 0.53 at hour 20). A flat ratio-to-hour-1
  forecast misses this.
- **Tiny hour-1 categories are dropped** (hour-1 < 5,000 viewers): those
  launches were nearly dead at midnight UTC and their ratios explode
  100–2000x — not what a rolling-midnight console launch looks like.
- **The pre-release level is not used as an anchor**: on the practice it
  scored catastrophically (mean loss 1.26 vs 0.39 baseline), because
  pre-release hype says nothing about launch-day scale.

`streams` and `categories_hourly` are deliberately unused: they are null in
the practice cases, so no use of them could be validated locally.

## Practice score (local, pinball loss on the log scale, lower is better)

Scored against the SullyGnome hourly averages for the four practice launches
(`dev/eval.py`, answers fetched once from the public API named in the kit's
`input/NOTICE.txt`):

| case | launch | this entry | house baseline |
|------|--------|-----------:|---------------:|
| p00 | Marvel's Wolverine | 0.3054 | 0.7437 |
| p01 | Red Dead Redemption 2 | 0.2113 | 0.2868 |
| p02 | God of War Ragnarök | 0.1534 | 0.2146 |
| p03 | Marvel's Spider-Man 2 | 0.1917 | 0.3033 |
| **mean** | | **0.2155** | **0.3871** |

−44% vs the house baseline, better on every practice case. Typical miss
≈ 44% (baseline ≈ 90%), 90% coverage ≈ 0.90 (baseline ≈ 0.78). The
flat-ratio variant of the same idea scored 0.285; the pre-release-anchored
variant scored 1.26 — both rejected in `dev/proto.py` / `dev/proto2.py`.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Deterministic: two runs on
the same case write byte-identical output. ~0.05 s per case, far under the
8 s cap.
