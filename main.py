"""Comparables log-quantile GTA VI Twitch forecaster (Python 3.13, stdlib only).

Forecasts the Twitch concurrent viewers of Grand Theft Auto VI for hours 2..24
of its release day (2026-11-19 UTC), as five quantiles an hour, from everything
recorded up to the end of hour 1.

Model: per-hour log-log regression on past big launches ("comparables").
For each forecast hour h, take every comparable launch that has an hourly
count for both hour 1 and hour h of its own release day, and fit an OLS line

    log viewers(h) = a_h + b_h * log viewers(1)

across those launches. The forecast median for the case is the line evaluated
at the case's own hour-1 print; the spread is the regression's residual
standard deviation (floored at 0.15 log units), giving normal quantiles

    q_tau(h) = exp(median_h + z_tau * sigma_h).

Why this shape, and what the practice cases showed (see README.md):
  - Early hours scale ~1:1 with hour 1 (fitted b_2 ~ 1.05, b_5 ~ 0.99):
    the first hours are still the same small, pre-dawn audience.
  - Later hours regress toward the typical launch-day level (b_20 ~ 0.53):
    a launch that was already huge at hour 1 does not keep growing in
    proportion, and a launch that was quiet at hour 1 still climbs toward
    the usual peak. A flat ratio-to-hour-1 forecast misses this and scores
    much worse on the practice (0.285 vs 0.215 mean pinball loss).
  - Comparables whose hour-1 count is below 5,000 viewers are dropped:
    those are categories that were nearly dead at midnight UTC (single
    global unlock later in the day, or slow-burn launches), whose ratios
    explode 100-2000x and are not what a midnight-rolling console launch
    like GTA VI looks like. The GTA VI category's own pre-release history
    (a comparable with no release-day hours) is dropped by the same rule:
    it has no hour 1 to regress from.
  - The pre-release level of the category is NOT used as an anchor: on the
    practice it scored catastrophically (mean loss 1.26 vs 0.39 baseline),
    because pre-release hype levels say nothing about launch-day scale.
  - No use is made of `streams` or `categories_hourly`: they are null in
    the practice cases, so no use of them could be validated locally.

The program is fully deterministic: no clock, no randomness, no network, no
data beyond the case file it is given. Two runs on the same case write the
same bytes. Runtime is far under the 8 s cap (tens of ms per case).

I/O contract (same as the kit's starter): read every <id>.in in the input
directory (MAKEFUN_INPUT_DIR, else /input), write <id>.out to the output
directory (MAKEFUN_OUTPUT_DIR, else /output) as
{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}, exactly 23
rows, hours 2..24 each once, each q five finite numbers above 0 and at most
1e9 (viewers; fractions allowed), non-decreasing; at most 65,536 bytes.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# A comparable only counts when its release-day hour-1 audience was at least
# this large: below it the category was effectively dead at midnight UTC and
# its hour-1-to-peak ratios (up to ~2000x) do not describe a rolling-midnight
# console launch. Tuned on the practice (5,000 and 10,000 score the same to
# 0.001; 5,000 keeps more launches in the fit).
MIN_HOUR1_VIEWERS = 5000
# Floor on the per-hour log spread, so a lucky tight fit never claims
# near-certainty about a launch day.
SIGMA_FLOOR = 0.15
# A regression needs at least this many launches at an hour to be trusted;
# with fewer, fall back to a flat forecast at the hour-1 print.
MIN_FIT_POINTS = 5
# Fallback spread (log units) when a fit is unavailable.
FALLBACK_SIGMA = 0.5
# Scorer bounds: every quantile must be above 0 and at most 1e9 viewers.
MAX_VIEWERS = 1_000_000_000


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def release_day_hours(comp: dict) -> dict[int, float]:
    """A comparable's release-day hourly viewers, keyed by hour 1..24.

    Only points inside [day_start, day_start + 24h) count; the comparable's
    14 pre-release days and 3 post-release days are ignored. A point at
    `from` covers the hour starting there.
    """
    day = parse_ts(comp["day_start"])
    by_hour: dict[int, float] = {}
    for row in comp.get("hourly") or []:
        v = row.get("viewers")
        if v is None or v <= 0:
            continue
        dh = (parse_ts(row["from"]) - day).total_seconds() / 3600
        if 0 <= dh < 24:
            by_hour[int(dh) + 1] = float(v)
    return by_hour


def ols(xs: list[float], ys: list[float]) -> tuple[float, float, list[float]]:
    """Least-squares fit y = a + b*x; returns (a, b, residuals)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx > 0 else 0.0
    a = my - b * mx
    return a, b, [y - (a + b * x) for x, y in zip(xs, ys)]


def hour_quantiles(log_p1: float, curves: list[dict[int, float]], h: int) -> list[float | int]:
    """The five quantiles for hour h from the comparables' log-log fit."""
    xs = [math.log(by_hour[1]) for by_hour in curves if h in by_hour]
    ys = [math.log(by_hour[h]) for by_hour in curves if h in by_hour]
    if len(xs) < MIN_FIT_POINTS:
        median_log = log_p1
        sigma = FALLBACK_SIGMA
    else:
        a, b, resid = ols(xs, ys)
        median_log = a + b * log_p1
        dof = max(1, len(resid) - 2)
        sigma = max(math.sqrt(sum(r * r for r in resid) / dof), SIGMA_FLOOR)
    row = [round2(math.exp(median_log + z * sigma)) for z in Z]
    row = [max(1.0, min(MAX_VIEWERS, float(v))) for v in row]
    for i in range(1, 5):
        row[i] = max(row[i], row[i - 1])
    return row


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    log_p1 = math.log(p1)
    curves = []
    for comp in case.get("comparables") or []:
        by_hour = release_day_hours(comp)
        if by_hour.get(1, 0) >= MIN_HOUR1_VIEWERS:
            curves.append(by_hour)
    # Sort by hour: the output rows are hours 2..24 each once, in ascending
    # order, whatever order the case's `forecast` list them in.
    rows = sorted(case["forecast"], key=lambda r: r["hour"])
    return {
        "hours": [
            {"hour": row["hour"], "q": hour_quantiles(log_p1, curves, row["hour"])}
            for row in rows
        ]
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
