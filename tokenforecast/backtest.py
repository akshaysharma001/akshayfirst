"""Step 2: rolling-origin backtest of baselines, scored on calendar-month totals.

Usage: python -m tokenforecast.backtest data.csv [--prices prices.csv] [--out results.csv] [--no-restate]
"""
import argparse

import numpy as np
import pandas as pd

from . import config as C
from .data import add_value, load, panel, restate_to_current_team, weekly_to_monthly


def naive(y, h):
    return np.repeat(y[-1:], h, axis=0)


def mean_k(k):
    def f(y, h):
        return np.repeat(y[-k:].mean(axis=0, keepdims=True), h, axis=0)
    return f


def linear(k):
    def f(y, h):
        y = y[-k:]
        x = np.arange(len(y), dtype=float)
        xm, ym = x.mean(), y.mean(axis=0)
        slope = ((x - xm)[:, None] * (y - ym)).sum(axis=0) / ((x - xm) ** 2).sum()
        fx = np.arange(len(y), len(y) + h, dtype=float)
        return np.clip(ym + slope * (fx - xm)[:, None], 0, None)
    return f


def damped_holt(alpha=0.3, beta=0.1, phi=0.9):
    def f(y, h):
        level, trend = y[0].astype(float), (y[1] - y[0]).astype(float)
        for t in range(1, len(y)):
            prev = level
            level = alpha * y[t] + (1 - alpha) * (level + phi * trend)
            trend = beta * (level - prev) + (1 - beta) * phi * trend
        damp = np.cumsum(phi ** np.arange(1, h + 1))
        return np.clip(level + damp[:, None] * trend, 0, None)
    return f


METHODS = {
    "naive": naive,
    "mean4": mean_k(4),
    "mean8": mean_k(8),
    "linear12": linear(12),
    "damped_holt": damped_holt(),
}


def run(df, prices=None, restate=True):
    if restate:
        df = restate_to_current_team(df)
    weeks = pd.DatetimeIndex(np.sort(df["week"].unique()))
    n, h = len(weeks), C.HORIZON_WEEKS
    origins = list(range(C.FIRST_TRAIN_WEEKS, n - h + 1, C.ORIGIN_STEP_WEEKS))
    if not origins:
        raise ValueError(f"Need >= {C.FIRST_TRAIN_WEEKS + h} weeks; have {n}")
    print(f"{n} weeks, horizon {h}, origins at weeks {origins} (overlapping windows: treat as indicative)")

    targets = [t for t in C.TOKEN_TYPES if t in set(df.token_type)] + ["all_tokens"]
    if prices is not None:
        targets.append("cost")

    rows = []
    for target in targets:
        vdf = add_value(df, target, prices)
        for level, cols in C.LEVELS.items():
            p = panel(vdf, cols, weeks)
            for o in origins:
                train, test = p.iloc[:o].to_numpy(), p.iloc[o:o + h]
                actual_m = weekly_to_monthly(test)
                for name, method in METHODS.items():
                    fc = pd.DataFrame(method(train, h), index=test.index, columns=test.columns)
                    fc_m = weekly_to_monthly(fc)
                    rows.append({
                        "target": target, "level": level, "origin": o, "method": name,
                        "abs_err": float((fc_m - actual_m).abs().to_numpy().sum()),
                        "err": float((fc_m - actual_m).to_numpy().sum()),
                        "actual": float(actual_m.to_numpy().sum()),
                    })
    return pd.DataFrame(rows)


def summarise(res):
    g = res.groupby(["target", "level", "method"])[["abs_err", "err", "actual"]].sum()
    g["wape"] = g["abs_err"] / g["actual"]
    g["bias"] = g["err"] / g["actual"]
    wape = g["wape"].unstack("method")
    wape["best"] = wape.idxmin(axis=1)
    return g, wape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--prices", help="CSV with model, token_type, usd_per_million")
    ap.add_argument("--out", help="write per-origin results CSV")
    ap.add_argument("--no-restate", action="store_true", help="keep historical team labels")
    args = ap.parse_args()

    prices = pd.read_csv(args.prices) if args.prices else None
    res = run(load(args.csv), prices, restate=not args.no_restate)
    g, wape = summarise(res)
    bias = g["bias"].unstack("method")
    level_order = {k: i for i, k in enumerate(C.LEVELS)}
    wape, bias = (t.sort_index(level="level", key=lambda s: s.map(level_order), sort_remaining=False)
                  for t in (wape, bias))
    print("\nMonthly WAPE (lower is better)\n")
    print(wape.round(3).to_markdown())
    print("\nMonthly bias (+ = over-forecast)\n")
    print(bias.round(3).to_markdown())
    if args.out:
        res.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
