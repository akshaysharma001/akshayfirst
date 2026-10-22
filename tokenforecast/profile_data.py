"""Step 1: profile the extract. The numbers here set segment thresholds and model choices.

Usage: python -m tokenforecast.profile_data data.csv [--out profile.md]
"""
import argparse

import numpy as np
import pandas as pd

from . import config as C
from .data import load


def classify(series):
    """Syntetos-Boylan demand class on weeks from first use onward."""
    s = series[series.cumsum() > 0]
    nz = s[s > 0]
    if len(nz) < 2:
        return "too_short"
    adi = len(s) / len(nz)
    cv2 = (nz.std(ddof=0) / nz.mean()) ** 2
    if adi < 1.32:
        return "smooth" if cv2 < 0.49 else "erratic"
    return "intermittent" if cv2 < 0.49 else "lumpy"


def ratio_cv(df, keys, num, den="input"):
    w = df[df["token_type"].isin([num, den])].pivot_table(
        index=keys + ["week"], columns="token_type", values="tokens", aggfunc="sum").fillna(0)
    if num not in w or den not in w:
        return np.nan
    w = w[w[den] > 0]
    r = (w[num] / w[den]).groupby(level=keys)
    stats = r.agg(["mean", "std", "count"])
    stats = stats[(stats["count"] >= 4) & (stats["mean"] > 0)]
    return float((stats["std"] / stats["mean"]).median()) if len(stats) else np.nan


def profile(df):
    out = []
    p = out.append

    weeks = np.sort(df["week"].unique())
    gaps = np.diff(weeks).astype("timedelta64[D]").astype(int)
    p("# Data profile\n")
    p(f"- Rows: {len(df):,}; weeks: {len(weeks)} ({pd.Timestamp(weeks[0]).date()} to {pd.Timestamp(weeks[-1]).date()})")
    p(f"- Week-start weekdays seen: {sorted(pd.DatetimeIndex(weeks).day_name().unique())}")
    if (gaps != 7).any():
        p(f"- **Irregular spacing**: gaps of {sorted(set(gaps))} days. Fix before modelling.")
    p(f"- Orgs: {df.org.nunique()}, teams: {df.team.nunique()}, models: {df.model.nunique()}, SIDs: {df.sid.nunique()}")
    p(f"- Token types: {sorted(df.token_type.unique())}")

    p("\n## Data quality\n")
    unknown = sorted(set(df.token_type) - set(C.TOKEN_TYPES))
    p(f"- Null tokens: {df.tokens.isna().sum():,}; negative: {(df.tokens < 0).sum():,}")
    p(f"- Duplicate (week, sid, model, token_type) rows: {df.duplicated(['week', 'sid', 'model', 'token_type']).sum():,}")
    if unknown:
        p(f"- **Unrecognised token types** {unknown}: add to TOKEN_TYPE_ALIASES in config.py")

    df = df.dropna(subset=["tokens"])
    vol_type = "input" if "input" in set(df.token_type) else None
    vol = df[df.token_type == vol_type] if vol_type else df
    sid_week = vol.pivot_table(index="week", columns="sid", values="tokens", aggfunc="sum").reindex(weeks).fillna(0)

    p(f"\n## SID behaviour (volume = {vol_type or 'all tokens'})\n")
    first = (sid_week > 0).idxmax()
    last = (sid_week > 0)[::-1].idxmax()
    active_weeks = (sid_week > 0).sum()
    recent_cut, gone_cut = weeks[-8], weeks[-4]
    p(f"- SIDs with < 8 active weeks: {(active_weeks < 8).sum()} ({(active_weeks < 8).mean():.0%})")
    p(f"- New joiners (first use in last 8 weeks): {(first >= recent_cut).sum()}")
    p(f"- Possible leavers (no use in last 4 weeks): {(last < gone_cut).sum()}")
    since_first = sid_week.where(sid_week.index.values[:, None] >= first.values[None, :])
    p(f"- Zero weeks after first use: {(since_first == 0).sum().sum() / since_first.notna().sum().sum():.0%}")

    classes = sid_week.apply(classify).value_counts()
    p("- Demand classes (Syntetos-Boylan):")
    for k, v in classes.items():
        p(f"  - {k}: {v} ({v / classes.sum():.0%})")

    tot = sid_week.sum().sort_values(ascending=False)
    n = len(tot)
    p(f"- Concentration: top 1% of SIDs = {tot.iloc[:max(1, n // 100)].sum() / tot.sum():.0%} of tokens, "
      f"top 10% = {tot.iloc[:max(1, n // 10)].sum() / tot.sum():.0%}")

    p("\n## Team and org moves\n")
    moves = df.groupby("sid")[["org", "team"]].nunique()
    p(f"- SIDs seen in > 1 team: {(moves.team > 1).sum()}; in > 1 org: {(moves.org > 1).sum()}")
    if (moves.team > 1).any():
        p("- Restate history to current team before modelling (backtest does this by default).")

    p("\n## Token mix stability (median CV of weekly ratio vs input)\n")
    p("| ratio | per SID x model | per team x model |")
    p("|---|---|---|")
    for t in ["output", "cache_read", "cache_write"]:
        a, b = ratio_cv(df, ["sid", "model"], t), ratio_cv(df, ["team", "model"], t)
        p(f"| {t}/input | {a:.2f} | {b:.2f} |")
    p("\nCV above ~0.5 at SID level means SID ratios are noise: use team x model ratios for the split.")

    p("\n## Model mix by month (share of all tokens)\n")
    mix = df.assign(month=df.week.dt.to_period("M")).pivot_table(
        index="month", columns="model", values="tokens", aggfunc="sum").fillna(0)
    p((mix.div(mix.sum(axis=1), axis=0) * 100).round(1).to_markdown())

    p("\n## Org trend\n")
    org = df.pivot_table(index="week", columns="org", values="tokens", aggfunc="sum").reindex(weeks).fillna(0)
    for o in org.columns:
        y = org[o].to_numpy()
        pos = y > 0
        slope = np.polyfit(np.arange(len(y))[pos], np.log(y[pos]), 1)[0] if pos.sum() > 2 else np.nan
        p(f"- {o}: last 8w mean / first 8w mean = {y[-8:].mean() / max(y[:8].mean(), 1e-9):.2f}x; "
          f"log-linear weekly growth = {np.expm1(slope):+.1%}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out")
    args = ap.parse_args()
    report = profile(load(args.csv))
    print(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
