import numpy as np
import pandas as pd

from . import config as C

REQUIRED = ["date", "org", "team", "sid", "model", "token_type", "tokens"]


def load(path):
    df = pd.read_csv(path)
    df = df.rename(columns={src: canon for canon, src in C.COLUMNS.items() if src in df.columns})

    if "token_type" not in df.columns:
        wide = [c for c in C.WIDE_TOKEN_COLUMNS if c in df.columns]
        if not wide:
            raise ValueError("No token_type column and no WIDE_TOKEN_COLUMNS found; check config.py")
        ids = [c for c in ["date", "org", "team", "sid", "model"] if c in df.columns]
        df = df.melt(id_vars=ids, value_vars=wide, var_name="token_type", value_name="tokens")
        df["token_type"] = df["token_type"].map(C.WIDE_TOKEN_COLUMNS)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after mapping: {missing}; check config.py")

    df = df[REQUIRED].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"] - pd.Timedelta(days=6) if C.WEEK_DATE_IS == "end" else df["date"]
    tt = df["token_type"].astype(str).str.strip().str.lower()
    df["token_type"] = tt.map(lambda t: C.TOKEN_TYPE_ALIASES.get(t, t))
    df["tokens"] = pd.to_numeric(df["tokens"], errors="coerce")
    for c in ["org", "team", "sid", "model"]:
        df[c] = df[c].astype(str)
    return df


def current_assignment(df):
    """Each SID's org/team in its latest active week (ties broken by token volume)."""
    last = df[df["week"] == df.groupby("sid")["week"].transform("max")]
    return (
        last.groupby(["sid", "org", "team"])["tokens"].sum().reset_index()
        .sort_values("tokens").drop_duplicates("sid", keep="last")[["sid", "org", "team"]]
    )


def restate_to_current_team(df):
    """Move each SID's full history under its current team so reorgs don't look like usage shifts."""
    cur = current_assignment(df)
    return df.drop(columns=["org", "team"]).merge(cur, on="sid", how="left")


def add_value(df, target, prices=None):
    """Value column for a target: a token type, 'all_tokens', or 'cost' (needs prices)."""
    if target == "cost":
        merged = df.merge(prices, on=["model", "token_type"], how="left")
        unpriced = merged["usd_per_million"].isna()
        if unpriced.any():
            pairs = merged.loc[unpriced, ["model", "token_type"]].drop_duplicates()
            print(f"WARNING: {len(pairs)} model/token_type pairs have no price; excluded from cost:\n{pairs}")
        merged = merged[~unpriced]
        return merged.assign(value=merged["tokens"] * merged["usd_per_million"] / 1e6)
    if target == "all_tokens":
        return df.assign(value=df["tokens"])
    return df[df["token_type"] == target].assign(value=lambda d: d["tokens"])


def panel(df, level_cols, weeks):
    """Dense weeks x series matrix, zero-filled."""
    p = df.pivot_table(index="week", columns=level_cols, values="value", aggfunc="sum")
    return p.reindex(weeks).fillna(0.0)


def weekly_to_monthly(p):
    """Split each week across its 7 days (weekday-weighted) and sum into calendar months."""
    starts = p.index.values.astype("datetime64[D]")
    days = pd.DatetimeIndex((starts[:, None] + np.arange(7)).ravel())
    w = np.where(days.weekday >= 5, C.WEEKEND_WEIGHT, 1.0).reshape(-1, 7)
    w = w / w.sum(axis=1, keepdims=True)
    vals = (p.to_numpy()[:, None, :] * w[:, :, None]).reshape(-1, p.shape[1])
    daily = pd.DataFrame(vals, index=days, columns=p.columns)
    return daily.groupby(daily.index.to_period("M")).sum()
