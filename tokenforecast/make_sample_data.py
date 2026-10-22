"""Synthetic extract for smoke-testing the pipeline. Not representative of real usage.

Usage: python -m tokenforecast.make_sample_data sample.csv
"""
import sys

import numpy as np
import pandas as pd


def make(seed=0, n_weeks=43):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2025-12-29", periods=n_weeks, freq="W-MON")
    teams = {"org_a": ["payments", "risk", "platform"], "org_b": ["search", "ads", "data"]}
    models = {"model_large": 0.3, "model_mid": 0.5, "model_small": 0.2}
    ratios = {"output": 0.25, "cache_read": 3.0, "cache_write": 0.4}
    rows = []
    sid = 0
    for org, ts in teams.items():
        for team in ts:
            for _ in range(rng.integers(20, 45)):
                sid += 1
                base = rng.lognormal(13, 1.0)
                active_p = rng.choice([0.95, 0.6, 0.25], p=[0.6, 0.3, 0.1])
                start = rng.integers(0, n_weeks - 2) if rng.random() < 0.15 else 0
                growth = rng.normal(0.01, 0.01)
                moved_at = rng.integers(10, 30) if rng.random() < 0.03 else None
                for w, wk in enumerate(weeks):
                    if w < start or rng.random() > active_p:
                        continue
                    cur_team = ts[(ts.index(team) + 1) % len(ts)] if moved_at and w >= moved_at else team
                    vol = base * np.exp(growth * w) * rng.lognormal(0, 0.35)
                    cache_mult = 4.0 if team == "platform" and w >= 30 else 1.0
                    for m, share in models.items():
                        inp = vol * share * rng.lognormal(0, 0.2)
                        vals = {"input": inp}
                        for t, r in ratios.items():
                            vals[t] = inp * r * (cache_mult if t.startswith("cache") else 1) * rng.lognormal(0, 0.25)
                        for t, v in vals.items():
                            rows.append((f"S{sid:05d}", wk.date(), org, cur_team, m, t, round(v)))
    return pd.DataFrame(rows, columns=["SID", "date", "org", "team", "model", "token_type", "tokens"])


if __name__ == "__main__":
    make().to_csv(sys.argv[1] if len(sys.argv) > 1 else "sample.csv", index=False)
