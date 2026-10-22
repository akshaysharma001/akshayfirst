# akshayfirst

## tokenforecast

SID-level token and cost forecasting for 3-month budgeting. The design is middle-out: the main forecast is at team x model level, then aggregated up to team and org and split down to SIDs.

Built so far:

1. **Profiling** (`profile_data.py`): data quality, gaps, new joiners and leavers, demand classes, concentration, team moves, token-mix stability, model mix and trend.
2. **Backtest harness** (`backtest.py`): rolling-origin backtest of baselines (naive, mean4, mean8, linear12, damped Holt) at org / team / team_model / SID level. It is scored on calendar-month WAPE and bias, per token type and for cost.

Weekly data is modelled weekly. Each week is split across its days (weekdays weighted over weekends) and summed into calendar months. SID histories are moved under their current team before modelling (`--no-restate` turns this off).

### Usage

```bash
pip install -r requirements.txt
# map your column names / token type labels in tokenforecast/config.py first
python -m tokenforecast.profile_data data.csv --out profile.md
python -m tokenforecast.backtest data.csv --prices prices.csv --out backtest.csv
```

Expected input is long format: `SID, date, org, team, model, token_type, tokens`. Wide format (one column per token type) also works; set `WIDE_TOKEN_COLUMNS` in `config.py`.

`prices.csv` has columns `model, token_type, usd_per_million`. Use your contracted rates. Pairs with no price are left out of cost, and the script prints a warning.

To smoke-test without real data: `python -m tokenforecast.make_sample_data sample.csv` (synthetic, not representative).

### Next

3. Team x model volume models, chosen on the backtest.
4. Token-mix ratios at team x model level, then cost.
5. Split down to SIDs using forecast shares.
6. MinT as a challenger, prediction ranges (never sum SID quantiles), scenario config.
