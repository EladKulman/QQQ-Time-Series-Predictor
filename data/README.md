# Bundled data

The processed modeling table contains **5,201 observed NYSE sessions**, from
2006-01-03 through 2026-09-04, and **79 columns**. `log_return_1d` is the target;
forecasts at horizons 1–5 refer to individual daily log returns, not cumulative returns.

- `processed/daily_feature_table.csv`: aligned target, raw OHLCV, and covariates.
- `processed/daily_feature_table_groups.json`: each column's group and information role.
- `processed/daily_feature_table.metadata.json`: hashes of source snapshots.
- `processed/event_calendar_daily.csv`: known-future schedule inputs.
- Other `processed/` files: source-specific daily tables and historical readiness output.
- `raw/`: source caches, schedules, statement text, and labels used by the builders.

Sources include Yahoo Finance (QQQ and cross-asset prices), CBOE, FRED/ALFRED,
Economic Policy Uncertainty, Federal Reserve releases, GDELT, SEC EDGAR, and BLS
schedules. The [data notes](../docs/DATA_NOTES.md) describe source attribution,
timing, alignment rules, missing coverage, and research limitations. These are
historical research snapshots; they are not a current market-data feed.

No credentials are needed for the demo, training, or evaluation using these files.
Refreshing FRED and SEC data requires your own configuration:

```bash
cp .env.example .env
# Fill FRED_API_KEY and SEC_USER_AGENT_EMAIL in .env locally.
```

Run that command from the repository root. Never commit `.env`. Network retrieval
also depends on provider availability and rate limits. See
[the reproduction guide](../docs/REPRODUCING.md) for download and cached-replay commands.
The source data and third-party model retain their respective provider terms;
this repository does not grant a separate license to those materials.
