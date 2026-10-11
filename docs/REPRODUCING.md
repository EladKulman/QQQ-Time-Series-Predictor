# Detailed reproduction guide

Workshop project by Elad Kulman and Tom Weitman. The current experiment forecasts QQQ
one-day log returns at horizons of one to five NYSE sessions, using pretrained Chronos-2,
LoRA fine-tuning, and feature-group ablations. Each forecast origin is the NYSE close.

This guide covers the data pipeline, training, prediction, and experiment evaluation
used in the accompanying paper. See `docs/DATA_NOTES.md` for source timing and limitations.

All shell commands run from the repository root. Start with [the quick start](../README.md).

## Reproducing the experiments

The modeling data is included under `data/processed/`, so the steps below do not need
data-provider API keys. Model execution requires the pinned Chronos-2 weights, downloaded
on first use or supplied from a local cache.

### 1. Install dependencies and run the tests (no model, no GPU)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

### 2. Recompute evaluation statistics from saved results (no GPU)

Per-window losses, predictions and metrics of every experiment are committed under
`docs/results/`. These scripts read them and print or redraw the headline statistics:

```bash
python scripts/ladder_paired_deltas.py        # zero-shot ladder deltas versus the control (report section 7, figure 1 dotted curves)
python scripts/analyze_gain_conditions.py     # fine-tuning gain by volatility tercile and horizon, control minus Gaussian (section 9.2, table 7)
```

The summary files in `docs/results/` (`walk_forward_summary.md`, `ladder_summary.md`,
`importance_summary.csv`) were produced by `summarize_walk_forward.py`, `summarize_ladder.py`
and `summarize_permutation_importance.py` from the raw run directories under `models/`. Those
directories hold adapters and are not in the repository; the summaries they produced are.

| Results document | Backs report section |
|---|---|
| `docs/NEWS_ABLATION_RESULTS.md`, `docs/GDELT_TOPIC_RESULTS.md` | 6, first screen and replication |
| `docs/FINETUNED_LADDER_RESULTS.md` | 7 and 9, four-year ladder and what fine-tuning changes |
| `docs/PRETRAINED_LADDER_RESULTS.md` | 7, pretrained curves |
| `docs/PERMUTATION_PRETRAINED_RESULTS.md` | 8, swap test on the pretrained model |

### 3. Rerun the pretrained-only experiments on a laptop (inference only)

The zero-shot ladder and the swap test need no training. They download the pinned Chronos-2
revision from Hugging Face on first use and run on Apple MPS or CPU. The ladder took about
8 minutes on a Mac with MPS.

```bash
python scripts/run_pretrained_ladder.py                       # writes models/pretrained-ladder/
python scripts/summarize_ladder.py --root models/pretrained-ladder
COLS="$(python -c "import json;print(','.join(json.load(open('docs/results/permutation_pretrained/metadata.json'))['feature_columns']))")"
python scripts/permutation_importance.py --end-date 2026-06-30 --validation-rows 252 --repeats 3 \
  --feature-columns "$COLS" --output-dir models/permutation-pretrained
python scripts/summarize_permutation_importance.py models/permutation-pretrained
```

The swap test above repeats the report's pretrained run: the 42 all-external columns, 50 windows
of five sessions from June 30, 2025 to June 30, 2026, three donor draws per feature. Add
`--prepare-only` to any command in this section to check the plan without loading the model.

### 4. Rerun the fine-tuning (GPU cluster)

The 126 LoRA fits ran as Slurm array jobs on the TAU cluster, one RTX 2080 Ti with 11 GB per
task. The job scripts hardcode that cluster's account, partition and GPU constraint, so edit
the `#SBATCH` lines for another cluster. `slurm/README.md` covers staging, the offline model
cache (`scripts/download_chronos.py`) and the PyTorch wheel that matches the cluster driver.

```bash
sbatch slurm/chronos_walk_forward.sbatch                # 8 rungs x 4 folds x 3 seeds, configs/walk_forward.json
sbatch slurm/chronos_walk_forward_pretrained.sbatch     # pretrained and Gaussian arms on the same windows
WALK_FORWARD_ROOT=models/walk-forward-<jobid> sbatch slurm/chronos_permutation.sbatch
```

The 42-feature rung and its matched control used `WALK_FORWARD_EXTRA_ARGS="--train-batch-windows 4 --num-steps 1000"`
because eight all-external windows exceed 11 GB. `scripts/run_walk_forward.py --list-tasks` prints the
task index to (rung, fold, seed) map. `slurm/chronos_news_screen_v2.sbatch` was prepared for a rerun of
the first-round screen with the corrected batch; that rerun was not executed and has no results.

## Status (2026-10-10)

- Data layer complete: 5,201 sessions, 79 columns, publication-aware alignment, leakage audit
  and 48 history checks pass. See `docs/DATA_NOTES.md`.
- Round A (first screen): 18 LoRA fits, one seed and one evaluation year, plus three setups at
  three seeds. Results in `docs/NEWS_ABLATION_RESULTS.md` and `docs/GDELT_TOPIC_RESULTS.md`.
- Round B (walk-forward ladder): 108 LoRA fits, eight rungs, folds 2022 to 2025, three seeds,
  plus pretrained arms and swap tests on pretrained and fine-tuned checkpoints. Results in
  `docs/FINETUNED_LADDER_RESULTS.md`, `docs/PRETRAINED_LADDER_RESULTS.md` and
  `docs/PERMUTATION_PRETRAINED_RESULTS.md`.

## Setup

Use Python 3.11 or 3.12. Create a local environment, then install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

FRED downloads require `FRED_API_KEY`; SEC downloads require a real contact address in
`SEC_USER_AGENT_EMAIL`. The saved FRED/SEC snapshots support `--cached` replay without keys.
Keep `.env`, local connection notes in `oath/`, and model files out of version control.

## Refresh the data

Run from the repository root using the environment's Python:

```bash
# 2005 price history warms up the rolling features; the modeling sample begins in 2006.
python scripts/fetch_qqq_prices.py --start 2005-01-01
python scripts/fetch_cross_asset_prices.py
python scripts/fetch_cboe_vol_indices.py
python scripts/fetch_epu.py
python scripts/fetch_gdelt_news.py --start 2017-01-01
python scripts/fetch_fomc_statements.py --start 2006-01-01
python scripts/score_fomc_tone.py
python scripts/fetch_fred_macro.py --start 2006-01-01
python scripts/fetch_sec_filings.py --start 2006-01-01
python scripts/build_event_calendar.py
python scripts/build_daily_feature_table.py
python scripts/check_data_readiness.py --output data/processed/data_readiness.json
python scripts/validate_feature_table.py
python scripts/audit_processed_data.py --leakage
```

GDELT extends each topic's cache with an overlapping tail and saves every successful API
response. HTTP 429 responses trigger backoff; refreshes can take much longer than half an
hour. Use `--refresh --start YYYY-MM-DD --end YYYY-MM-DD` to revisit a historical interval.
Use `--topics fed inflation` to target retries, or `--cached` to rebuild from completed topic caches. An API outage remains missing, never a fabricated zero. Do not delete the raw caches.

For an offline replay of the repaired Fed/FRED/SEC/calendar sources:

```bash
python scripts/fetch_fomc_statements.py --cached --end 2026-09-07
python scripts/fetch_fred_macro.py --cached --end 2026-09-07
python scripts/fetch_sec_filings.py --cached --end 2026-09-07
python scripts/build_event_calendar.py --cached --end 2027-01-05
python scripts/build_daily_feature_table.py
```

Tone scoring needs the cached classifier or a first model download. It pins the tested
classifier revision and writes source/model hashes to `fomc_tone_daily.metadata.json`.
`fetch_vix_cboe.py` and `vix_daily.csv` are legacy; the builder uses `cboe_vol_daily.csv`.

## Feature and time contract

The builder writes `data/processed/daily_feature_table.csv` and its matching
`daily_feature_table_groups.json`. A build metadata file hashes every source so a source
refresh cannot silently leave an outdated modeling table in use. The full schema has one target, six raw OHLCV columns,
57 past covariates, and 14 known-future calendar covariates, plus `date` (79 columns total).
The core model selects 21 past and six known-future covariates. Raw OHLCV is never selected
as model covariates. SEC earnings counts are past observations, not scheduled earnings dates.

Use `--feature-profile core`, `full`, or `target` in the modeling scripts. Combine
`--feature-profile full --feature-groups qqq,calendar,market` for a group ablation, or pass
`--feature-columns` explicitly. The roles file validates selection and future inputs.
`--profile core` on the builder can optionally write a smaller table.

No modeling script drops sessions because a covariate is NaN. Invalid dates, missing
returns, and incomplete known-future inputs fail clearly. Forecast dates use the NYSE
calendar, including holidays. Unknown schedules beyond their published horizon remain NaN.
The readiness gate checks source tails and the exact next five exchange sessions; it
cannot prove every historical observation was available at the forecast origin.

## Pretrained baseline and LoRA

First inspect the split and features without loading a model:

```bash
python scripts/compare_chronos2_validation.py --smoke-test --prepare-only
python scripts/compare_chronos2_validation.py --feature-profile core --prepare-only
```

Run a small local comparison and reload its adapter:

```bash
python scripts/compare_chronos2_validation.py --smoke-test --device-map cpu
python scripts/summarize_chronos2_comparison.py
python scripts/predict_chronos2.py \
  --checkpoint models/chronos2-validation-comparison/finetuned/finetuned-ckpt \
  --holdout-rows 5 --device-map cpu
```

Comparison smoke defaults: 512 total rows, final 60 for validation, five three-session
windows, 64-session model context, five optimizer steps, six covariates. The separate
`fine_tune_chronos2.py --smoke-test` command has 384-row defaults and saves under
`models/chronos2-qqq-smoke/`.

A pretrained target-only arm uses `--feature-profile target --base-only`. Live pretrained
prediction uses `predict_chronos2.py --pretrained --prediction-length 5`. For the actual
GPU smoke test and first full comparison, follow `slurm/README.md`. The controlled
news-source screen is specified in `docs/NEWS_ABLATION_PLAN.md` and
`configs/news_ablation.json`; completed results are in `docs/NEWS_ABLATION_RESULTS.md`.

Full comparison defaults reserve the final 252 sessions, train LoRA on earlier rows only,
and evaluate pretrained, fine-tuned, and zero-mean Gaussian forecasts on identical windows.
With stride five, 50 complete windows score 250 sessions. Primary score: weighted quantile
loss over 21 quantiles. Also report MAE, RMSE, direction, and 80%/98% interval coverage;
coverage should approach its nominal level. Narrower intervals alone are not better.

Outputs include predictions, metrics, checkpoint metadata, data/role/script hashes,
model revision, seed, and package versions. Use development validation to choose settings;
reserve an untouched test interval before making final model-selection claims. Adapters
from the old feature schema must be retrained.

## Checks and directories

```bash
python -m unittest discover -s tests -v
bash -n slurm/chronos_smoke.sbatch
bash -n slurm/chronos_validation.sbatch
bash -n slurm/chronos_news_ablation.sbatch
bash -n slurm/chronos_news_robustness.sbatch
bash -n slurm/chronos_gdelt_topic_ablation.sbatch
bash -n slurm/chronos_walk_forward.sbatch
bash -n slurm/chronos_walk_forward_pretrained.sbatch
bash -n slurm/chronos_permutation.sbatch
bash -n slurm/chronos_news_screen_v2.sbatch
```

- `data/raw/`: downloaded observations, source schedules, statement text, and labels.
- `data/processed/`: source tables, event calendar, modeling table, and readiness report.
- `scripts/`: fetchers, transformations, validation, training, comparison, and prediction.
- `tests/`: time alignment, missingness, and future-input isolation regression checks.
- `slurm/`: single-GPU jobs and cluster setup instructions.
- `models/`: ignored model caches, adapters, predictions, and run reports.
