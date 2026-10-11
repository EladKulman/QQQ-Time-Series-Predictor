#!/usr/bin/env python3
"""Run an offline results demo or a fresh Chronos-2 holdout forecast.

The default mode uses only Python's standard library. Paths are resolved from
this file, so the command also works when launched outside the repository.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess
import sys
import webbrowser

ROOT = Path(__file__).resolve().parent
SAVED_RUN = ROOT / "docs/results/news_ablation/gdelt_fed-seed42"
LABELS = {"base_pretrained": "Pretrained Chronos-2", "fine_tuned": "LoRA fine-tuned",
          "zero_gaussian": "Zero-mean Gaussian", "pretrained_demo": "Pretrained Chronos-2"}


def load_forecasts(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"No forecasts found in {path}")
    for row in rows:
        for key in list(row):
            if key in {"window", "horizon"}:
                row[key] = int(row[key])
            elif key in {"prediction", "actual"} or key.startswith("q"):
                row[key] = float(row[key])
                if not math.isfinite(row[key]):
                    raise ValueError(f"Non-finite {key} in {path}")
        required = {"model", "window", "horizon", "forecast_origin", "date", "prediction", "actual", "q0.1", "q0.9"}
        if not required <= row.keys():
            raise ValueError(f"Missing forecast columns: {sorted(required - row.keys())}")
        if not row["forecast_origin"] < row["date"]:
            raise ValueError("Forecast origin must precede every predicted session")
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Compute scores from quantiles and actuals, never from saved error columns."""
    result = []
    for model in dict.fromkeys(row["model"] for row in rows):
        group = [r for r in rows if r["model"] == model]
        quantiles = sorted([k for k in group[0] if k.startswith("q")], key=lambda k: float(k[1:]))
        n = len(group)
        errors = [r["prediction"] - r["actual"] for r in group]
        pinball = sum(max(float(q[1:]) * (r["actual"] - r[q]),
                          (float(q[1:]) - 1) * (r["actual"] - r[q]))
                      for r in group for q in quantiles) / (n * len(quantiles))
        denominator = sum(abs(r["actual"]) for r in group) / n
        result.append({"model": model, "label": LABELS.get(model, model), "rows": n,
                       "quantiles": len(quantiles),
                       "weighted_quantile_loss": 2 * pinball / denominator if denominator else None,
                       "mae": sum(abs(e) for e in errors) / n,
                       "rmse": math.sqrt(sum(e * e for e in errors) / n),
                       "coverage_80": sum(r["q0.1"] <= r["actual"] <= r["q0.9"] for r in group) / n})
    return result


def run_pretrained(args, output: Path) -> tuple[list[dict], dict]:
    plan = json.loads((ROOT / "configs/news_ablation.json").read_text())
    raw = output / "pretrained_forecast.csv"
    command = [sys.executable, str(ROOT / "scripts/predict_chronos2.py"),
               "--pretrained", "--model-id", args.model_id,
               "--model-revision", plan["model_revision"], "--device-map", args.device,
               "--prediction-length", "5", "--holdout-rows", "5",
               "--context-length", "64", "--max-context-rows", "64",
               "--feature-profile", "core", "--output", str(raw)]
    if args.prepare_only:
        command.append("--prepare-only")
    subprocess.run(command, cwd=ROOT, check=True)
    if args.prepare_only:
        return [], {}
    with (ROOT / "data/processed/daily_feature_table.csv").open(newline="") as source:
        history = list(csv.DictReader(source))
    with raw.open(newline="") as source:
        forecasts = list(csv.DictReader(source))
    for horizon, row in enumerate(forecasts, start=1):
        row.update(model="pretrained_demo", window=0, horizon=horizon, forecast_origin=history[-6]["date"])
    normalized = output / "forecasts.csv"
    write_csv(normalized, forecasts)
    return load_forecasts(normalized), {"model_id": args.model_id,
            "model_revision": plan["model_revision"], "context_length": 64,
            "feature_profile": "core", "device": args.device,
            "data_sha256": hashlib.sha256((ROOT / "data/processed/daily_feature_table.csv").read_bytes()).hexdigest()}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["saved", "pretrained"], default="saved",
                        help="saved: offline replay (default); pretrained: run the model")
    parser.add_argument("--output-dir", type=Path, help="Default: output/demo/<mode> under this repository")
    parser.add_argument("--window", type=int, default=0, help="Initial saved forecast window to display (0–49)")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="cpu")
    parser.add_argument("--model-id", default="amazon/chronos-2", help="Hugging Face model ID or local snapshot")
    parser.add_argument("--prepare-only", action="store_true", help="Validate pretrained inputs without loading weights")
    parser.add_argument("--open", action="store_true", help="Open the generated HTML in your browser")
    args = parser.parse_args(argv)
    if args.prepare_only and args.mode != "pretrained":
        parser.error("--prepare-only requires --mode pretrained")
    if args.window < 0 or args.window > (49 if args.mode == "saved" else 0):
        parser.error("--window must be 0–49 for saved mode, or 0 for pretrained mode")
    output = (args.output_dir or ROOT / "output/demo" / args.mode).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "saved":
        rows = load_forecasts(SAVED_RUN / "validation_predictions.csv")
        metadata = json.loads((SAVED_RUN / "comparison_metadata.json").read_text())
        metadata["predictions_sha256"] = hashlib.sha256((SAVED_RUN / "validation_predictions.csv").read_bytes()).hexdigest()
        title = "Saved experiment · offline demo"
        description = ("Archived GDELT Fed screen, seed 42: pretrained Chronos-2, its LoRA adapter, "
                       "and a zero-mean Gaussian baseline on 250 held-out sessions. "
                       "This mode replays saved predictions and recomputes scores; it does not run the model. "
                       "This exploratory screen used an earlier training recipe; see the report for the controlled four-year comparison.")
    else:
        rows, metadata = run_pretrained(args, output)
        if args.prepare_only:
            return 0
        title = "Fresh pretrained inference · historical holdout"
        description = ("New Chronos-2 inference using 64 context sessions and the core covariates. "
                       "The final five sessions of the bundled dataset are held out. "
                       "Future targets and observed news are excluded from model inputs. "
                       "These five points demonstrate execution, not model quality.")
    metrics = summarize(rows)
    summary = {"mode": args.mode, "description": description, "source": metadata, "metrics": metrics}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(output / "forecasts.csv", rows)
    payload = {"rows": rows, "metrics": metrics, "initialWindow": args.window}
    template = (ROOT / "demo/report_template.html").read_text(encoding="utf-8")
    rendered = template.replace("__TITLE__", html.escape(title)).replace("__DESCRIPTION__", html.escape(description))
    rendered = rendered.replace("__PAYLOAD__", json.dumps(payload, allow_nan=False).replace("<", "\\u003c"))
    page = output / "index.html"
    page.write_text(rendered, encoding="utf-8")
    print(f"\n{title}\n{description}\n")
    for metric in metrics:
        wql = metric['weighted_quantile_loss']
        wql_label = f"{wql:.6f}" if wql is not None else "undefined"
        print(f"{metric['label']:<24} WQL={wql_label}  MAE={metric['mae']:.6f}  80% coverage={metric['coverage_80']:.1%}")
    print(f"\nOpen: {page}\nData: {output / 'forecasts.csv'}\nScores: {output / 'metrics.json'}")
    if args.open:
        webbrowser.open(page.as_uri())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Demo failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
