"""Verify the no-install demo, score parity, and historical forecast integrity."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qqq_demo", ROOT / "demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class DemoTests(unittest.TestCase):
    def test_offline_demo_works_without_site_packages_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "demo output"
            result = subprocess.run([sys.executable, "-I", "-S", str(ROOT / "demo.py"),
                                     "--output-dir", str(output), "--window", "49"],
                                    cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(demo.load_forecasts(output / "forecasts.csv")), 750)
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(metrics["mode"], "saved")
            self.assertEqual(len(metrics["metrics"]), 3)
            page = (output / "index.html").read_text()
            self.assertNotIn("__PAYLOAD__", page)
            self.assertNotIn("__TITLE__", page)
            self.assertIn('"initialWindow": 49', page)
            self.assertNotIn('<script src=', page)

    def test_recomputed_scores_match_original_evaluation(self):
        rows = demo.load_forecasts(demo.SAVED_RUN / "validation_predictions.csv")
        with (demo.SAVED_RUN / "validation_metrics.csv").open() as source:
            expected = {r["model"]: r for r in csv.DictReader(source) if r["horizon"] == "overall"}
        self.assertEqual(set(expected), {"base_pretrained", "fine_tuned", "zero_gaussian"})
        for metric in demo.summarize(rows):
            saved = expected[metric["model"]]
            for field, source_field in [("weighted_quantile_loss", "weighted_quantile_loss"),
                                        ("mae", "mae_log_return"), ("rmse", "rmse_log_return"),
                                        ("coverage_80", "q10_q90_coverage")]:
                self.assertAlmostEqual(metric[field], float(saved[source_field]), places=12)
            self.assertEqual(metric["rows"], 250)

    def test_saved_actuals_and_forecast_origins_match_the_bundled_data(self):
        with (ROOT / "data/processed/daily_feature_table.csv").open() as source:
            history = list(csv.DictReader(source))
        indices = {r["date"]: i for i, r in enumerate(history)}
        for row in demo.load_forecasts(demo.SAVED_RUN / "validation_predictions.csv"):
            origin = indices[row["forecast_origin"]]
            target = history[origin + row["horizon"]]
            self.assertEqual(row["date"], target["date"])
            self.assertAlmostEqual(row["actual"], float(target["log_return_1d"]), places=12)
            self.assertLessEqual(row["q0.1"], row["prediction"])
            self.assertLessEqual(row["prediction"], row["q0.9"])


if __name__ == "__main__":
    unittest.main()
