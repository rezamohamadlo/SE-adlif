"""Pure paired reductions and temporary report artifacts; no model execution."""
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.adaptation_analysis.reporting import _pearson, _summarize, write_report


def row(sample=0, seed=42, condition="clean", replicate=0, benefit=1):
    return dict(seed=seed, condition=condition, replicate=replicate,
                sample_id=sample, target=sample % 20, speaker=sample % 10,
                parent_loss=2+benefit, variant_loss=2, benefit=benefit,
                parent_correct=0, variant_correct=1, correctness_difference=1,
                mean_rate_hz=sample, fast_slow_contrast=sample/10,
                autocorr_tau_ms=None, active_channel_fraction=.5, duration_ms=1000)


class ReportingTests(unittest.TestCase):
    def test_across_two_seeds_and_three_replicates_uses_seed_means(self):
        rows = []
        for replicate in range(3):
            for seed, count, benefit in [(42, 1, 10*replicate), (123, 3, 10*replicate+2)]:
                for sample in range(count):
                    value = row(sample, seed=seed, condition="jitter", replicate=replicate,
                                benefit=benefit)
                    value["variant_correct"] = int(seed == 123)
                    value["correctness_difference"] = value["variant_correct"]
                    rows.append(value)
        summary = _summarize(rows, "pc2", {"seeds": [42, 123]})
        aggregates = summary["across_seed_aggregates"]
        self.assertEqual(len(aggregates), 3)
        self.assertEqual(len(summary["groups"]), 6)
        for aggregate in aggregates:
            metrics = aggregate["metrics"]
            self.assertEqual(aggregate["n_training_seeds"], 2)
            self.assertTrue(aggregate["complete_training_seed_cohort"])
            self.assertEqual(aggregate["missing_training_seeds"], [])
            self.assertEqual(metrics["mean_loss_benefit"]["mean"], 10*aggregate["replicate"]+1)
            self.assertAlmostEqual(metrics["mean_loss_benefit"]["sample_sd"], math.sqrt(2))
            self.assertEqual(metrics["variant_accuracy"]["mean"], .5)
            self.assertAlmostEqual(metrics["variant_accuracy"]["sample_sd"], math.sqrt(.5))
            self.assertEqual(metrics["accuracy_difference_pp"]["mean"], 50)

    def test_missing_seed_is_partial_and_one_seed_sd_is_undefined(self):
        summary = _summarize([row()], "pc1", {"seeds": [42, 123]})
        aggregate = summary["across_seed_aggregates"][0]
        self.assertEqual(aggregate["n_training_seeds"], 1)
        self.assertFalse(aggregate["complete_training_seed_cohort"])
        self.assertEqual(aggregate["missing_training_seeds"], [123])
        self.assertTrue(all(value["sample_sd"] is None for value in aggregate["metrics"].values()))

    def test_each_seed_condition_replicate_stays_separate(self):
        rows = [row(0, benefit=1), row(1, benefit=3),
                row(0, seed=123, benefit=-1),
                row(0, condition="jitter", replicate=0, benefit=-2),
                row(0, condition="jitter", replicate=1, benefit=4)]
        report = _summarize(rows, "pc2", {})
        self.assertEqual(len(report["groups"]), 4)
        selected = next(group for group in report["groups"]
                        if group["seed"] == 42 and group["condition"] == "clean")
        self.assertEqual(selected["mean_loss_benefit"], 2)
        self.assertEqual(selected["n_validation_samples"], 2)
        self.assertEqual(selected["accuracy_difference_pp"], 100)
        self.assertEqual(report["n_unique_validation_samples"], 2)
        self.assertFalse(report["final_test_evidence"])
        self.assertFalse(report["architecture_selection"])

    def test_pearson_is_finite_clean_only_and_explicit_when_undefined(self):
        rows = [row(index, benefit=index+1) for index in range(3)]
        rows += [row(9, condition="jitter", benefit=-20)]
        report = _summarize(rows, "pc1", {})
        rate = next(value for value in report["feature_associations"]
                    if value["feature"] == "mean_rate_hz")
        self.assertEqual(rate["n_finite_pairs"], 3)
        self.assertAlmostEqual(rate["pearson_r"], 1)
        self.assertEqual(_pearson(rows[:2], "mean_rate_hz")["undefined_reason"],
                         "fewer_than_three_finite_pairs")
        self.assertEqual(_pearson(rows[:3], "duration_ms")["undefined_reason"],
                         "constant_feature_or_benefit")
        rows[0]["mean_rate_hz"] = float("nan")
        self.assertEqual(_pearson(rows[:3], "mean_rate_hz")["n_finite_pairs"], 2)

    def test_duplicate_rows_and_inconsistent_signs_are_rejected(self):
        with self.assertRaises(ValueError):
            _summarize([row(), row()], "pc1", {})
        bad = row(); bad["benefit"] = -1
        with self.assertRaises(ValueError):
            _summarize([bad], "pc1", {})
        with self.assertRaises(ValueError):
            _summarize([row(), row(replicate=1)], "pc1", {})

    def test_report_artifacts_all_pcs_are_headless_and_svg_has_no_trailing_space(self):
        rows = [row(index, benefit=index-1) for index in range(3)]
        with tempfile.TemporaryDirectory(prefix="adaptation-report-fixture-") as directory:
            for pc in ("pc1", "pc2", "pc3"):
                output = Path(directory) / pc
                paths = write_report(output, rows, pc, {"checkpoint": "results/example.ckpt"})
                self.assertEqual(len(paths), 4)
                self.assertTrue(all(path.is_file() and path.stat().st_size for path in paths))
                result = json.loads((output/"SUMMARY.json").read_text(encoding="utf-8"))
                self.assertEqual(result["metadata"]["checkpoint"], "results/example.ckpt")
                self.assertEqual(result["scope"], "validation_only")
                svg = next(path for path in paths if path.suffix == ".svg").read_text(encoding="utf-8")
                self.assertTrue(all(line == line.rstrip() for line in svg.splitlines()))


if __name__ == "__main__":
    unittest.main()
