import unittest

import numpy as np

from app.metrics import metrics


class MetricsTests(unittest.TestCase):
    def test_negative_and_out_of_range_outputs_are_in_the_overall_error(self):
        result = metrics([-10, 20, 400], [10, 10, 100], [0, 1, 1], bootstrap_samples=100)
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["finite_n"], 3)
        self.assertEqual(result["out_of_range_n"], 2)
        self.assertEqual(result["mae_months"], 110.0)
        self.assertEqual(result["signed_bias_months"], 96.6667)
        self.assertTrue(result["complete"])
        self.assertEqual(result["by_sex"]["F"]["mae_months"], 20)
        self.assertEqual(result["by_sex"]["F"]["mae_ci95_months"], [20, 20])

    def test_missing_and_nonfinite_cases_do_not_disappear_from_denominator(self):
        result = metrics([10, None, np.inf], [10, 12, 13], ["F", "M", "M"], bootstrap_samples=50)
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["finite_n"], 1)
        self.assertEqual(result["nonfinite_prediction_n"], 2)
        self.assertEqual(result["invalid_n"], 2)
        self.assertFalse(result["complete"])
        self.assertEqual(result["mae_months"], 0)
        self.assertEqual(result["by_sex"]["M"]["n"], 2)
        self.assertIsNone(result["by_sex"]["M"]["mae_months"])

    def test_bootstrap_is_reproducible_and_nontrivial(self):
        args = ([0, 4, 15, 20, 40], [1, 5, 10, 40, 70], [0, 0, 1, 1, 1])
        first = metrics(*args, bootstrap_seed=20, bootstrap_samples=200)
        self.assertEqual(first, metrics(*args, bootstrap_seed=20, bootstrap_samples=200))
        lo, hi = first["mae_ci95_months"]
        self.assertLess(lo, first["mae_months"])
        self.assertGreater(hi, first["mae_months"])
        self.assertEqual(first["bootstrap"]["unit"], "image")
        self.assertEqual(first["by_boneage"]["0-24"]["n"], 3)

    def test_empty_and_nonfinite_labels_are_explicit(self):
        empty = metrics([], [], [], bootstrap_samples=1)
        self.assertEqual(empty["n"], 0)
        self.assertFalse(empty["complete"])
        self.assertIsNone(empty["mae_ci95_months"])
        result = metrics([1, 2], [np.nan, 2], ["unknown", "F"], bootstrap_samples=10)
        self.assertEqual(result["nonfinite_label_n"], 1)
        self.assertEqual(result["unknown_sex_n"], 1)
        self.assertFalse(result["complete"])

    def test_length_and_bootstrap_configuration_are_validated(self):
        with self.assertRaises(ValueError):
            metrics([1, 2], [1], [1])
        with self.assertRaises(ValueError):
            metrics([1], [1], [1], bootstrap_samples=0)


if __name__ == "__main__":
    unittest.main()
