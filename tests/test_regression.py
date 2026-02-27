"""
Unit tests for regression quality checker.

Tests RegressionChecker from regression/regression_checker.py,
which orchestrates RegressionMetrics.
"""

import unittest
import sys
from pathlib import Path

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from regression.regression_checker import RegressionChecker


class TestRegressionChecker(unittest.TestCase):
    """Test cases for RegressionChecker."""

    def setUp(self):
        """Initialize regression checker for each test."""
        self.checker = RegressionChecker()
        np.random.seed(42)
        self.n_samples = 100
        self.y_true = np.random.rand(self.n_samples) * 100
        self.noise = np.random.normal(0, 5, self.n_samples)
        self.y_pred = self.y_true + self.noise

    def test_check_quality(self):
        """Test quality check for regression."""
        metrics = self.checker.check_quality(
            y_true=self.y_true,
            y_pred=self.y_pred
        )

        self.assertIn('r_squared', metrics)
        self.assertIn('explained_variance', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('mse', metrics)

    def test_check_quality_with_cache(self):
        """Test that caching works correctly."""
        metrics_first = self.checker.check_quality(
            y_true=self.y_true,
            y_pred=self.y_pred,
            cache_key='reg_cache'
        )
        metrics_cached = self.checker.check_quality(
            y_true=self.y_true,
            y_pred=self.y_pred,
            cache_key='reg_cache'
        )

        self.assertEqual(metrics_first, metrics_cached)
        self.assertIn('reg_cache', self.checker.metrics_cache)

    def test_clear_cache(self):
        """Test that cache is cleared correctly."""
        self.checker.check_quality(
            y_true=self.y_true,
            y_pred=self.y_pred,
            cache_key='reg_cache'
        )
        self.checker.clear_cache()

        self.assertEqual(len(self.checker.metrics_cache), 0)

    def test_compute_r_squared(self):
        """Test R-squared computation."""
        r_squared = self.checker.compute_r_squared(self.y_true, self.y_pred)

        self.assertIsInstance(r_squared, float)
        # R-squared can be negative for very poor models
        self.assertLess(r_squared, 1.0)

    def test_compute_rmse(self):
        """Test RMSE computation."""
        rmse = self.checker.compute_rmse(self.y_true, self.y_pred)

        self.assertIsInstance(rmse, float)
        self.assertGreaterEqual(rmse, 0.0)

    def test_compute_mae(self):
        """Test MAE computation."""
        mae = self.checker.compute_mae(self.y_true, self.y_pred)

        self.assertIsInstance(mae, float)
        self.assertGreaterEqual(mae, 0.0)

    def test_compute_mse(self):
        """Test MSE computation."""
        mse = self.checker.compute_mse(self.y_true, self.y_pred)

        self.assertIsInstance(mse, float)
        self.assertGreaterEqual(mse, 0.0)

    def test_compute_explained_variance(self):
        """Test explained variance computation."""
        explained_var = self.checker.compute_explained_variance(self.y_true, self.y_pred)

        self.assertIsInstance(explained_var, float)
        self.assertLess(explained_var, 1.0)

    def test_rmse_greater_than_or_equal_mae(self):
        """Test that RMSE >= MAE (mathematical property)."""
        rmse = self.checker.compute_rmse(self.y_true, self.y_pred)
        mae = self.checker.compute_mae(self.y_true, self.y_pred)

        self.assertGreaterEqual(rmse, mae)

    def test_mse_equals_rmse_squared(self):
        """Test that MSE == RMSE^2."""
        rmse = self.checker.compute_rmse(self.y_true, self.y_pred)
        mse = self.checker.compute_mse(self.y_true, self.y_pred)

        self.assertAlmostEqual(mse, rmse ** 2, places=5)

    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        r_squared = self.checker.compute_r_squared(y_true, y_pred)
        rmse = self.checker.compute_rmse(y_true, y_pred)
        mae = self.checker.compute_mae(y_true, y_pred)

        self.assertAlmostEqual(r_squared, 1.0, places=5)
        self.assertAlmostEqual(rmse, 0.0, places=5)
        self.assertAlmostEqual(mae, 0.0, places=5)

    def test_get_residuals(self):
        """Test residuals computation."""
        residuals = self.checker.get_residuals(self.y_true, self.y_pred)

        self.assertEqual(len(residuals), self.n_samples)
        self.assertIsInstance(residuals, np.ndarray)
        # Residuals should equal y_true - y_pred
        np.testing.assert_array_almost_equal(residuals, self.y_true - self.y_pred)

    def test_get_residual_stats(self):
        """Test residual statistics contain all expected keys."""
        residual_stats = self.checker.get_residual_stats(self.y_true, self.y_pred)

        self.assertIn('mean', residual_stats)
        self.assertIn('std', residual_stats)
        self.assertIn('min', residual_stats)
        self.assertIn('max', residual_stats)
        self.assertIn('median', residual_stats)
        self.assertIn('q25', residual_stats)
        self.assertIn('q75', residual_stats)

    def test_get_residual_stats_values(self):
        """Test residual statistics are mathematically consistent."""
        residual_stats = self.checker.get_residual_stats(self.y_true, self.y_pred)

        self.assertLessEqual(residual_stats['min'], residual_stats['q25'])
        self.assertLessEqual(residual_stats['q25'], residual_stats['median'])
        self.assertLessEqual(residual_stats['median'], residual_stats['q75'])
        self.assertLessEqual(residual_stats['q75'], residual_stats['max'])

    def test_get_metrics_summary(self):
        """Test metrics summary string."""
        summary = self.checker.get_metrics_summary()

        self.assertIsInstance(summary, str)
        self.assertIn('r_squared', summary)
        self.assertIn('rmse', summary)

    def test_mismatched_lengths(self):
        """Test with mismatched input lengths raises ValueError."""
        with self.assertRaises(ValueError):
            self.checker.check_quality(
                y_true=np.array([1.0, 2.0, 3.0]),
                y_pred=np.array([1.0, 2.0])
            )

    def test_list_inputs(self):
        """Test that list inputs are accepted."""
        metrics = self.checker.check_quality(
            y_true=[1.0, 2.0, 3.0],
            y_pred=[1.1, 2.1, 3.1]
        )
        self.assertIn('r_squared', metrics)


if __name__ == '__main__':
    unittest.main()

# Made with Bob
