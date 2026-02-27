"""
Unit tests for binary classification quality checker.

Tests BinaryClassifierChecker from binary_classification/binary_classifier_checker.py,
which orchestrates BinaryClassificationMetrics.
"""

import unittest
import sys
from pathlib import Path

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from binary_classification.binary_classifier_checker import BinaryClassifierChecker


class TestBinaryClassificationChecker(unittest.TestCase):
    """Test cases for BinaryClassifierChecker."""

    def setUp(self):
        """Initialize binary classification checker for each test."""
        self.checker = BinaryClassifierChecker()
        np.random.seed(42)
        self.n_samples = 100
        self.y_true = np.random.randint(0, 2, self.n_samples)
        self.y_pred = np.random.randint(0, 2, self.n_samples)
        self.y_pred_proba = np.random.rand(self.n_samples)

    def test_check_quality_basic(self):
        """Test basic quality check with binary predictions."""
        metrics = self.checker.check_quality(
            y_true=self.y_true,
            y_pred=self.y_pred
        )

        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_measure', metrics)
        self.assertIn('true_positive_rate', metrics)
        self.assertIn('false_positive_rate', metrics)

    def test_check_quality_with_probabilities(self):
        """Test quality check with probability predictions."""
        metrics = self.checker.check_quality(
            y_true=self.y_true,
            y_pred=self.y_pred,
            y_pred_proba=self.y_pred_proba
        )

        self.assertIn('roc_auc', metrics)
        self.assertIn('log_loss', metrics)
        self.assertIn('auc_pr', metrics)

    def test_check_quality_with_cache(self):
        """Test that caching works correctly."""
        metrics_first = self.checker.check_quality(
            y_true=self.y_true,
            y_pred=self.y_pred,
            cache_key='test_cache'
        )
        metrics_cached = self.checker.check_quality(
            y_true=self.y_true,
            y_pred=self.y_pred,
            cache_key='test_cache'
        )

        self.assertEqual(metrics_first, metrics_cached)
        self.assertIn('test_cache', self.checker.metrics_cache)

    def test_clear_cache(self):
        """Test that cache is cleared correctly."""
        self.checker.check_quality(
            y_true=self.y_true,
            y_pred=self.y_pred,
            cache_key='test_cache'
        )
        self.checker.clear_cache()

        self.assertEqual(len(self.checker.metrics_cache), 0)

    def test_compute_precision(self):
        """Test precision computation."""
        precision = self.checker.compute_precision(self.y_true, self.y_pred)

        self.assertIsInstance(precision, float)
        self.assertGreaterEqual(precision, 0.0)
        self.assertLessEqual(precision, 1.0)

    def test_compute_recall(self):
        """Test recall computation."""
        recall = self.checker.compute_recall(self.y_true, self.y_pred)

        self.assertIsInstance(recall, float)
        self.assertGreaterEqual(recall, 0.0)
        self.assertLessEqual(recall, 1.0)

    def test_compute_f1_measure(self):
        """Test F1-measure computation."""
        f1 = self.checker.compute_f1_measure(self.y_true, self.y_pred)

        self.assertIsInstance(f1, float)
        self.assertGreaterEqual(f1, 0.0)
        self.assertLessEqual(f1, 1.0)

    def test_compute_roc_auc(self):
        """Test ROC AUC computation."""
        roc_auc = self.checker.compute_roc_auc(self.y_true, self.y_pred_proba)

        self.assertIsInstance(roc_auc, float)
        self.assertGreaterEqual(roc_auc, 0.0)
        self.assertLessEqual(roc_auc, 1.0)

    def test_compute_log_loss(self):
        """Test log loss computation."""
        log_loss_val = self.checker.compute_log_loss(self.y_true, self.y_pred_proba)

        self.assertIsInstance(log_loss_val, float)
        self.assertGreater(log_loss_val, 0.0)

    def test_get_roc_curve_data(self):
        """Test ROC curve data retrieval."""
        fpr, tpr, thresholds = self.checker.get_roc_curve_data(
            self.y_true, self.y_pred_proba
        )

        self.assertIsInstance(fpr, np.ndarray)
        self.assertIsInstance(tpr, np.ndarray)
        self.assertIsInstance(thresholds, np.ndarray)
        self.assertEqual(len(fpr), len(tpr))

    def test_get_pr_curve_data(self):
        """Test Precision-Recall curve data retrieval."""
        precision, recall, thresholds = self.checker.get_pr_curve_data(
            self.y_true, self.y_pred_proba
        )

        self.assertIsInstance(precision, np.ndarray)
        self.assertIsInstance(recall, np.ndarray)
        self.assertIsInstance(thresholds, np.ndarray)

    def test_get_metrics_summary(self):
        """Test metrics summary string."""
        summary = self.checker.get_metrics_summary()

        self.assertIsInstance(summary, str)
        self.assertIn('roc_auc', summary)
        self.assertIn('precision', summary)

    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])

        precision = self.checker.compute_precision(y_true, y_pred)
        recall = self.checker.compute_recall(y_true, y_pred)
        f1 = self.checker.compute_f1_measure(y_true, y_pred)

        self.assertAlmostEqual(precision, 1.0)
        self.assertAlmostEqual(recall, 1.0)
        self.assertAlmostEqual(f1, 1.0)

    def test_mismatched_lengths(self):
        """Test with mismatched input lengths raises ValueError."""
        with self.assertRaises(ValueError):
            self.checker.check_quality(
                y_true=np.array([0, 1, 0]),
                y_pred=np.array([1, 0])
            )

    def test_mismatched_proba_lengths(self):
        """Test with mismatched y_pred_proba length raises ValueError."""
        with self.assertRaises(ValueError):
            self.checker.check_quality(
                y_true=np.array([0, 1, 0]),
                y_pred=np.array([0, 1, 0]),
                y_pred_proba=np.array([0.1, 0.9])
            )

    def test_list_inputs(self):
        """Test that list inputs are accepted."""
        metrics = self.checker.check_quality(
            y_true=[0, 1, 0, 1],
            y_pred=[0, 1, 0, 1]
        )
        self.assertIn('precision', metrics)


if __name__ == '__main__':
    unittest.main()

# Made with Bob
