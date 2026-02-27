"""
Unit tests for generative AI text model metrics (ROUGE and BLEU).

Tests DataQualityChecker and QualityMetricsAggregator from data_quality_checker.py,
which orchestrate ROUGECalculator, BLEUCalculator, ROUGEAggregator, and BLEUAggregator.
"""

import unittest
import sys
from pathlib import Path

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_quality_checker import DataQualityChecker, QualityMetricsAggregator


class TestDataQualityChecker(unittest.TestCase):
    """Test cases for DataQualityChecker (ROUGE and BLEU orchestration)."""

    def setUp(self):
        """Initialize checker for each test."""
        self.checker = DataQualityChecker()

    def test_rouge_single_text(self):
        """Test ROUGE computation with single texts."""
        pred = "The cat sat on the mat"
        ref = "A cat was on the mat"

        result = self.checker.compute_rouge(pred, ref)

        self.assertIn('rouge1', result)
        self.assertIn('rouge2', result)
        self.assertIn('rougeL', result)

        # Verify scores are between 0 and 1
        for key in ['rouge1', 'rouge2', 'rougeL']:
            self.assertGreaterEqual(result[key], 0)
            self.assertLessEqual(result[key], 1)

    def test_rouge_multiple_texts(self):
        """Test ROUGE computation with multiple texts."""
        preds = ["The cat sat", "Dogs run fast"]
        refs = ["A cat sat", "The dog runs"]

        result = self.checker.compute_rouge(preds, refs)

        self.assertIn('rouge1', result)
        self.assertIsInstance(result['rouge1'], (int, float))

    def test_rouge_with_stemmer(self):
        """Test ROUGE computation with stemmer enabled."""
        pred = "The cats are running"
        ref = "A cat runs"

        result = self.checker.compute_rouge(pred, ref, use_stemmer=True)

        self.assertIn('rouge1', result)
        self.assertGreaterEqual(result['rouge1'], 0)

    def test_rouge_without_aggregator(self):
        """Test ROUGE computation without aggregation (per-sample scores)."""
        preds = ["The cat sat", "Dogs run fast"]
        refs = ["A cat sat", "The dog runs"]

        result = self.checker.compute_rouge(preds, refs, use_aggregator=False)

        self.assertIn('rouge1', result)

    def test_bleu_single_text(self):
        """Test BLEU computation with single texts."""
        pred = ["the cat sat on the mat"]
        ref = ["a cat was on the mat"]

        result = self.checker.compute_bleu(pred, ref)

        self.assertIn('bleu', result)
        self.assertGreaterEqual(result['bleu'], 0)
        self.assertLessEqual(result['bleu'], 1)

    def test_bleu_multiple_references(self):
        """Test BLEU with multiple references per prediction."""
        pred = ["the cat sat"]
        refs = [["a cat sat", "the cat was sitting"]]

        result = self.checker.compute_bleu(pred, refs)

        self.assertIn('bleu', result)
        self.assertIsInstance(result['bleu'], (int, float))

    def test_bleu_with_smoothing(self):
        """Test BLEU computation with smoothing enabled."""
        pred = ["the cat sat on the mat"]
        ref = ["a cat was on the mat"]

        result = self.checker.compute_bleu(pred, ref, smooth=True)

        self.assertIn('bleu', result)
        self.assertGreaterEqual(result['bleu'], 0)

    def test_bleu_custom_max_order(self):
        """Test BLEU computation with custom max_order."""
        pred = ["the cat sat on the mat"]
        ref = ["a cat was on the mat"]

        result = self.checker.compute_bleu(pred, ref, max_order=2)

        self.assertIn('bleu', result)

    def test_all_metrics(self):
        """Test computing all metrics at once."""
        pred = ["Machine learning is powerful"]
        ref = ["ML is very powerful"]

        result = self.checker.compute_all_metrics(pred, ref)

        self.assertIn('rouge', result)
        self.assertIn('bleu', result)

    def test_all_metrics_rouge_only(self):
        """Test computing only ROUGE metrics."""
        pred = ["Machine learning is powerful"]
        ref = ["ML is very powerful"]

        result = self.checker.compute_all_metrics(pred, ref, compute_bleu=False)

        self.assertIn('rouge', result)
        self.assertNotIn('bleu', result)

    def test_all_metrics_bleu_only(self):
        """Test computing only BLEU metrics."""
        pred = ["Machine learning is powerful"]
        ref = ["ML is very powerful"]

        result = self.checker.compute_all_metrics(pred, ref, compute_rouge=False)

        self.assertNotIn('rouge', result)
        self.assertIn('bleu', result)

    def test_batch_compute(self):
        """Test batch computation."""
        preds = [
            ["The cat sat", "Dogs run"],
            ["Birds fly", "Fish swim"]
        ]
        refs = [
            ["A cat sat", "The dog runs"],
            ["Birds are flying", "Fish are swimming"]
        ]

        results = self.checker.batch_compute_metrics(preds, refs)

        self.assertEqual(len(results), 2)
        self.assertIn('rouge', results[0])
        self.assertIn('bleu', results[1])

    def test_batch_compute_rouge_only(self):
        """Test batch computation with only ROUGE metrics."""
        preds = [["The cat sat"]]
        refs = [["A cat sat"]]

        results = self.checker.batch_compute_metrics(preds, refs, metric_types=['rouge'])

        self.assertEqual(len(results), 1)
        self.assertIn('rouge', results[0])
        self.assertNotIn('bleu', results[0])

    def test_rouge_types_filter(self):
        """Test selective ROUGE types."""
        pred = "The quick brown fox"
        ref = "A quick brown fox"

        result = self.checker.compute_rouge(
            pred, ref,
            rouge_types=['rouge1', 'rouge2']
        )

        self.assertIn('rouge1', result)
        self.assertIn('rouge2', result)
        self.assertNotIn('rougeL', result)

    def test_format_results(self):
        """Test result formatting."""
        results = {'metric1': 0.5, 'metric2': 0.75}

        formatted = self.checker.format_results({'test': results})

        self.assertIn('0.5', formatted)
        self.assertIn('TEST', formatted)

    def test_format_results_non_numeric_value(self):
        """Test result formatting with non-numeric values."""
        results = {'metric1': 0.5, 'label': 'some_string'}

        formatted = self.checker.format_results({'test': results})

        self.assertIn('some_string', formatted)

    def test_get_rouge_score_explanation_valid(self):
        """Test ROUGE score explanation for valid types."""
        for rouge_type in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
            explanation = self.checker.get_rouge_score_explanation(rouge_type)
            self.assertIsInstance(explanation, str)
            self.assertGreater(len(explanation), 0)

    def test_get_rouge_score_explanation_unknown(self):
        """Test ROUGE score explanation for unknown type."""
        explanation = self.checker.get_rouge_score_explanation('rouge99')
        self.assertIn('Unknown', explanation)

    def test_rouge_invalid_type_raises(self):
        """Test that invalid rouge_types raises ValueError."""
        with self.assertRaises(ValueError):
            self.checker.compute_rouge("text", "text", rouge_types=['invalid_type'])

    def test_rouge_none_predictions_raises(self):
        """Test that None predictions raises ValueError."""
        with self.assertRaises(ValueError):
            self.checker.compute_rouge(None, "reference")

    def test_rouge_none_references_raises(self):
        """Test that None references raises ValueError."""
        with self.assertRaises(ValueError):
            self.checker.compute_rouge("prediction", None)

    def test_rouge_mismatched_lengths_raises(self):
        """Test that mismatched prediction/reference lengths raises ValueError."""
        with self.assertRaises(ValueError):
            self.checker.compute_rouge(["pred1", "pred2"], ["ref1"])

    def test_bleu_none_predictions_raises(self):
        """Test that None predictions raises ValueError."""
        with self.assertRaises(ValueError):
            self.checker.compute_bleu(None, ["reference"])

    def test_bleu_none_references_raises(self):
        """Test that None references raises ValueError."""
        with self.assertRaises(ValueError):
            self.checker.compute_bleu(["prediction"], None)

    def test_bleu_invalid_max_order_raises(self):
        """Test that invalid max_order raises ValueError."""
        with self.assertRaises(ValueError):
            self.checker.compute_bleu(["prediction"], ["reference"], max_order=0)

    def test_bleu_mismatched_lengths_raises(self):
        """Test that mismatched prediction/reference lengths raises ValueError."""
        with self.assertRaises(ValueError):
            self.checker.compute_bleu(["pred1", "pred2"], ["ref1"])


class TestQualityMetricsAggregator(unittest.TestCase):
    """Test cases for QualityMetricsAggregator."""

    def setUp(self):
        """Initialize aggregator for each test."""
        self.aggregator = QualityMetricsAggregator()

    def test_aggregate_rouge_mean(self):
        """Test ROUGE aggregation with mean."""
        scores = [
            {'rouge1': 0.5, 'rouge2': 0.3},
            {'rouge1': 0.7, 'rouge2': 0.5}
        ]

        result = self.aggregator.aggregate_rouge_scores(scores, 'mean')

        self.assertAlmostEqual(result['rouge1'], 0.6)
        self.assertAlmostEqual(result['rouge2'], 0.4)

    def test_aggregate_rouge_median(self):
        """Test ROUGE aggregation with median."""
        scores = [
            {'rouge1': 0.2},
            {'rouge1': 0.5},
            {'rouge1': 0.8}
        ]

        result = self.aggregator.aggregate_rouge_scores(scores, 'median')

        self.assertAlmostEqual(result['rouge1'], 0.5)

    def test_aggregate_rouge_min_max(self):
        """Test ROUGE aggregation with min and max."""
        scores = [
            {'rouge1': 0.3},
            {'rouge1': 0.7}
        ]

        result_min = self.aggregator.aggregate_rouge_scores(scores, 'min')
        result_max = self.aggregator.aggregate_rouge_scores(scores, 'max')

        self.assertAlmostEqual(result_min['rouge1'], 0.3)
        self.assertAlmostEqual(result_max['rouge1'], 0.7)

    def test_aggregate_rouge_invalid_type_raises(self):
        """Test that invalid aggregation type raises ValueError."""
        scores = [{'rouge1': 0.5}]
        with self.assertRaises(ValueError):
            self.aggregator.aggregate_rouge_scores(scores, 'invalid')

    def test_aggregate_bleu_mean(self):
        """Test BLEU aggregation with mean."""
        scores = [
            {'bleu': 0.5},
            {'bleu': 0.7}
        ]

        result = self.aggregator.aggregate_bleu_scores(scores, 'mean')

        self.assertAlmostEqual(result['bleu'], 0.6)

    def test_aggregate_bleu_median(self):
        """Test BLEU aggregation with median."""
        scores = [{'bleu': 0.2}, {'bleu': 0.5}, {'bleu': 0.8}]

        result = self.aggregator.aggregate_bleu_scores(scores, 'median')

        self.assertAlmostEqual(result['bleu'], 0.5)

    def test_aggregate_bleu_min_max(self):
        """Test BLEU aggregation with min and max."""
        scores = [{'bleu': 0.3}, {'bleu': 0.7}]

        result_min = self.aggregator.aggregate_bleu_scores(scores, 'min')
        result_max = self.aggregator.aggregate_bleu_scores(scores, 'max')

        self.assertAlmostEqual(result_min['bleu'], 0.3)
        self.assertAlmostEqual(result_max['bleu'], 0.7)

    def test_aggregate_bleu_invalid_type_raises(self):
        """Test that invalid aggregation type raises ValueError."""
        scores = [{'bleu': 0.5}]
        with self.assertRaises(ValueError):
            self.aggregator.aggregate_bleu_scores(scores, 'invalid')

    def test_aggregate_rouge_empty_list(self):
        """Test ROUGE aggregation with empty list returns empty dict."""
        result = self.aggregator.aggregate_rouge_scores([], 'mean')
        self.assertEqual(result, {})

    def test_aggregate_bleu_empty_list(self):
        """Test BLEU aggregation with empty list returns empty dict."""
        result = self.aggregator.aggregate_bleu_scores([], 'mean')
        self.assertEqual(result, {})


class TestRougeEdgeCases(unittest.TestCase):
    """Test ROUGE edge cases and error handling."""

    def setUp(self):
        """Initialize checker for each test."""
        self.checker = DataQualityChecker()

    def test_empty_strings(self):
        """Test ROUGE with empty strings."""
        result = self.checker.compute_rouge("", "")
        self.assertIn('rouge1', result)

    def test_identical_texts(self):
        """Test ROUGE with identical prediction and reference."""
        text = "This is identical text"
        result = self.checker.compute_rouge(text, text)

        # All ROUGE scores should be 1.0
        for key in ['rouge1', 'rouge2', 'rougeL']:
            self.assertAlmostEqual(result[key], 1.0, places=2)

    def test_completely_different_texts(self):
        """Test ROUGE with completely different texts."""
        pred = "apple banana cherry"
        ref = "xyz qwerty asdf"

        result = self.checker.compute_rouge(pred, ref)

        # Scores should be 0 or very low
        for key in ['rouge1', 'rouge2', 'rougeL']:
            self.assertLess(result[key], 0.1)

    def test_rouge_types_separate(self):
        """Test computing each ROUGE type separately."""
        pred = "The cat sat on the mat"
        ref = "A cat was on the mat"

        results = self.checker.rouge_calculator.compute_rouge_types_separate(pred, ref)

        self.assertIn('rouge1', results)
        self.assertIn('rouge2', results)
        self.assertIn('rougeL', results)

    def test_bleu_detailed(self):
        """Test detailed BLEU computation with n-gram breakdown."""
        pred = ["the cat sat on the mat"]
        ref = ["a cat was on the mat"]

        results = self.checker.bleu_calculator.compute_bleu_detailed(pred, ref)

        self.assertIn('bleu', results)
        self.assertIn('precisions', results)
        self.assertIn('n_gram_details', results)
        self.assertIn('1-gram', results['n_gram_details'])


if __name__ == '__main__':
    unittest.main()

# Made with Bob
