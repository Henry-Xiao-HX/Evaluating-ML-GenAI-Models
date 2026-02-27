"""
BLEU (Bilingual Evaluation Understudy) metric module.
Computes BLEU score for machine translation and text generation evaluation.
Uses Hugging Face evaluate library for metric computation.
"""

from typing import Any, List, Dict, Union, Optional
import warnings
import numpy as np

try:
    import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False
    warnings.warn("Hugging Face evaluate library not installed. Install with: pip install evaluate")


class BLEUCalculator:
    """
    BLEU (Bilingual Evaluation Understudy) metrics calculator.

    BLEU measures n-gram precision between predictions and references,
    with a brevity penalty for shorter texts.

    Scores range from 0 to 1:
    - 0: No n-gram matches
    - 1: Perfect match with references
    """

    def __init__(self):
        """Initialize the BLEU calculator."""
        if not EVALUATE_AVAILABLE:
            raise ImportError(
                "This module requires the 'evaluate' library. "
                "Install it with: pip install evaluate"
            )
        self.bleu = evaluate.load('bleu')  # type: ignore[possibly-unbound]

    def compute_bleu(
        self,
        predictions: Union[List[str], str],
        references: Union[List[List[str]], List[str], Any],
        max_order: int = 4,
        smooth: bool = False
    ) -> Dict[str, Any]:
        """
        Compute BLEU score.

        Args:
            predictions: Model predictions (list of strings or single string)
            references: Reference texts.
                       For single reference per prediction, use List[str].
                       For multiple references, use List[List[str]].
            max_order: Maximum n-gram order to compute (default: 4).
                       Must be a positive integer.
                       Computes 1-grams, 2-grams, ..., n-grams up to max_order.
            smooth: Whether to apply smoothing for zero counts (default: False)

        Returns:
            Dictionary with 'bleu' score and 'precisions' list.
            Example: {'bleu': 0.45, 'precisions': [0.8, 0.6, 0.4, 0.2]}

        Raises:
            ValueError: If predictions and references have different lengths,
                        if inputs are None, or if max_order is not a positive integer.

        Example:
            >>> calculator = BLEUCalculator()
            >>> scores = calculator.compute_bleu(
            ...     predictions=["the cat sat on the mat"],
            ...     references=["a cat was sitting on the mat"]
            ... )
            >>> # scores = {'bleu': 0.45, 'precisions': [0.8, 0.6, 0.4, 0.2], ...}
        """
        if predictions is None:
            raise ValueError("predictions cannot be None")
        if references is None:
            raise ValueError("references cannot be None")
        if not isinstance(max_order, int) or max_order < 1:
            raise ValueError(f"max_order must be a positive integer, got: {max_order}")

        # Normalize inputs
        if isinstance(predictions, str):
            predictions = [predictions]

        # Normalize references structure
        if isinstance(references, list) and len(references) > 0:
            if isinstance(references[0], str):
                # Convert single references to list of lists
                references = [[ref] for ref in references]

        # Validate lengths match
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions and references must have the same length. "
                f"Got {len(predictions)} predictions and {len(references)} references."
            )

        results = self.bleu.compute(
            predictions=predictions,
            references=references,
            max_order=max_order,
            smooth=smooth
        )

        return results or {}

    def compute_bleu_detailed(
        self,
        predictions: Union[List[str], str],
        references: Union[List[List[str]], List[str]],
        max_order: int = 4,
        smooth: bool = False
    ) -> Dict[str, Any]:
        """
        Compute BLEU score with detailed n-gram breakdown.

        Args:
            predictions: Model predictions
            references: Reference texts
            max_order: Maximum n-gram order
            smooth: Whether to apply smoothing

        Returns:
            Dictionary with overall BLEU score, individual n-gram precisions,
            and a named n_gram_details breakdown.

        Example:
            >>> calculator = BLEUCalculator()
            >>> scores = calculator.compute_bleu_detailed(
            ...     predictions=["the cat sat on the mat"],
            ...     references=["a cat was sitting on the mat"]
            ... )
            >>> # scores = {
            >>> #   'bleu': 0.45,
            >>> #   'precisions': [0.8, 0.6, 0.4, 0.2],
            >>> #   'n_gram_details': {'1-gram': 0.8, '2-gram': 0.6, '3-gram': 0.4, '4-gram': 0.2}
            >>> # }
        """
        results = self.compute_bleu(predictions, references, max_order, smooth)

        # Add more detailed breakdown
        detailed_results = {
            'bleu': results.get('bleu', 0.0),
            'precisions': results.get('precisions', []),
            'n_gram_details': {}
        }

        precisions = results.get('precisions', [])
        for i, precision in enumerate(precisions, 1):
            detailed_results['n_gram_details'][f'{i}-gram'] = precision

        return detailed_results


class BLEUAggregator:
    """Aggregate BLEU scores across multiple samples."""

    @staticmethod
    def aggregate_bleu_scores(
        scores_list: List[Dict[str, Any]],
        aggregation_type: str = 'mean'
    ) -> Dict[str, float]:
        """
        Aggregate BLEU scores across samples.

        Args:
            scores_list: List of BLEU score dictionaries.
                         Example: [{'bleu': 0.5}, {'bleu': 0.7}]
            aggregation_type: Aggregation method ('mean', 'median', 'min', 'max')

        Returns:
            Aggregated scores dictionary.
            Example: {'bleu': 0.6}

        Raises:
            ValueError: If aggregation_type is not one of 'mean', 'median', 'min', 'max'.

        Example:
            >>> scores = [{'bleu': 0.5, 'brevity_penalty': 1.0},
            ...           {'bleu': 0.7, 'brevity_penalty': 0.9}]
            >>> result = BLEUAggregator.aggregate_bleu_scores(scores, 'mean')
            >>> # result = {'bleu': 0.6, 'brevity_penalty': 0.95}
        """
        valid_aggregations = {'mean', 'median', 'min', 'max'}
        if aggregation_type not in valid_aggregations:
            raise ValueError(
                f"Invalid aggregation_type: '{aggregation_type}'. "
                f"Valid options are: {valid_aggregations}"
            )

        if not scores_list:
            return {}

        aggregated = {}

        # Get all unique keys
        all_keys = set()
        for scores in scores_list:
            all_keys.update(scores.keys())

        for key in all_keys:
            # Handle both float and list values
            values = []
            for s in scores_list:
                if key in s:
                    val = s[key]
                    if isinstance(val, (int, float)):
                        values.append(float(val))

            if values:
                if aggregation_type == 'mean':
                    aggregated[key] = float(np.mean(values))
                elif aggregation_type == 'median':
                    aggregated[key] = float(np.median(values))
                elif aggregation_type == 'min':
                    aggregated[key] = float(np.min(values))
                elif aggregation_type == 'max':
                    aggregated[key] = float(np.max(values))

        return aggregated

# Made with Bob
