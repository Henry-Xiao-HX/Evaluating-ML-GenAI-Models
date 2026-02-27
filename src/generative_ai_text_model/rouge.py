"""
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metrics module.
Computes ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-S metrics.
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


class ROUGECalculator:
    """
    ROUGE metrics calculator for text summarization and generation evaluation.

    Supported ROUGE variants:
    - ROUGE-1: Unigram overlap
    - ROUGE-2: Bigram overlap
    - ROUGE-L: Longest common subsequence
    - ROUGE-S: Skip-bigram overlap
    """

    VALID_ROUGE_TYPES = {'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'rougeS'}

    def __init__(self):
        """Initialize the ROUGE calculator."""
        if not EVALUATE_AVAILABLE:
            raise ImportError(
                "This module requires the 'evaluate' library. "
                "Install it with: pip install evaluate"
            )
        self.rouge = evaluate.load('rouge')  # type: ignore[possibly-unbound]

    def compute_rouge(
        self,
        predictions: Union[List[str], str],
        references: Union[List[str], str],
        rouge_types: Optional[List[str]] = None,
        use_stemmer: bool = False,
        use_aggregator: bool = True
    ) -> Dict[str, Any]:
        """
        Compute ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-S).

        Args:
            predictions: Model predictions (list of strings or single string)
            references: Reference/ground truth texts (list of strings or single string)
            rouge_types: Types of ROUGE to compute.
                         Defaults to ['rouge1', 'rouge2', 'rougeL', 'rougeS'].
                         Valid values: 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'rougeS'
            use_stemmer: Whether to use stemmer for normalization (default: False)
            use_aggregator: Whether to aggregate scores across samples (default: True)

        Returns:
            Dictionary with ROUGE scores

        Raises:
            ValueError: If predictions and references have different lengths,
                        if inputs are None, or if invalid rouge_types are specified.

        Example:
            >>> calculator = ROUGECalculator()
            >>> scores = calculator.compute_rouge(
            ...     predictions="The cat sat on the mat",
            ...     references="A cat was sitting on the mat"
            ... )
            >>> # scores = {'rouge1': 0.727, 'rouge2': 0.4, 'rougeL': 0.727, 'rougeS': 0.5}
        """
        if predictions is None:
            raise ValueError("predictions cannot be None")
        if references is None:
            raise ValueError("references cannot be None")

        if rouge_types is None:
            rouge_types = ['rouge1', 'rouge2', 'rougeL', 'rougeS']

        # Validate rouge_types
        invalid_types = set(rouge_types) - self.VALID_ROUGE_TYPES
        if invalid_types:
            raise ValueError(
                f"Invalid rouge_types: {invalid_types}. "
                f"Valid options are: {self.VALID_ROUGE_TYPES}"
            )

        # Normalize inputs to lists
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [references]

        # Validate lengths match
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions and references must have the same length. "
                f"Got {len(predictions)} predictions and {len(references)} references."
            )

        # Compute ROUGE
        results = self.rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=rouge_types,
            use_stemmer=use_stemmer,
            use_aggregator=use_aggregator
        )

        return results or {}

    def compute_rouge_types_separate(
        self,
        predictions: Union[List[str], str],
        references: Union[List[str], str],
        use_stemmer: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute each ROUGE type separately with individual results.

        Args:
            predictions: Model predictions
            references: Reference texts
            use_stemmer: Whether to use stemmer (default: False)

        Returns:
            Dictionary with each ROUGE type and its metrics

        Example:
            >>> calculator = ROUGECalculator()
            >>> results = calculator.compute_rouge_types_separate(
            ...     predictions="The cat sat on the mat",
            ...     references="A cat was sitting on the mat"
            ... )
            >>> # results = {'rouge1': {...}, 'rouge2': {...}, 'rougeL': {...}, 'rougeS': {...}}
        """
        rouge_types = ['rouge1', 'rouge2', 'rougeL', 'rougeS']
        results = {}

        for rouge_type in rouge_types:
            results[rouge_type] = self.compute_rouge(
                predictions, references, [rouge_type], use_stemmer, use_aggregator=True
            )

        return results

    def get_rouge_score_explanation(self, rouge_type: str) -> str:
        """
        Get explanation of ROUGE metric types.

        Args:
            rouge_type: Type of ROUGE metric (e.g., 'rouge1', 'rouge2', 'rougeL', 'rougeS')

        Returns:
            Explanation string
        """
        explanations = {
            'rouge1': 'ROUGE-1: Overlap of unigrams (single words)',
            'rouge2': 'ROUGE-2: Overlap of bigrams (consecutive word pairs)',
            'rougeL': 'ROUGE-L: Longest common subsequence',
            'rougeLsum': 'ROUGE-Lsum: Longest common subsequence at summary level',
            'rougeS': 'ROUGE-S: Skip-bigram overlap (allows gaps between matched words)',
        }
        return explanations.get(rouge_type, f'Unknown ROUGE type: {rouge_type}')


class ROUGEAggregator:
    """Aggregate ROUGE scores across multiple samples."""

    @staticmethod
    def aggregate_rouge_scores(
        scores_list: List[Dict[str, Any]],
        aggregation_type: str = 'mean'
    ) -> Dict[str, float]:
        """
        Aggregate ROUGE scores across samples.

        Args:
            scores_list: List of ROUGE score dictionaries.
                         Example: [{'rouge1': 0.5, 'rouge2': 0.3}, {'rouge1': 0.7, 'rouge2': 0.5}]
            aggregation_type: Aggregation method ('mean', 'median', 'min', 'max')

        Returns:
            Aggregated scores dictionary.
            Example: {'rouge1': 0.6, 'rouge2': 0.4}

        Raises:
            ValueError: If aggregation_type is not one of 'mean', 'median', 'min', 'max'.
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
            values = [s[key] for s in scores_list if key in s and isinstance(s[key], (int, float))]

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
