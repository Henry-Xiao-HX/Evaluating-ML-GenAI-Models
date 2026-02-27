"""
Unified Data Quality Checker - Master Orchestrator.

Brings together quality checks for:
- Binary Classification Models
- Regression Models  
- Generative AI Text Models

Provides a unified interface for comprehensive model evaluation across all model types.
"""

from typing import Dict, Union, Optional, Any
import numpy as np

try:
    from .binary_classification import BinaryClassifierChecker
    from .regression import RegressionChecker
    from .generative_ai_text_model import BLEUCalculator, ROUGECalculator, BLEUAggregator, ROUGEAggregator
except ImportError:
    # Fallback for direct imports
    from binary_classification import BinaryClassifierChecker
    from regression import RegressionChecker
    from generative_ai_text_model import BLEUCalculator, ROUGECalculator, BLEUAggregator, ROUGEAggregator


class UnifiedDataQualityChecker:
    """
    Master orchestrator for comprehensive data quality checking.
    
    Coordinates quality assessment across three model categories:
    1. Binary Classification: AUROC, precision, recall, F1, log loss, etc.
    2. Regression: R², RMSE, MAE, MSE, explained variance
    3. Generative AI Text: BLEU, ROUGE-1/2/L/S scores
    """
    
    def __init__(self):
        """Initialize all quality checkers."""
        self.binary_classifier_checker = BinaryClassifierChecker()
        self.regression_checker = RegressionChecker()
        self.bleu_calculator = BLEUCalculator()
        self.rouge_calculator = ROUGECalculator()
        self.bleu_aggregator = BLEUAggregator()
        self.rouge_aggregator = ROUGEAggregator()
    
    # Binary Classification Methods
    def check_binary_classification_quality(
        self,
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
        y_pred_proba: Optional[Union[np.ndarray, list]] = None,
        sample_weight: Optional[np.ndarray] = None,
        cache_key: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Check quality of binary classification predictions.
        
        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            y_pred_proba: Predicted probabilities (optional).
            sample_weight: Optional sample weights.
            cache_key: Optional cache key.
        
        Returns:
            Dictionary of binary classification metrics.
        """
        return self.binary_classifier_checker.check_quality(
            y_true, y_pred, y_pred_proba, sample_weight, cache_key
        )
    
    # Regression Methods
    def check_regression_quality(
        self,
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
        sample_weight: Optional[np.ndarray] = None,
        cache_key: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Check quality of regression predictions.
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            sample_weight: Optional sample weights.
            cache_key: Optional cache key.
        
        Returns:
            Dictionary of regression metrics.
        """
        return self.regression_checker.check_quality(
            y_true, y_pred, sample_weight, cache_key
        )
    
    # Generative AI Text Methods
    def compute_bleu(
        self,
        predictions: Union[list, str],
        references: Union[list, str],
        max_order: int = 4,
        smooth: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute BLEU score for text generation.
        
        Args:
            predictions: Model predictions.
            references: Reference texts.
            max_order: Maximum n-gram order.
            smooth: Whether to apply smoothing.
        
        Returns:
            BLEU scores.
        """
        return self.bleu_calculator.compute_bleu(
            predictions, references, max_order, smooth
        )
    
    def compute_rouge(
        self,
        predictions: Union[list, str],
        references: Union[list, str],
        rouge_types: Optional[list] = None,
        use_stemmer: bool = False,
        use_aggregator: bool = True,
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores for text generation.
        
        Args:
            predictions: Model predictions.
            references: Reference texts.
            rouge_types: Types of ROUGE to compute.
            use_stemmer: Whether to use stemmer.
            use_aggregator: Whether to aggregate scores.
        
        Returns:
            ROUGE scores.
        """
        return self.rouge_calculator.compute_rouge(
            predictions, references, rouge_types, use_stemmer, use_aggregator
        )
    
    def aggregate_bleu_scores(
        self,
        scores_list: list,
        aggregation_type: str = 'mean',
    ) -> Dict[str, float]:
        """
        Aggregate BLEU scores across samples.
        
        Args:
            scores_list: List of BLEU score dictionaries.
            aggregation_type: Aggregation method.
        
        Returns:
            Aggregated BLEU scores.
        """
        return self.bleu_aggregator.aggregate_bleu_scores(scores_list, aggregation_type)
    
    def aggregate_rouge_scores(
        self,
        scores_list: list,
        aggregation_type: str = 'mean',
    ) -> Dict[str, float]:
        """
        Aggregate ROUGE scores across samples.
        
        Args:
            scores_list: List of ROUGE score dictionaries.
            aggregation_type: Aggregation method.
        
        Returns:
            Aggregated ROUGE scores.
        """
        return self.rouge_aggregator.aggregate_rouge_scores(scores_list, aggregation_type)
    
    def check_quality(
        self,
        predictions: Any,
        references: Any,
        task_type: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Unified entry point for quality checking across all task types.

        Args:
            predictions: Model predictions. Type depends on task_type:
                         - 'classification': np.ndarray of binary labels
                         - 'regression': np.ndarray of numeric values
                         - 'text_generation': List[str] of generated texts
            references: Ground truth / reference values. Same type constraints as predictions.
            task_type: One of 'classification', 'regression', 'text_generation'.
            **kwargs: Additional keyword arguments forwarded to the underlying checker.
                      e.g. y_pred_proba for classification, rouge_types for text_generation.

        Returns:
            Dictionary of quality metrics for the given task type.

        Raises:
            ValueError: If task_type is not one of the supported values.

        Example:
            >>> checker = UnifiedDataQualityChecker()
            >>> report = checker.check_quality(
            ...     predictions=["the cat sat"],
            ...     references=["a cat sat"],
            ...     task_type='text_generation'
            ... )
        """
        supported_tasks = {'classification', 'regression', 'text_generation'}
        if task_type not in supported_tasks:
            raise ValueError(
                f"Unsupported task_type: '{task_type}'. "
                f"Valid options are: {supported_tasks}"
            )

        if task_type == 'classification':
            y_pred_proba = kwargs.pop('y_pred_proba', None)
            sample_weight = kwargs.pop('sample_weight', None)
            cache_key = kwargs.pop('cache_key', None)
            return self.check_binary_classification_quality(
                y_true=references,
                y_pred=predictions,
                y_pred_proba=y_pred_proba,
                sample_weight=sample_weight,
                cache_key=cache_key,
            )

        if task_type == 'regression':
            sample_weight = kwargs.pop('sample_weight', None)
            cache_key = kwargs.pop('cache_key', None)
            return self.check_regression_quality(
                y_true=references,
                y_pred=predictions,
                sample_weight=sample_weight,
                cache_key=cache_key,
            )

        # task_type == 'text_generation'
        rouge_types = kwargs.pop('rouge_types', None)
        use_stemmer = kwargs.pop('use_stemmer', False)
        max_order = kwargs.pop('max_order', 4)
        smooth = kwargs.pop('smooth', False)

        rouge_scores = self.compute_rouge(
            predictions=predictions,
            references=references,
            rouge_types=rouge_types,
            use_stemmer=use_stemmer,
        )
        bleu_scores = self.compute_bleu(
            predictions=predictions,
            references=references,
            max_order=max_order,
            smooth=smooth,
        )
        return {'rouge': rouge_scores, 'bleu': bleu_scores}

    def get_available_modules(self) -> Dict[str, str]:
        """
        Get summary of available quality check modules.

        Returns:
            Dictionary describing available modules.
        """
        return {
            'binary_classification': 'AUROC, precision, recall, F1, log loss, FPR, TPR, AUPR',
            'regression': 'R², RMSE, MAE, MSE, explained variance',
            'generative_ai_text': 'BLEU, ROUGE-1/2/L/S',
        }

    def clear_all_caches(self) -> None:
        """Clear all metric caches."""
        self.binary_classifier_checker.clear_cache()
        self.regression_checker.clear_cache()


# Alias for backwards compatibility with README documentation
UnifiedQualityChecker = UnifiedDataQualityChecker
