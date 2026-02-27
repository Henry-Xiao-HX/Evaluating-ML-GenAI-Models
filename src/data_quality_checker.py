"""
Data Quality Checker driver module.
Orchestrates ROUGE and BLEU metrics computation.
Uses Hugging Face evaluate library for metric computation.
"""

from typing import Any, List, Dict, Union, Optional
import numpy as np

try:
    from .generative_ai_text_model.rouge import ROUGECalculator, ROUGEAggregator
    from .generative_ai_text_model.bleu import BLEUCalculator, BLEUAggregator
except ImportError:
    # Fallback for direct imports
    from generative_ai_text_model.rouge import ROUGECalculator, ROUGEAggregator
    from generative_ai_text_model.bleu import BLEUCalculator, BLEUAggregator


class DataQualityChecker:
    """
    Comprehensive data quality checker supporting ROUGE and BLEU metrics.
    
    This is a driver class that orchestrates metric computation by delegating to:
    - ROUGECalculator: Handles ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-S metrics
    - BLEUCalculator: Handles BLEU (Bilingual Evaluation Understudy) metric
    
    Metrics:
    - ROUGE-1: Unigram overlap
    - ROUGE-2: Bigram overlap
    - ROUGE-L: Longest common subsequence
    - ROUGE-S: Skip-bigram overlap
    - BLEU: Bilingual Evaluation Understudy
    """
    
    def __init__(self):
        """Initialize the data quality checker with metric calculators."""
        self.rouge_calculator = ROUGECalculator()
        self.bleu_calculator = BLEUCalculator()
        self.rouge_aggregator = ROUGEAggregator()
        self.bleu_aggregator = BLEUAggregator()
    
    def compute_rouge(
        self,
        predictions: Union[List[str], str],
        references: Union[List[str], str],
        rouge_types: Optional[List[str]] = None,
        use_stemmer: bool = False,
        use_aggregator: bool = True
    ) -> Dict[str, float]:
        """
        Compute ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-S).
        
        Delegates to ROUGECalculator.compute_rouge().
        
        Args:
            predictions: Model predictions (list of strings or single string)
            references: Reference/ground truth texts (list of strings or single string)
            rouge_types: Types of ROUGE to compute. Defaults to all.
            use_stemmer: Whether to use stemmer (default: False)
            use_aggregator: Whether to aggregate scores (default: True)
        
        Returns:
            Dictionary with ROUGE scores
        
        Example:
            >>> checker = DataQualityChecker()
            >>> scores = checker.compute_rouge(
            ...     predictions="The cat sat on the mat",
            ...     references="A cat was sitting on the mat"
            ... )
        """
        return self.rouge_calculator.compute_rouge(
            predictions, references, rouge_types, use_stemmer, use_aggregator
        )
    
    def compute_bleu(
        self,
        predictions: Union[List[str], str],
        references: Union[List[List[str]], List[str], Any],
        max_order: int = 4,
        smooth: bool = False
    ) -> Dict[str, Any]:
        """
        Compute BLEU score.
        
        Delegates to BLEUCalculator.compute_bleu().
        
        Args:
            predictions: Model predictions (list of strings or single string)
            references: Reference texts (list of lists or list of strings)
                       For single reference per prediction, use List[str]
                       For multiple references, use List[List[str]]
            max_order: Maximum n-gram order (default: 4)
            smooth: Whether to apply smoothing (default: False)
        
        Returns:
            Dictionary with BLEU score and precisions
        
        Example:
            >>> checker = DataQualityChecker()
            >>> scores = checker.compute_bleu(
            ...     predictions=["the cat sat on the mat"],
            ...     references=["a cat was sitting on the mat"]
            ... )
        """
        return self.bleu_calculator.compute_bleu(predictions, references, max_order, smooth)
    
    def compute_all_metrics(
        self,
        predictions: Union[List[str], str],
        references: Union[List[str], str],
        compute_rouge: bool = True,
        compute_bleu: bool = True,
        rouge_types: Optional[List[str]] = None,
        stemmer: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute all supported metrics at once.
        
        Args:
            predictions: Model predictions
            references: Reference texts
            compute_rouge: Whether to compute ROUGE metrics (default: True)
            compute_bleu: Whether to compute BLEU score (default: True)
            rouge_types: ROUGE types to compute
            stemmer: Use stemmer for ROUGE (default: False)
        
        Returns:
            Dictionary with all computed metrics
        
        Example:
            >>> checker = DataQualityChecker()
            >>> all_scores = checker.compute_all_metrics(
            ...     predictions=["the cat is sitting"],
            ...     references=["a cat is sitting"]
            ... )
        """
        results = {}
        
        if compute_rouge:
            results['rouge'] = self.compute_rouge(
                predictions, references, rouge_types, stemmer
            )
        
        if compute_bleu:
            results['bleu'] = self.compute_bleu(predictions, references)
        
        return results
    
    def batch_compute_metrics(
        self,
        predictions_list: List[List[str]],
        references_list: List[List[str]],
        metric_types: Optional[List[str]] = None
    ) -> List[Dict[str, float]]:
        """
        Compute metrics for multiple batches.
        
        Args:
            predictions_list: List of prediction batches
            references_list: List of reference batches
            metric_types: Types of metrics to compute
        
        Returns:
            List of dictionaries with metrics for each batch
        """
        if metric_types is None:
            metric_types = ['rouge', 'bleu']
        
        results = []
        for preds, refs in zip(predictions_list, references_list):
            batch_result = {}
            
            if 'rouge' in metric_types:
                batch_result['rouge'] = self.compute_rouge(preds, refs)
            if 'bleu' in metric_types:
                batch_result['bleu'] = self.compute_bleu(preds, refs)
            
            results.append(batch_result)
        
        return results
    
    def get_rouge_score_explanation(self, rouge_type: str) -> str:
        """
        Get explanation of ROUGE metric types.
        
        Delegates to ROUGECalculator.
        """
        return self.rouge_calculator.get_rouge_score_explanation(rouge_type)
    
    def format_results(
        self,
        results: Dict,
        precision: int = 4
    ) -> str:
        """
        Format results for display.
        
        Args:
            results: Dictionary of metric results
            precision: Decimal precision for display
        
        Returns:
            Formatted string representation
        """
        lines = []
        
        for metric_group, scores in results.items():
            lines.append(f"\n{metric_group.upper()}:")
            lines.append("-" * 40)
            
            if isinstance(scores, dict):
                for key, value in scores.items():
                    if isinstance(value, (int, float)):
                        lines.append(f"  {key}: {value:.{precision}f}")
                    else:
                        lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)


class QualityMetricsAggregator:
    """
    Aggregate quality metrics across multiple samples.
    
    This class combines ROUGE and BLEU aggregation capabilities.
    """
    
    def __init__(self):
        """Initialize the aggregator with individual aggregators."""
        self.rouge_aggregator = ROUGEAggregator()
        self.bleu_aggregator = BLEUAggregator()
    
    def aggregate_rouge_scores(
        self,
        scores_list: List[Dict[str, float]],
        aggregation_type: str = 'mean'
    ) -> Dict[str, float]:
        """
        Aggregate ROUGE scores across samples.
        
        Delegates to ROUGEAggregator.
        
        Args:
            scores_list: List of ROUGE score dictionaries
            aggregation_type: 'mean', 'median', 'min', 'max'
        
        Returns:
            Aggregated scores
        """
        return self.rouge_aggregator.aggregate_rouge_scores(scores_list, aggregation_type)
    
    def aggregate_bleu_scores(
        self,
        scores_list: List[Dict[str, float]],
        aggregation_type: str = 'mean'
    ) -> Dict[str, float]:
        """
        Aggregate BLEU scores across samples.
        
        Delegates to BLEUAggregator.
        
        Args:
            scores_list: List of BLEU score dictionaries
            aggregation_type: 'mean', 'median', 'min', 'max'
        
        Returns:
            Aggregated scores
        """
        return self.bleu_aggregator.aggregate_bleu_scores(scores_list, aggregation_type)
