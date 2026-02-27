# Evaluating ML/GenAI Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Evaluating binary classification, regression, and generative AI models. Supports ROUGE/BLEU metrics for text generative AI, binary classification metrics (AUROC/precision/recall/F1/log loss/AUPR), and regression model metrics (R²/RMSE/MAE/MSE/explained variance). For a more detailed explanation on the different evaluation metrics, read https://henry-xiao-hx.github.io/henry_xiao_blogs/2026/02/17/Evaluating-ML-GenAI-Models.html

## Project Structure

```
src/
├── data_quality_checker.py           # Legacy text metrics driver (ROUGE, BLEU)
├── unified_quality_checker.py        # Multi-task orchestrator
├── utils.py                          # Utility functions
├── binary_classification/
│   ├── binary_classifier_checker.py  # Binary classification orchestrator
│   └── metrics.py                    # Binary classification metric functions
├── regression/
│   ├── regression_checker.py         # Regression orchestrator
│   └── metrics.py                    # Regression metric functions
└── generative_ai_text_model/
    ├── bleu.py                       # BLEU calculator & aggregator
    └── rouge.py                      # ROUGE calculator & aggregator

tests/
├── conftest.py                       # Pytest fixtures
├── test_binary_classification.py
├── test_generative_ai.py
├── test_regression.py
└── test_utils.py

examples/
├── binary_classification/
│   ├── example_1_basic_metrics.py
│   └── example_2_advanced_metrics.py
├── generative_ai/
│   ├── example_1_basic_bleu_rouge.py
│   └── example_2_llm_evaluation.py
└── regression/
    ├── example_1_basic_metrics.py
    └── example_2_advanced_metrics.py
```

## Installation

```bash
git clone https://github.com/Henry-Xiao-HX/Evaluating-ML-GenAI-Models.git
cd Evaluating-ML-GenAI-Models
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, evaluate, numpy, scikit-learn, rouge-score, nltk

## Quick Start

### Text Generation (ROUGE/BLEU)
```python
from src.data_quality_checker import DataQualityChecker

checker = DataQualityChecker()
rouge = checker.compute_rouge("The quick brown fox", "A quick brown fox")
bleu = checker.compute_bleu(["the cat sat"], [["a cat sat"]])
```

Or use the lower-level calculators directly:
```python
from src.generative_ai_text_model import BLEUCalculator, ROUGECalculator

bleu_calc = BLEUCalculator()
rouge_calc = ROUGECalculator()

bleu_scores = bleu_calc.compute_bleu(["the cat sat"], [["a cat sat"]])
rouge_scores = rouge_calc.compute_rouge("The quick brown fox", "A quick brown fox")
```

### Binary Classification
```python
from src.binary_classification.binary_classifier_checker import BinaryClassifierChecker

checker = BinaryClassifierChecker()
metrics = checker.check_quality(
    y_true=y_true,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba,
)
# Returns: auroc, precision, recall, f1, log_loss, fpr, tpr, aupr
```

### Regression
```python
from src.regression.regression_checker import RegressionChecker

checker = RegressionChecker()
metrics = checker.check_quality(
    y_true=y_true,
    y_pred=y_pred,
)
# Returns: r2, explained_variance, rmse, mae, mse
```

### Unified Checker
```python
from src.unified_quality_checker import UnifiedDataQualityChecker

checker = UnifiedDataQualityChecker()

# Binary classification
report = checker.check_quality(predictions, references, task_type='classification')

# Regression
report = checker.check_quality(predictions, references, task_type='regression')

# Text generation
report = checker.check_quality(predictions, references, task_type='text_generation')
# Returns: {'rouge': {...}, 'bleu': {...}}
```

## Testing

### Run All Tests
```bash
python -m pytest tests/ -v

# Using manage.sh
bash manage.sh test-quick        # Without coverage
```

### Run Specific Tests
```bash
python -m pytest tests/test_binary_classification.py -v
python -m pytest tests/test_regression.py -v
python -m pytest tests/test_generative_ai.py -v
python -m pytest tests/test_utils.py -v
```

### Run Examples
```bash
# Run all examples
python run_examples.py

# Run all examples in a category
python run_examples.py generative_ai
python run_examples.py binary_classification
python run_examples.py regression

# Run a specific example
python run_examples.py generative_ai/example_1_basic_bleu_rouge
python run_examples.py binary_classification/example_1_basic_metrics
python run_examples.py regression/example_1_basic_metrics

# List all available examples
python run_examples.py -l

# Using manage.sh
bash manage.sh examples
```

## API Reference

### `src.generative_ai_text_model.BLEUCalculator`

| Method | Parameters | Returns |
|--------|-----------|---------|
| `compute_bleu(predictions, references, max_order=4, smooth=False)` | predictions: str/List[str], references: List[List[str]]/List[str], max_order: int, smooth: bool | Dict with bleu, precisions, brevity_penalty |

### `src.generative_ai_text_model.BLEUAggregator`

| Method | Parameters | Returns |
|--------|-----------|---------|
| `aggregate_bleu_scores(scores_list, aggregation_type='mean')` | scores_list: List[Dict], aggregation_type: str ('mean'/'median'/'min'/'max') | Aggregated BLEU scores |

### `src.generative_ai_text_model.ROUGECalculator`

| Method | Parameters | Returns |
|--------|-----------|---------|
| `compute_rouge(predictions, references, rouge_types=None, use_stemmer=False, use_aggregator=True)` | predictions: str/List[str], references: str/List[str], rouge_types: List[str] | Dict with rouge1, rouge2, rougeL, rougeS scores |

### `src.generative_ai_text_model.ROUGEAggregator`

| Method | Parameters | Returns |
|--------|-----------|---------|
| `aggregate_rouge_scores(scores_list, aggregation_type='mean')` | scores_list: List[Dict], aggregation_type: str ('mean'/'median'/'min'/'max') | Aggregated ROUGE scores |

### `src.data_quality_checker.DataQualityChecker`

Legacy driver class that wraps `BLEUCalculator` and `ROUGECalculator`.

| Method | Parameters | Returns |
|--------|-----------|---------|
| `compute_rouge(predictions, references, rouge_types=None, use_stemmer=False, use_aggregator=True)` | predictions: str/List[str], references: str/List[str] | Dict with rouge1, rouge2, rougeL, rougeS scores |
| `compute_bleu(predictions, references, max_order=4, smooth=False)` | predictions: str/List[str], references: List[List[str]]/List[str] | Dict with bleu, precisions, brevity_penalty |
| `compute_all_metrics(predictions, references, compute_rouge=True, compute_bleu=True)` | predictions, references | Dict combining rouge and bleu metrics |
| `batch_compute_metrics(predictions_list, references_list, metric_types=None)` | predictions_list: List, references_list: List | List of metric dicts per batch |
| `format_results(results, precision=4)` | results: Dict, precision: int | Formatted string |

### `src.data_quality_checker.QualityMetricsAggregator`

| Method | Parameters | Returns |
|--------|-----------|---------|
| `aggregate_rouge_scores(scores_list, aggregation_type='mean')` | scores_list: List[Dict], aggregation_type: str | Aggregated ROUGE scores |
| `aggregate_bleu_scores(scores_list, aggregation_type='mean')` | scores_list: List[Dict], aggregation_type: str | Aggregated BLEU scores |

### `src.binary_classification.BinaryClassifierChecker`

| Method | Parameters | Returns |
|--------|-----------|---------|
| `check_quality(y_true, y_pred, y_pred_proba=None, sample_weight=None, cache_key=None)` | y_true: array, y_pred: array, y_pred_proba: array (optional) | Dict with auroc, precision, recall, f1, log_loss, fpr, tpr, aupr |
| `compute_roc_auc(y_true, y_pred_proba)` | y_true: array, y_pred_proba: array | float |
| `compute_precision(y_true, y_pred)` | y_true: array, y_pred: array | float |
| `compute_recall(y_true, y_pred)` | y_true: array, y_pred: array | float |
| `compute_f1_measure(y_true, y_pred)` | y_true: array, y_pred: array | float |
| `compute_log_loss(y_true, y_pred_proba)` | y_true: array, y_pred_proba: array | float |
| `clear_cache()` | — | None |

### `src.regression.RegressionChecker`

| Method | Parameters | Returns |
|--------|-----------|---------|
| `check_quality(y_true, y_pred, sample_weight=None, cache_key=None)` | y_true: array, y_pred: array | Dict with r2, explained_variance, rmse, mae, mse |
| `compute_r_squared(y_true, y_pred)` | y_true: array, y_pred: array | float |
| `compute_explained_variance(y_true, y_pred)` | y_true: array, y_pred: array | float |
| `compute_rmse(y_true, y_pred)` | y_true: array, y_pred: array | float |
| `compute_mae(y_true, y_pred)` | y_true: array, y_pred: array | float |
| `compute_mse(y_true, y_pred)` | y_true: array, y_pred: array | float |
| `clear_cache()` | — | None |

### `src.unified_quality_checker.UnifiedDataQualityChecker`

| Method | Parameters | Returns |
|--------|-----------|---------|
| `check_quality(predictions, references, task_type, **kwargs)` | task_type: 'classification'/'regression'/'text_generation' | Metrics dict for the given task |
| `check_binary_classification_quality(y_true, y_pred, y_pred_proba=None, ...)` | Same as `BinaryClassifierChecker.check_quality` | Binary classification metrics |
| `check_regression_quality(y_true, y_pred, sample_weight=None, ...)` | Same as `RegressionChecker.check_quality` | Regression metrics |
| `compute_bleu(predictions, references, max_order=4, smooth=False)` | Same as `BLEUCalculator.compute_bleu` | BLEU scores |
| `compute_rouge(predictions, references, rouge_types=None, ...)` | Same as `ROUGECalculator.compute_rouge` | ROUGE scores |
| `aggregate_bleu_scores(scores_list, aggregation_type='mean')` | scores_list: List[Dict] | Aggregated BLEU scores |
| `aggregate_rouge_scores(scores_list, aggregation_type='mean')` | scores_list: List[Dict] | Aggregated ROUGE scores |
| `get_available_modules()` | — | Dict describing available modules |
| `clear_all_caches()` | — | None |

### `src.utils`

| Function | Parameters | Returns |
|----------|-----------|---------|
| `preprocess_text(text, lowercase=True, remove_punctuation=False)` | text: str | Preprocessed str |
| `tokenize_text(text)` | text: str | List[str] of tokens |
| `get_ngrams(tokens, n)` | tokens: List[str], n: int | List of n-gram tuples |
| `detect_data_quality_issues(predictions, references)` | predictions: List, references: List | Dict with empty, short, duplicate, mismatch issues |
| `calculate_precision_recall_f1(predictions, references)` | predictions: List[str], references: List[str] | Dict with precision, recall, f1 |
| `calculate_length_similarity(predictions, references)` | predictions: List[str], references: List[str] | Dict with avg lengths and ratio |

## Development Commands

```bash
# Setup
bash manage.sh install               # Install dependencies

# Testing
bash manage.sh test-quick            # Run tests without coverage
bash manage.sh test <file>           # Run a specific test file

# Examples
bash manage.sh examples              # Run all examples
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: evaluate` | `pip install evaluate` |
| `ImportError: rouge_score` | `pip install rouge-score` |
| ROUGE scores are 0 | Check case sensitivity, punctuation, tokenization, language match |
| BLEU score too low | Try ROUGE-L (order-independent), provide multiple references |
| Memory errors | Use batch processing with `batch_compute_metrics()` |

## References

- [ROUGE Metric](https://aclanthology.org/W04-1013/)
- [BLEU Score](https://aclanthology.org/P02-1040/)
- [Hugging Face Evaluate](https://huggingface.co/docs/evaluate/)

## License

MIT - See [LICENSE](LICENSE)

---

**Version:** 0.1.0 | **Author:** Henry Xiao
