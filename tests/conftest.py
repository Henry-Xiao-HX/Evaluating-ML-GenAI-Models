"""
Pytest configuration and shared fixtures for quality checks tests.
"""

import pytest
import numpy as np


@pytest.fixture
def binary_classification_data():
    """Fixture for binary classification test data."""
    np.random.seed(42)
    return {
        'y_true': np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0]),
        'y_pred': np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1]),
        'y_pred_proba': np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.15, 0.85, 0.92, 0.1, 0.6]),
    }


@pytest.fixture
def regression_data():
    """Fixture for regression test data."""
    np.random.seed(42)
    y_true = np.random.rand(50) * 100
    y_pred = y_true + np.random.normal(0, 5, 50)
    return {
        'y_true': y_true,
        'y_pred': y_pred,
    }


@pytest.fixture
def text_data():
    """Fixture for text generation test data."""
    return {
        'predictions': [
            "the quick brown fox jumps over the lazy dog",
            "machine learning is a subset of artificial intelligence",
        ],
        'references': [
            "a quick brown fox jumped over the lazy dog",
            "machine learning is part of artificial intelligence",
        ],
    }


def pytest_collection_modifyitems(config, items):
    """Add markers to tests for better organization."""
    for item in items:
        if "test_generative_ai" in str(item.fspath):
            item.add_marker(pytest.mark.text_metrics)
        elif "test_binary_classification" in str(item.fspath):
            item.add_marker(pytest.mark.binary_classification)
        elif "test_regression" in str(item.fspath):
            item.add_marker(pytest.mark.regression)
        elif "test_utils" in str(item.fspath):
            item.add_marker(pytest.mark.utils)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "text_metrics: tests for text generation metrics (BLEU, ROUGE)"
    )
    config.addinivalue_line(
        "markers", "binary_classification: tests for binary classification metrics"
    )
    config.addinivalue_line(
        "markers", "regression: tests for regression metrics"
    )
    config.addinivalue_line(
        "markers", "utils: tests for utility functions"
    )

# Made with Bob
