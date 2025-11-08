"""
Unit tests for evaluation metrics.
"""

import pytest
import sys
sys.path.append('..')

import numpy as np
from src.utils.evaluation import (
    rmse, mae, precision_at_k, recall_at_k,
    average_precision_at_k, ndcg_at_k,
    coverage, diversity
)


def test_rmse():
    """Test RMSE calculation."""
    predictions = np.array([3.0, 4.0, 5.0])
    actuals = np.array([3.0, 4.0, 5.0])
    assert rmse(predictions, actuals) == pytest.approx(0.0, abs=1e-6)
    
    predictions = np.array([3.0, 4.0, 5.0])
    actuals = np.array([3.5, 4.5, 5.5])
    assert rmse(predictions, actuals) == pytest.approx(0.5, rel=1e-3)


def test_mae():
    """Test MAE calculation."""
    predictions = np.array([3.0, 4.0, 5.0])
    actuals = np.array([3.0, 4.0, 5.0])
    assert mae(predictions, actuals) == pytest.approx(0.0, abs=1e-6)
    
    predictions = np.array([3.0, 4.0, 5.0])
    actuals = np.array([3.5, 4.5, 5.5])
    assert mae(predictions, actuals) == pytest.approx(0.5, rel=1e-3)


def test_precision_at_k():
    """Test Precision@K calculation."""
    recommended = [1, 2, 3, 4, 5]
    relevant = [2, 4, 6]
    
    # At k=5, we recommended 2 and 4 which are relevant
    p = precision_at_k(recommended, relevant, k=5)
    assert p == pytest.approx(0.4, rel=1e-3)  # 2/5


def test_recall_at_k():
    """Test Recall@K calculation."""
    recommended = [1, 2, 3, 4, 5]
    relevant = [2, 4, 6]
    
    # At k=5, we found 2 out of 3 relevant items
    r = recall_at_k(recommended, relevant, k=5)
    assert r == pytest.approx(0.667, rel=1e-2)  # 2/3


def test_coverage():
    """Test coverage calculation."""
    all_recs = [
        [1, 2, 3],
        [2, 3, 4],
        [5, 6, 7]
    ]
    catalog = list(range(1, 11))  # 1-10
    
    # Unique items recommended: 1,2,3,4,5,6,7 = 7 out of 10
    cov = coverage(all_recs, catalog)
    assert cov == pytest.approx(0.7, rel=1e-3)


def test_diversity():
    """Test diversity calculation."""
    # Identical lists = no diversity
    all_recs = [[1, 2, 3], [1, 2, 3]]
    div = diversity(all_recs)
    assert div == pytest.approx(0.0, abs=1e-6)
    
    # Completely different lists = maximum diversity
    all_recs = [[1, 2, 3], [4, 5, 6]]
    div = diversity(all_recs)
    assert div == pytest.approx(1.0, rel=1e-3)