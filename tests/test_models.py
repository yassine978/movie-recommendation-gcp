"""
Unit tests for baseline models.
"""

import pytest
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from src.models.baseline import (
    GlobalAverageBaseline,
    MovieAverageBaseline,
    UserAverageBaseline,
    PopularityBasedRecommender
)


@pytest.fixture
def sample_ratings():
    """Create sample ratings data for testing."""
    return pd.DataFrame({
        'userId': [1, 1, 2, 2, 3, 3, 4, 4],
        'movieId': [1, 2, 1, 3, 2, 3, 1, 4],
        'rating': [4.0, 3.5, 5.0, 4.0, 3.0, 4.5, 4.5, 2.5]
    })


@pytest.fixture
def sample_movies():
    """Create sample movies data for testing."""
    return pd.DataFrame({
        'movieId': [1, 2, 3, 4],
        'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
        'genres': ['Action', 'Comedy', 'Action|Drama', 'Comedy']
    })


def test_global_average_baseline(sample_ratings):
    """Test global average baseline model."""
    model = GlobalAverageBaseline()
    
    # Test training
    model.train(sample_ratings)
    assert model.trained == True
    assert model.global_avg == pytest.approx(3.875, rel=1e-3)
    
    # Test prediction
    pred = model.predict(1, 1)
    assert pred == pytest.approx(3.875, rel=1e-3)
    
    # Test batch prediction
    pairs = [(1, 1), (2, 2), (3, 3)]
    preds = model.predict_batch(pairs)
    assert len(preds) == 3
    assert all(p == pytest.approx(3.875, rel=1e-3) for p in preds)


def test_movie_average_baseline(sample_ratings):
    """Test movie average baseline model."""
    model = MovieAverageBaseline()
    
    # Test training
    model.train(sample_ratings)
    assert model.trained == True
    
    # Test prediction for known movie
    pred = model.predict(1, 1)  # Movie 1 avg = (4.0 + 5.0 + 4.5) / 3 = 4.5
    assert pred == pytest.approx(4.5, rel=1e-3)
    
    # Test prediction for unknown movie (should return global avg)
    pred = model.predict(1, 999)
    assert pred == pytest.approx(model.global_avg, rel=1e-3)


def test_user_average_baseline(sample_ratings):
    """Test user average baseline model."""
    model = UserAverageBaseline()
    
    # Test training
    model.train(sample_ratings)
    assert model.trained == True
    
    # Test prediction for known user
    pred = model.predict(1, 1)  # User 1 avg = (4.0 + 3.5) / 2 = 3.75
    assert pred == pytest.approx(3.75, rel=1e-3)
    
    # Test prediction for unknown user (should return global avg)
    pred = model.predict(999, 1)
    assert pred == pytest.approx(model.global_avg, rel=1e-3)


def test_popularity_based_recommender(sample_ratings, sample_movies):
    """Test popularity-based recommender."""
    model = PopularityBasedRecommender(min_ratings=2)
    
    # Test training
    model.train(sample_ratings, sample_movies)
    assert model.trained == True
    
    # Test recommendations
    recs = model.get_top_n_recommendations(n=2)
    assert len(recs) <= 2
    assert 'title' in recs.columns
    assert 'popularity_score' in recs.columns
    
    # Test genre filtering
    action_recs = model.get_top_n_recommendations(n=5, genre='Action')
    assert all('Action' in genres for genres in action_recs['genres'])


def test_model_not_trained_error():
    """Test that models raise error when predicting before training."""
    model = GlobalAverageBaseline()
    
    with pytest.raises(ValueError, match="Model not trained"):
        model.predict(1, 1)