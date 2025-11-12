"""
Unit tests for the MovieRecommender model.
"""

import pytest
import pandas as pd
import numpy as np
from src.models.recommender import MovieRecommender
import tempfile
import os


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create sample ratings
    ratings = pd.DataFrame({
        'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3] * 10,
        'movieId': [1, 2, 3, 1, 4, 5, 2, 3, 6] * 10,
        'rating': [5.0, 4.0, 3.5, 4.5, 5.0, 2.0, 3.0, 4.0, 5.0] * 10
    })
    
    # Create sample movies
    movies = pd.DataFrame({
        'movieId': range(1, 7),
        'title': [f'Movie {i}' for i in range(1, 7)],
        'genres': ['Action|Drama', 'Comedy', 'Drama', 'Action', 'Comedy|Romance', 'Drama']
    })
    
    return ratings, movies


@pytest.fixture
def trained_model(sample_data):
    """Create a trained model for testing."""
    ratings, movies = sample_data
    recommender = MovieRecommender(n_factors=10, n_epochs=5, verbose=False)
    recommender.train(ratings, movies, test_size=0.2)
    return recommender


class TestMovieRecommender:
    """Test suite for MovieRecommender class."""
    
    def test_initialization(self):
        """Test model initialization."""
        recommender = MovieRecommender(n_factors=50, n_epochs=10)
        
        assert recommender.n_factors == 50
        assert recommender.n_epochs == 10
        assert recommender.is_trained == False
        assert recommender.model is not None
    
    def test_training(self, sample_data):
        """Test model training."""
        ratings, movies = sample_data
        recommender = MovieRecommender(n_factors=10, n_epochs=5, verbose=False)
        
        metrics = recommender.train(ratings, movies, test_size=0.2)
        
        assert recommender.is_trained == True
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['n_train'] > 0
        assert metrics['n_test'] > 0
    
    def test_prediction(self, trained_model):
        """Test rating prediction."""
        # Test prediction for known user-movie pair
        pred = trained_model.predict_rating(user_id=1, movie_id=2)
        
        assert isinstance(pred, float)
        assert 0.5 <= pred <= 5.0
    
    def test_recommendations(self, trained_model):
        """Test getting recommendations."""
        recs = trained_model.get_top_n_recommendations(user_id=1, n=3)
        
        assert len(recs) <= 3
        assert all('movieId' in rec for rec in recs)
        assert all('title' in rec for rec in recs)
        assert all('predicted_rating' in rec for rec in recs)
        assert all(0.5 <= rec['predicted_rating'] <= 5.0 for rec in recs)
    
    def test_similar_movies(self, trained_model):
        """Test finding similar movies."""
        similar = trained_model.get_similar_movies(movie_id=1, n=2)
        
        # May return empty if movie not in training set
        if similar:
            assert len(similar) <= 2
            assert all('movieId' in movie for movie in similar)
            assert all('similarity_score' in movie for movie in similar)
    
    def test_save_load_model(self, trained_model):
        """Test saving and loading model."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save model
            trained_model.save_model(tmp_path)
            assert os.path.exists(tmp_path)
            
            # Load model
            loaded_model = MovieRecommender.load_model(tmp_path)
            
            assert loaded_model.is_trained == True
            assert loaded_model.n_factors == trained_model.n_factors
            
            # Test that loaded model can make predictions
            pred1 = trained_model.predict_rating(1, 2)
            pred2 = loaded_model.predict_rating(1, 2)
            
            # Predictions should be identical
            assert abs(pred1 - pred2) < 0.001
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_evaluate(self, trained_model, sample_data):
        """Test model evaluation."""
        ratings, _ = sample_data
        test_data = ratings.iloc[:10]  # Small test set
        
        metrics = trained_model.evaluate(test_data)
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0


def test_edge_cases():
    """Test edge cases and error handling."""
    recommender = MovieRecommender()
    
    # Test calling predict before training
    with pytest.raises(ValueError, match="Model not trained"):
        recommender.predict_rating(1, 1)
    
    # Test calling recommendations before training
    with pytest.raises(ValueError, match="Model not trained"):
        recommender.get_top_n_recommendations(1)
    
    # Test saving untrained model
    with pytest.raises(ValueError, match="Model not trained"):
        recommender.save_model("test.pkl")