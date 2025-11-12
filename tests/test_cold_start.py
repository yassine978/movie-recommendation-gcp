"""
Tests for cold start handler.
"""

import pytest
import pandas as pd
import numpy as np
from src.models.cold_start import ColdStartHandler


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Sample ratings
    ratings = pd.DataFrame({
        'userId': [1]*5 + [2]*3 + [3]*10 + [4]*2,
        'movieId': [1,2,3,4,5, 1,3,5, 1,2,3,4,5,6,7,8,9,10, 1,2],
        'rating': [5,4,3,5,4, 3,4,5, 5,5,4,4,3,5,4,3,4,5, 2,3]
    })
    
    # Sample movies
    movies = pd.DataFrame({
        'movieId': range(1, 11),
        'title': [f'Movie {i}' for i in range(1, 11)],
        'genres': [
            'Action|Drama', 'Comedy', 'Drama', 'Action', 
            'Comedy|Romance', 'Drama|Romance', 'Action|Comedy',
            'Drama', 'Action|Drama', 'Comedy'
        ]
    })
    
    return ratings, movies


class TestColdStartHandler:
    """Test suite for ColdStartHandler."""
    
    def test_initialization(self, sample_data):
        """Test handler initialization."""
        ratings, movies = sample_data
        handler = ColdStartHandler(ratings, movies)
        
        assert handler.popular_movies is not None
        assert len(handler.popular_movies) > 0
        assert 'popularity_score' in handler.popular_movies.columns
    
    def test_popular_recommendations(self, sample_data):
        """Test popular movie recommendations."""
        ratings, movies = sample_data
        handler = ColdStartHandler(ratings, movies)
        
        # Get popular recommendations
        recs = handler.get_popular_recommendations(n=5)
        
        assert len(recs) <= 5
        assert all('movieId' in rec for rec in recs)
        assert all('popularity_score' in rec for rec in recs)
        
        # Check ordering (should be descending by popularity)
        scores = [r['popularity_score'] for r in recs]
        assert scores == sorted(scores, reverse=True)
    
    def test_genre_filtering(self, sample_data):
        """Test genre-based filtering."""
        ratings, movies = sample_data
        handler = ColdStartHandler(ratings, movies)
        
        # Get Action movies
        action_recs = handler.get_popular_recommendations(n=5, genre='Action')
        
        # All should contain Action genre
        for rec in action_recs:
            movie = movies[movies['movieId'] == rec['movieId']].iloc[0]
            assert 'Action' in movie['genres']
    
    def test_exclusion(self, sample_data):
        """Test movie exclusion."""
        ratings, movies = sample_data
        handler = ColdStartHandler(ratings, movies)
        
        # Exclude movies 1 and 2
        recs = handler.get_popular_recommendations(
            n=5, 
            exclude_movie_ids=[1, 2]
        )
        
        # Check that excluded movies are not in recommendations
        rec_ids = [r['movieId'] for r in recs]
        assert 1 not in rec_ids
        assert 2 not in rec_ids
    
    def test_genre_based_recommendations(self, sample_data):
        """Test genre-based recommendations."""
        ratings, movies = sample_data
        handler = ColdStartHandler(ratings, movies)
        
        # Get recommendations for users who like Action
        recs = handler.get_genre_based_recommendations(
            preferred_genres=['Action'],
            n=3
        )
        
        assert len(recs) <= 3
        # At least some should have Action genre
        action_count = sum(
            1 for r in recs 
            if 'Action' in movies[movies['movieId'] == r['movieId']].iloc[0]['genres']
        )
        assert action_count > 0
    
    def test_hybrid_recommendations(self, sample_data):
        """Test hybrid recommendations."""
        ratings, movies = sample_data
        handler = ColdStartHandler(ratings, movies)
        
        # Test with no ratings (should return popular)
        recs_no_ratings = handler.get_hybrid_recommendations({}, n=5)
        assert len(recs_no_ratings) <= 5
        
        # Test with some ratings
        user_ratings = {1: 5.0, 4: 4.5}  # Likes Action movies
        recs_with_ratings = handler.get_hybrid_recommendations(
            user_ratings, 
            n=5
        )
        
        assert len(recs_with_ratings) <= 5
        # Should exclude rated movies
        rec_ids = [r['movieId'] for r in recs_with_ratings]
        assert 1 not in rec_ids
        assert 4 not in rec_ids
    
    def test_user_recommendations(self, sample_data):
        """Test getting recommendations for specific users."""
        ratings, movies = sample_data
        handler = ColdStartHandler(ratings, movies)
        
        # Test for user with many ratings
        result = handler.get_recommendations_for_user(user_id=3, n=5)
        assert result['user_id'] == 3
        assert result['num_ratings'] == 10
        assert len(result['recommendations']) <= 5
        
        # Test for new user (not in dataset)
        result = handler.get_recommendations_for_user(user_id=999, n=5)
        assert result['user_id'] == 999
        assert result['num_ratings'] == 0
        assert result['recommendation_type'] == 'cold_start_popular'