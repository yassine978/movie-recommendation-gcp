"""
Baseline recommendation models.
Used as benchmarks to compare against collaborative filtering models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlobalAverageBaseline:
    """
    Predicts the global average rating for all user-movie pairs.
    Simplest possible baseline.
    """
    
    def __init__(self):
        """Initialize the baseline model."""
        self.global_avg = None
        self.trained = False
    
    def train(self, ratings: pd.DataFrame):
        """
        Train model by calculating global average.
        
        Args:
            ratings: DataFrame with 'rating' column
        """
        logger.info("Training Global Average Baseline...")
        self.global_avg = ratings['rating'].mean()
        self.trained = True
        logger.info(f"Global average rating: {self.global_avg:.4f}")
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating (returns global average).
        
        Args:
            user_id: User ID (ignored)
            movie_id: Movie ID (ignored)
            
        Returns:
            Predicted rating (global average)
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.global_avg
    
    def predict_batch(self, user_movie_pairs: List[Tuple[int, int]]) -> np.ndarray:
        """
        Predict ratings for multiple user-movie pairs.
        
        Args:
            user_movie_pairs: List of (user_id, movie_id) tuples
            
        Returns:
            Array of predictions
        """
        return np.full(len(user_movie_pairs), self.global_avg)


class MovieAverageBaseline:
    """
    Predicts the average rating for each movie.
    Falls back to global average for unseen movies.
    """
    
    def __init__(self):
        """Initialize the baseline model."""
        self.movie_avgs = {}
        self.global_avg = None
        self.trained = False
    
    def train(self, ratings: pd.DataFrame):
        """
        Train model by calculating per-movie averages.
        
        Args:
            ratings: DataFrame with 'movieId' and 'rating' columns
        """
        logger.info("Training Movie Average Baseline...")
        
        # Calculate global average for fallback
        self.global_avg = ratings['rating'].mean()
        
        # Calculate per-movie averages
        self.movie_avgs = ratings.groupby('movieId')['rating'].mean().to_dict()
        
        self.trained = True
        logger.info(f"Learned averages for {len(self.movie_avgs)} movies")
        logger.info(f"Global average (fallback): {self.global_avg:.4f}")
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating using movie average.
        
        Args:
            user_id: User ID (ignored)
            movie_id: Movie ID
            
        Returns:
            Predicted rating (movie average or global average)
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Return movie average if available, else global average
        return self.movie_avgs.get(movie_id, self.global_avg)
    
    def predict_batch(self, user_movie_pairs: List[Tuple[int, int]]) -> np.ndarray:
        """
        Predict ratings for multiple user-movie pairs.
        
        Args:
            user_movie_pairs: List of (user_id, movie_id) tuples
            
        Returns:
            Array of predictions
        """
        predictions = []
        for user_id, movie_id in user_movie_pairs:
            predictions.append(self.predict(user_id, movie_id))
        
        return np.array(predictions)


class UserAverageBaseline:
    """
    Predicts the average rating given by each user.
    Falls back to global average for unseen users.
    """
    
    def __init__(self):
        """Initialize the baseline model."""
        self.user_avgs = {}
        self.global_avg = None
        self.trained = False
    
    def train(self, ratings: pd.DataFrame):
        """
        Train model by calculating per-user averages.
        
        Args:
            ratings: DataFrame with 'userId' and 'rating' columns
        """
        logger.info("Training User Average Baseline...")
        
        # Calculate global average for fallback
        self.global_avg = ratings['rating'].mean()
        
        # Calculate per-user averages
        self.user_avgs = ratings.groupby('userId')['rating'].mean().to_dict()
        
        self.trained = True
        logger.info(f"Learned averages for {len(self.user_avgs)} users")
        logger.info(f"Global average (fallback): {self.global_avg:.4f}")
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating using user average.
        
        Args:
            user_id: User ID
            movie_id: Movie ID (ignored)
            
        Returns:
            Predicted rating (user average or global average)
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Return user average if available, else global average
        return self.user_avgs.get(user_id, self.global_avg)
    
    def predict_batch(self, user_movie_pairs: List[Tuple[int, int]]) -> np.ndarray:
        """
        Predict ratings for multiple user-movie pairs.
        
        Args:
            user_movie_pairs: List of (user_id, movie_id) tuples
            
        Returns:
            Array of predictions
        """
        predictions = []
        for user_id, movie_id in user_movie_pairs:
            predictions.append(self.predict(user_id, movie_id))
        
        return np.array(predictions)


class PopularityBasedRecommender:
    """
    Recommends popular movies (most rated + highest rated).
    Uses Bayesian average to balance popularity and quality.
    """
    
    def __init__(self, min_ratings: int = 10):
        """
        Initialize popularity-based recommender.
        
        Args:
            min_ratings: Minimum ratings required for a movie to be considered
        """
        self.min_ratings = min_ratings
        self.popular_movies = None
        self.trained = False
    
    def train(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        """
        Train by calculating popularity scores.
        
        Args:
            ratings: DataFrame with 'movieId' and 'rating' columns
            movies: DataFrame with 'movieId', 'title', 'genres' columns
        """
        logger.info("Training Popularity-Based Recommender...")
        
        # Calculate statistics per movie
        movie_stats = ratings.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        movie_stats.columns = ['movieId', 'num_ratings', 'avg_rating']
        
        # Filter by minimum ratings
        movie_stats = movie_stats[movie_stats['num_ratings'] >= self.min_ratings]
        
        # Calculate Bayesian average
        # WR = (v / (v + m)) * R + (m / (v + m)) * C
        # where:
        # v = number of ratings
        # m = minimum ratings required (confidence)
        # R = average rating
        # C = mean rating across all movies
        
        global_avg = ratings['rating'].mean()
        m = self.min_ratings
        
        movie_stats['popularity_score'] = (
            (movie_stats['num_ratings'] / (movie_stats['num_ratings'] + m)) * movie_stats['avg_rating'] +
            (m / (movie_stats['num_ratings'] + m)) * global_avg
        )
        
        # Merge with movie info
        self.popular_movies = movie_stats.merge(
            movies[['movieId', 'title', 'genres']], 
            on='movieId'
        ).sort_values('popularity_score', ascending=False)
        
        self.trained = True
        logger.info(f"Identified {len(self.popular_movies)} popular movies")
        logger.info(f"Top movie: {self.popular_movies.iloc[0]['title']} "
                   f"(score: {self.popular_movies.iloc[0]['popularity_score']:.2f})")
    
    def get_top_n_recommendations(self, n: int = 10, genre: str = None) -> pd.DataFrame:
        """
        Get top N popular movies.
        
        Args:
            n: Number of recommendations
            genre: Optional genre filter
            
        Returns:
            DataFrame with top N popular movies
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Filter by genre if specified
        if genre:
            filtered = self.popular_movies[
                self.popular_movies['genres'].str.contains(genre, case=False, na=False)
            ]
        else:
            filtered = self.popular_movies
        
        return filtered.head(n)[['movieId', 'title', 'genres', 'popularity_score', 'num_ratings', 'avg_rating']]


# Example usage and testing
if __name__ == "__main__":
    import sys
    sys.path.append('../..')
    from src.data.bigquery_loader import BigQueryLoader
    
    # Load data
    print("Loading data...")
    loader = BigQueryLoader()
    ratings = loader.load_ratings(limit=10000)
    movies = loader.load_movies()
    
    # Test Global Average
    print("\\n=== Global Average Baseline ===")
    global_model = GlobalAverageBaseline()
    global_model.train(ratings)
    pred = global_model.predict(1, 1)
    print(f"Prediction for user 1, movie 1: {pred:.4f}")
    
    # Test Movie Average
    print("\\n=== Movie Average Baseline ===")
    movie_model = MovieAverageBaseline()
    movie_model.train(ratings)
    pred = movie_model.predict(1, 1)
    print(f"Prediction for user 1, movie 1: {pred:.4f}")
    
    # Test User Average
    print("\\n=== User Average Baseline ===")
    user_model = UserAverageBaseline()
    user_model.train(ratings)
    pred = user_model.predict(1, 1)
    print(f"Prediction for user 1, movie 1: {pred:.4f}")
    
    # Test Popularity Recommender
    print("\\n=== Popularity-Based Recommender ===")
    pop_model = PopularityBasedRecommender(min_ratings=10)
    pop_model.train(ratings, movies)
    
    top_10 = pop_model.get_top_n_recommendations(n=10)
    print("\\nTop 10 Popular Movies:")
    print(top_10[['title', 'popularity_score', 'num_ratings']])
    
    # Test genre filtering
    action_movies = pop_model.get_top_n_recommendations(n=5, genre='Action')
    print("\\nTop 5 Action Movies:")
    print(action_movies[['title', 'popularity_score']])
    
    print("\\n✓ All baseline models working!")