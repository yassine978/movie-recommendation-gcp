"""
Data preprocessing pipeline for recommendation system.
Handles train/test splits, filtering, and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses movie ratings data for recommendation models.
    
    Attributes:
        ratings (pd.DataFrame): Ratings data
        movies (pd.DataFrame): Movies metadata
    """
    
    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        """
        Initialize preprocessor with data.
        
        Args:
            ratings: DataFrame with columns [userId, movieId, rating, timestamp]
            movies: DataFrame with columns [movieId, title, genres]
        """
        self.ratings = ratings.copy()
        self.movies = movies.copy()
        logger.info(f"Initialized with {len(ratings)} ratings and {len(movies)} movies")
    
    def create_train_test_split(self, test_size: float = 0.2, 
                                random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split ratings into train and test sets.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info(f"Creating train/test split (test_size={test_size})...")
        
        train_df, test_df = train_test_split(
            self.ratings, 
            test_size=test_size, 
            random_state=random_state
        )
        
        logger.info(f"Train set: {len(train_df):,} ratings")
        logger.info(f"Test set: {len(test_df):,} ratings")
        
        return train_df, test_df
    
    def filter_cold_start_users(self, min_ratings: int = 5) -> pd.DataFrame:
        """
        Remove users with fewer than min_ratings ratings.
        
        Args:
            min_ratings: Minimum number of ratings required
            
        Returns:
            Filtered ratings DataFrame
        """
        logger.info(f"Filtering users with <{min_ratings} ratings...")
        
        # Count ratings per user
        user_counts = self.ratings.groupby('userId').size()
        
        # Get users with enough ratings
        valid_users = user_counts[user_counts >= min_ratings].index
        
        # Filter ratings
        filtered_ratings = self.ratings[self.ratings['userId'].isin(valid_users)]
        
        removed = len(self.ratings) - len(filtered_ratings)
        logger.info(f"Removed {removed:,} ratings from {len(user_counts) - len(valid_users)} users")
        logger.info(f"Remaining: {len(filtered_ratings):,} ratings from {len(valid_users)} users")
        
        return filtered_ratings
    
    def filter_cold_start_movies(self, min_ratings: int = 5) -> pd.DataFrame:
        """
        Remove movies with fewer than min_ratings ratings.
        
        Args:
            min_ratings: Minimum number of ratings required
            
        Returns:
            Filtered ratings DataFrame
        """
        logger.info(f"Filtering movies with <{min_ratings} ratings...")
        
        # Count ratings per movie
        movie_counts = self.ratings.groupby('movieId').size()
        
        # Get movies with enough ratings
        valid_movies = movie_counts[movie_counts >= min_ratings].index
        
        # Filter ratings
        filtered_ratings = self.ratings[self.ratings['movieId'].isin(valid_movies)]
        
        removed = len(self.ratings) - len(filtered_ratings)
        logger.info(f"Removed {removed:,} ratings from {len(movie_counts) - len(valid_movies)} movies")
        logger.info(f"Remaining: {len(filtered_ratings):,} ratings for {len(valid_movies)} movies")
        
        return filtered_ratings
    
    def extract_genre_features(self) -> pd.DataFrame:
        """
        Extract genre features from movies data.
        
        Returns:
            DataFrame with one-hot encoded genre columns
        """
        logger.info("Extracting genre features...")
        
        # Get all unique genres
        all_genres = set()
        for genres_str in self.movies['genres'].dropna():
            all_genres.update(genres_str.split('|'))
        
        all_genres = sorted(list(all_genres))
        logger.info(f"Found {len(all_genres)} unique genres")
        
        # Create one-hot encoding
        genre_df = self.movies[['movieId']].copy()
        
        for genre in all_genres:
            genre_df[f'genre_{genre}'] = self.movies['genres'].apply(
                lambda x: 1 if isinstance(x, str) and genre in x else 0
            )
        
        return genre_df
    
    def add_temporal_features(self) -> pd.DataFrame:
        """
        Add temporal features from timestamps.
        
        Returns:
            Ratings DataFrame with added temporal features
        """
        logger.info("Adding temporal features...")
        
        ratings_with_time = self.ratings.copy()
        
        # Convert timestamp to datetime
        ratings_with_time['datetime'] = pd.to_datetime(
            ratings_with_time['timestamp'], unit='s'
        )
        
        # Extract features
        ratings_with_time['year'] = ratings_with_time['datetime'].dt.year
        ratings_with_time['month'] = ratings_with_time['datetime'].dt.month
        ratings_with_time['day_of_week'] = ratings_with_time['datetime'].dt.dayofweek
        
        logger.info("Added temporal features: year, month, day_of_week")
        
        return ratings_with_time
    
    def get_statistics(self, df: Optional[pd.DataFrame] = None) -> dict:
        """
        Calculate statistics for a ratings DataFrame.
        
        Args:
            df: DataFrame to analyze (uses self.ratings if None)
            
        Returns:
            Dictionary with statistics
        """
        if df is None:
            df = self.ratings
        
        stats = {
            'num_ratings': len(df),
            'num_users': df['userId'].nunique(),
            'num_movies': df['movieId'].nunique(),
            'avg_rating': df['rating'].mean(),
            'min_rating': df['rating'].min(),
            'max_rating': df['rating'].max(),
            'sparsity': 1 - len(df) / (df['userId'].nunique() * df['movieId'].nunique())
        }
        
        return stats
    
    def prepare_for_surprise(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare data in format required by surprise library.
        
        Args:
            df: DataFrame to prepare (uses self.ratings if None)
            
        Returns:
            DataFrame with columns [userId, movieId, rating] in correct types
        """
        if df is None:
            df = self.ratings
        
        logger.info("Preparing data for Surprise library...")
        
        # Select required columns
        surprise_df = df[['userId', 'movieId', 'rating']].copy()
        
        # Ensure correct types
        surprise_df['userId'] = surprise_df['userId'].astype(int)
        surprise_df['movieId'] = surprise_df['movieId'].astype(int)
        surprise_df['rating'] = surprise_df['rating'].astype(float)
        
        logger.info(f"Prepared {len(surprise_df):,} ratings for Surprise")
        
        return surprise_df


# Example usage and testing
if __name__ == "__main__":
    from bigquery_loader import BigQueryLoader
    
    # Load data
    print("Loading data...")
    loader = BigQueryLoader()
    ratings = loader.load_ratings(limit=10000)
    movies = loader.load_movies()
    
    # Initialize preprocessor
    print("\n=== Initializing Preprocessor ===")
    preprocessor = DataPreprocessor(ratings, movies)
    
    # Test train/test split
    print("\n=== Train/Test Split ===")
    train, test = preprocessor.create_train_test_split(test_size=0.2)
    print(f"Train: {len(train):,}, Test: {len(test):,}")
    
    # Test filtering
    print("\n=== Filtering Cold-Start Users ===")
    filtered = preprocessor.filter_cold_start_users(min_ratings=5)
    print(f"Filtered to {len(filtered):,} ratings")
    
    # Test statistics
    print("\n=== Statistics ===")
    stats = preprocessor.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value:,}")
    
    # Test genre features
    print("\n=== Genre Features ===")
    genre_df = preprocessor.extract_genre_features()
    print(f"Shape: {genre_df.shape}")
    print(genre_df.head())
    
    # Test Surprise preparation
    print("\n=== Surprise Format ===")
    surprise_df = preprocessor.prepare_for_surprise(train)
    print(surprise_df.head())
    
    print("\n✓ All preprocessing functions working!")