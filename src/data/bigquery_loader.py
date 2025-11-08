"""
BigQuery data loader for MovieLens dataset.
Handles connection and queries to master-ai-cloud.MoviePlatform.
"""

from google.cloud import bigquery
import pandas as pd
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BigQueryLoader:
    """
    Handles loading data from BigQuery MoviePlatform dataset.
    
    Attributes:
        project_id (str): GCP project ID
        dataset_id (str): BigQuery dataset ID
        client (bigquery.Client): BigQuery client
    """
    
    def __init__(self, project_id: str = "master-ai-cloud", 
                 dataset_id: str = "MoviePlatform"):
        """
        Initialize BigQuery loader.
        
        Args:
            project_id: GCP project containing the dataset
            dataset_id: BigQuery dataset name
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.client = bigquery.Client()
        logger.info(f"BigQuery client initialized for {project_id}.{dataset_id}")
    
    def load_movies(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load movies table from BigQuery.
        
        Args:
            limit: Maximum number of rows to return (None for all)
            
        Returns:
            DataFrame with columns: movieId, title, genres
        """
        query = f"""
        SELECT movieId, title, genres
        FROM `{self.project_id}.{self.dataset_id}.movies`
        """
        
        if limit:
            query += f"\nLIMIT {limit}"
        
        logger.info(f"Loading movies (limit={limit})...")
        df = self.client.query(query).to_dataframe()
        logger.info(f"Loaded {len(df)} movies")
        
        return df
    
    def load_ratings(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load ratings table from BigQuery.
        
        Args:
            limit: Maximum number of rows to return (None for all)
            
        Returns:
            DataFrame with columns: userId, movieId, rating, timestamp
        """
        query = f"""
        SELECT userId, movieId, rating, timestamp
        FROM `{self.project_id}.{self.dataset_id}.ratings`
        """
        
        if limit:
            query += f"\nLIMIT {limit}"
        
        logger.info(f"Loading ratings (limit={limit})...")
        df = self.client.query(query).to_dataframe()
        logger.info(f"Loaded {len(df)} ratings")
        
        return df
    
    def get_ratings_stats(self) -> dict:
        """
        Get basic statistics about the ratings table.
        
        Returns:
            Dictionary with stats: total_ratings, num_users, num_movies, etc.
        """
        query = f"""
        SELECT 
            COUNT(*) as total_ratings,
            COUNT(DISTINCT userId) as num_users,
            COUNT(DISTINCT movieId) as num_movies,
            AVG(rating) as avg_rating,
            MIN(rating) as min_rating,
            MAX(rating) as max_rating
        FROM `{self.project_id}.{self.dataset_id}.ratings`
        """
        
        logger.info("Fetching ratings statistics...")
        result = self.client.query(query).to_dataframe()
        
        stats = result.iloc[0].to_dict()
        logger.info(f"Stats: {stats}")
        
        return stats
    
    def get_movie_stats(self) -> dict:
        """
        Get basic statistics about the movies table.
        
        Returns:
            Dictionary with stats: total_movies, num_genres, etc.
        """
        query = f"""
        SELECT 
            COUNT(*) as total_movies,
            COUNT(DISTINCT genres) as unique_genre_combinations
        FROM `{self.project_id}.{self.dataset_id}.movies`
        """
        
        logger.info("Fetching movie statistics...")
        result = self.client.query(query).to_dataframe()
        
        stats = result.iloc[0].to_dict()
        logger.info(f"Stats: {stats}")
        
        return stats
    
    def load_popular_movies(self, min_ratings: int = 50, limit: int = 100) -> pd.DataFrame:
        """
        Load most popular movies based on number of ratings.
        
        Args:
            min_ratings: Minimum number of ratings required
            limit: Maximum number of movies to return
            
        Returns:
            DataFrame with popular movies and their statistics
        """
        query = f"""
        SELECT 
            m.movieId,
            m.title,
            m.genres,
            COUNT(r.rating) as num_ratings,
            AVG(r.rating) as avg_rating
        FROM `{self.project_id}.{self.dataset_id}.movies` m
        JOIN `{self.project_id}.{self.dataset_id}.ratings` r
            ON m.movieId = r.movieId
        GROUP BY m.movieId, m.title, m.genres
        HAVING num_ratings >= {min_ratings}
        ORDER BY num_ratings DESC
        LIMIT {limit}
        """
        
        logger.info(f"Loading popular movies (min_ratings={min_ratings}, limit={limit})...")
        df = self.client.query(query).to_dataframe()
        logger.info(f"Loaded {len(df)} popular movies")
        
        return df


# Example usage and testing
if __name__ == "__main__":
    # Test the loader
    loader = BigQueryLoader()
    
    # Test loading movies
    print("\n=== Testing Movies Load ===")
    movies_sample = loader.load_movies(limit=10)
    print(movies_sample.head())
    
    # Test loading ratings
    print("\n=== Testing Ratings Load ===")
    ratings_sample = loader.load_ratings(limit=10)
    print(ratings_sample.head())
    
    # Test statistics
    print("\n=== Ratings Statistics ===")
    rating_stats = loader.get_ratings_stats()
    print(rating_stats)
    
    print("\n=== Movie Statistics ===")
    movie_stats = loader.get_movie_stats()
    print(movie_stats)
    
    # Test popular movies
    print("\n=== Popular Movies ===")
    popular = loader.load_popular_movies(min_ratings=100, limit=10)
    print(popular)