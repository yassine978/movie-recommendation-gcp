"""
Cold Start Handler for new users with no or few ratings.
Implements multiple strategies for handling cold start problem.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)


class ColdStartHandler:
    """
    Handles cold start problem for new users.
    
    Strategies:
    1. Popularity-based: Recommend most popular movies (Bayesian average)
    2. Genre-based: Recommend based on user's preferred genres
    3. Hybrid: Combine popularity and genre preferences
    """
    
    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        """
        Initialize the cold start handler.
        
        Args:
            ratings_df: DataFrame with all ratings
            movies_df: DataFrame with movie metadata
        """
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        
        # Precompute popular movies
        self._compute_popularity_scores()
        
        # Precompute genre statistics
        self._compute_genre_statistics()
        
        logger.info("ColdStartHandler initialized")
    
    def _compute_popularity_scores(self):
        """
        Compute popularity scores using Bayesian average.
        
        Bayesian Average = (v / (v + m)) * R + (m / (v + m)) * C
        where:
        - R = average rating for the movie
        - v = number of votes for the movie
        - m = minimum votes required (threshold)
        - C = mean rating across all movies
        """
        # Calculate movie statistics
        movie_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        movie_stats.columns = ['movieId', 'avg_rating', 'num_ratings']
        
        # Calculate global mean rating
        C = self.ratings_df['rating'].mean()
        
        # Set minimum votes threshold (25th percentile)
        m = movie_stats['num_ratings'].quantile(0.25)
        
        # Calculate Bayesian average
        movie_stats['popularity_score'] = (
            (movie_stats['num_ratings'] / (movie_stats['num_ratings'] + m)) * movie_stats['avg_rating'] +
            (m / (movie_stats['num_ratings'] + m)) * C
        )
        
        # Merge with movie metadata
        self.popular_movies = pd.merge(
            movie_stats,
            self.movies_df,
            on='movieId',
            how='left'
        )
        
        # Sort by popularity score
        self.popular_movies = self.popular_movies.sort_values(
            'popularity_score', 
            ascending=False
        )
        
        logger.info(f"Computed popularity scores for {len(self.popular_movies)} movies")
        logger.info(f"Top movie: {self.popular_movies.iloc[0]['title']} "
                   f"(score: {self.popular_movies.iloc[0]['popularity_score']:.2f})")
    
    def _compute_genre_statistics(self):
        """
        Compute statistics for each genre.
        """
        # Expand genres (handle pipe-separated genres)
        genre_data = []
        for _, movie in self.movies_df.iterrows():
            if pd.notna(movie['genres']):
                genres = movie['genres'].split('|')
                for genre in genres:
                    genre_data.append({
                        'movieId': movie['movieId'],
                        'genre': genre
                    })
        
        self.genre_df = pd.DataFrame(genre_data)
        
        # Merge with ratings
        genre_ratings = pd.merge(
            self.genre_df,
            self.ratings_df,
            on='movieId'
        )
        
        # Calculate genre statistics
        self.genre_stats = genre_ratings.groupby('genre').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        self.genre_stats.columns = ['genre', 'avg_rating', 'num_ratings']
        
        logger.info(f"Computed statistics for {len(self.genre_stats)} genres")
    
    def get_popular_recommendations(
        self, 
        n: int = 10,
        genre: Optional[str] = None,
        exclude_movie_ids: List[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get popular movie recommendations.
        
        Args:
            n: Number of recommendations
            genre: Optional genre filter
            exclude_movie_ids: Movie IDs to exclude
            
        Returns:
            List of recommended movies
        """
        # Start with all popular movies
        candidates = self.popular_movies.copy()
        
        # Filter by genre if specified
        if genre:
            candidates = candidates[
                candidates['genres'].str.contains(genre, na=False)
            ]
        
        # Exclude specified movies
        if exclude_movie_ids:
            candidates = candidates[
                ~candidates['movieId'].isin(exclude_movie_ids)
            ]
        
        # Get top N
        top_movies = candidates.head(n)
        
        # Format results
        recommendations = []
        for _, movie in top_movies.iterrows():
            recommendations.append({
                'movieId': int(movie['movieId']),
                'title': movie['title'],
                'genres': movie['genres'],
                'popularity_score': round(movie['popularity_score'], 2),
                'num_ratings': int(movie['num_ratings'])
            })
        
        return recommendations
    
    def get_genre_based_recommendations(
        self,
        preferred_genres: List[str],
        n: int = 10,
        exclude_movie_ids: List[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations based on genre preferences.
        
        Args:
            preferred_genres: List of preferred genres
            n: Number of recommendations
            exclude_movie_ids: Movie IDs to exclude
            
        Returns:
            List of recommended movies
        """
        # Score movies based on genre match
        movie_scores = []
        
        for _, movie in self.popular_movies.iterrows():
            if pd.notna(movie['genres']):
                movie_genres = set(movie['genres'].split('|'))
                
                # Calculate genre match score
                genre_score = len(
                    set(preferred_genres).intersection(movie_genres)
                ) / len(preferred_genres)
                
                # Combine with popularity
                combined_score = (
                    0.6 * movie['popularity_score'] / 5.0 +  # Normalized popularity
                    0.4 * genre_score  # Genre match
                )
                
                movie_scores.append({
                    'movieId': movie['movieId'],
                    'title': movie['title'],
                    'genres': movie['genres'],
                    'score': combined_score,
                    'popularity_score': movie['popularity_score'],
                    'num_ratings': movie['num_ratings']
                })
        
        # Sort by score
        movie_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Exclude specified movies
        if exclude_movie_ids:
            movie_scores = [
                m for m in movie_scores 
                if m['movieId'] not in exclude_movie_ids
            ]
        
        # Return top N
        return movie_scores[:n]
    
    def get_hybrid_recommendations(
        self,
        user_ratings: Dict[int, float],
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get hybrid recommendations based on limited user ratings.
        
        Args:
            user_ratings: Dictionary of {movieId: rating} for user's ratings
            n: Number of recommendations
            
        Returns:
            List of recommended movies
        """
        if not user_ratings:
            # No ratings - return popular
            return self.get_popular_recommendations(n=n)
        
        # Extract preferred genres from rated movies
        preferred_genres = []
        rated_movie_ids = list(user_ratings.keys())
        
        for movie_id in rated_movie_ids:
            movie = self.movies_df[self.movies_df['movieId'] == movie_id]
            if not movie.empty and pd.notna(movie.iloc[0]['genres']):
                # Weight genres by rating
                rating = user_ratings[movie_id]
                if rating >= 3.5:  # User liked this movie
                    genres = movie.iloc[0]['genres'].split('|')
                    preferred_genres.extend(genres)
        
        if not preferred_genres:
            # No clear preferences - return popular excluding rated
            return self.get_popular_recommendations(
                n=n,
                exclude_movie_ids=rated_movie_ids
            )
        
        # Get most common genres
        genre_counts = Counter(preferred_genres)
        top_genres = [genre for genre, _ in genre_counts.most_common(3)]
        
        # Get genre-based recommendations
        return self.get_genre_based_recommendations(
            preferred_genres=top_genres,
            n=n,
            exclude_movie_ids=rated_movie_ids
        )
    
    def get_recommendations_for_user(
        self,
        user_id: int,
        n: int = 10,
        strategy: str = 'hybrid'
    ) -> Dict[str, Any]:
        """
        Get recommendations for a specific user based on their rating history.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            strategy: 'popular', 'genre', or 'hybrid'
            
        Returns:
            Dictionary with recommendations and metadata
        """
        # Get user's ratings
        user_ratings_df = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        if user_ratings_df.empty:
            # New user - cold start
            recommendations = self.get_popular_recommendations(n=n)
            recommendation_type = 'cold_start_popular'
            num_ratings = 0
        else:
            num_ratings = len(user_ratings_df)
            user_ratings = dict(zip(
                user_ratings_df['movieId'],
                user_ratings_df['rating']
            ))
            
            if num_ratings < 5:
                # Few ratings - use hybrid approach
                recommendations = self.get_hybrid_recommendations(
                    user_ratings=user_ratings,
                    n=n
                )
                recommendation_type = 'cold_start_hybrid'
            else:
                # Enough ratings - this would use collaborative filtering
                # For now, return hybrid as placeholder
                recommendations = self.get_hybrid_recommendations(
                    user_ratings=user_ratings,
                    n=n
                )
                recommendation_type = 'collaborative_filtering_ready'
        
        return {
            'user_id': user_id,
            'num_ratings': num_ratings,
            'recommendations': recommendations,
            'recommendation_type': recommendation_type
        }
    
    def analyze_cold_start_distribution(self) -> pd.DataFrame:
        """
        Analyze the distribution of cold start users.
        
        Returns:
            DataFrame with cold start analysis
        """
        user_stats = self.ratings_df.groupby('userId').size().reset_index(name='num_ratings')
        
        categories = []
        for _, user in user_stats.iterrows():
            if user['num_ratings'] == 0:
                category = 'new_user'
            elif user['num_ratings'] < 5:
                category = 'cold_start'
            elif user['num_ratings'] < 20:
                category = 'moderate'
            else:
                category = 'active'
            
            categories.append({
                'userId': user['userId'],
                'num_ratings': user['num_ratings'],
                'category': category
            })
        
        df = pd.DataFrame(categories)
        
        # Summary statistics
        summary = df.groupby('category').agg({
            'userId': 'count',
            'num_ratings': ['mean', 'min', 'max']
        }).reset_index()
        
        summary.columns = ['category', 'user_count', 'avg_ratings', 'min_ratings', 'max_ratings']
        
        return summary