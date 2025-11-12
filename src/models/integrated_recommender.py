"""
Integrated recommender combining SVD and cold start strategies.
"""

from src.models.recommender import MovieRecommender
from src.models.cold_start import ColdStartHandler


class IntegratedRecommender:
    """
    Combines collaborative filtering with cold start handling.
    """
    
    def __init__(self, model_path: str, ratings_df, movies_df):
        # Load trained SVD model
        self.svd_recommender = MovieRecommender.load_model(model_path)
        
        # Initialize cold start handler
        self.cold_start_handler = ColdStartHandler(ratings_df, movies_df)
        
        self.ratings_df = ratings_df
    
    def get_recommendations(self, user_id: int, n: int = 10):
        """
        Get recommendations using appropriate strategy.
        """
        # Check user's rating count
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        if len(user_ratings) < 5:
            # Use cold start strategy
            return self.cold_start_handler.get_recommendations_for_user(
                user_id, n
            )
        else:
            # Use collaborative filtering
            return {
                'user_id': user_id,
                'num_ratings': len(user_ratings),
                'recommendations': self.svd_recommender.get_top_n_recommendations(
                    user_id, n
                ),
                'recommendation_type': 'personalized'
            }