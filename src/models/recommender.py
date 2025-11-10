"""
Movie Recommender using SVD (Singular Value Decomposition)
This module implements the core recommendation engine using collaborative filtering.
"""

import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieRecommender:
    """
    A movie recommendation system using SVD collaborative filtering.
    
    This class handles:
    - Training SVD models on user-movie ratings
    - Generating personalized recommendations
    - Saving and loading trained models
    - Evaluating model performance
    """
    
    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize the MovieRecommender with SVD parameters.
        
        Args:
            n_factors: Number of latent factors for matrix factorization
            n_epochs: Number of training iterations
            lr_all: Learning rate for all parameters
            reg_all: Regularization term for all parameters
            random_state: Random seed for reproducibility
            verbose: Whether to print training progress
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state
        self.verbose = verbose
        
        # Initialize the SVD algorithm
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=random_state,
            verbose=verbose
        )
        
        # Model state
        self.is_trained = False
        self.training_metrics = {}
        self.movies_df = None
        self.trainset = None
        self.testset = None
        
        logger.info(f"MovieRecommender initialized with {n_factors} factors, {n_epochs} epochs")
    
    def prepare_data(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> Dataset:
        """
        Prepare the data for Surprise library format.
        
        Args:
            ratings_df: DataFrame with columns ['userId', 'movieId', 'rating']
            movies_df: DataFrame with movie metadata
            
        Returns:
            Surprise Dataset object
        """
        logger.info(f"Preparing data: {len(ratings_df)} ratings, {len(movies_df)} movies")
        
        # Store movies dataframe for later use
        self.movies_df = movies_df
        
        # Define the rating scale
        reader = Reader(rating_scale=(0.5, 5.0))
        
        # Create the dataset
        data = Dataset.load_from_df(
            ratings_df[['userId', 'movieId', 'rating']], 
            reader
        )
        
        return data
    
    def train(
        self, 
        ratings_df: pd.DataFrame, 
        movies_df: pd.DataFrame,
        test_size: float = 0.2,
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Train the SVD model on the ratings data.
        
        Args:
            ratings_df: DataFrame with user ratings
            movies_df: DataFrame with movie metadata
            test_size: Proportion of data to use for testing
            save_path: Optional path to save the trained model
            
        Returns:
            Dictionary with training metrics (RMSE, MAE)
        """
        logger.info("Starting model training...")
        start_time = datetime.now()
        
        # Prepare the data
        data = self.prepare_data(ratings_df, movies_df)
        
        # Create train/test split
        self.trainset, self.testset = train_test_split(
            data, 
            test_size=test_size, 
            random_state=self.random_state
        )
        
        # Train the model
        logger.info(f"Training on {self.trainset.n_ratings} ratings...")
        self.model.fit(self.trainset)
        
        # Evaluate on test set
        predictions = self.model.test(self.testset)
        
        # Calculate metrics
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        
        # Store training metrics
        self.training_metrics = {
            'rmse': rmse,
            'mae': mae,
            'n_train': self.trainset.n_ratings,
            'n_test': len(self.testset),
            'training_time': (datetime.now() - start_time).total_seconds()
        }
        
        self.is_trained = True
        
        logger.info(f"Training complete! RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        logger.info(f"Training time: {self.training_metrics['training_time']:.2f} seconds")
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
        
        return self.training_metrics
    
    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for a specific user-movie pair.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating (0.5 to 5.0)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make prediction
        prediction = self.model.predict(user_id, movie_id)
        
        # Return the estimated rating
        return prediction.est
    
    def get_top_n_recommendations(
        self, 
        user_id: int, 
        n: int = 10,
        exclude_rated: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get top N movie recommendations for a user.
        
        Args:
            user_id: User ID
            n: Number of recommendations to return
            exclude_rated: Whether to exclude already rated movies
            
        Returns:
            List of recommended movies with predicted ratings
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info(f"Generating {n} recommendations for user {user_id}")
        
        # Get all movie IDs
        all_movie_ids = self.movies_df['movieId'].unique()
        
        # Get movies already rated by the user (if excluding)
        rated_movies = set()
        if exclude_rated and self.trainset:
            try:
                user_inner_id = self.trainset.to_inner_uid(user_id)
                rated_movies = {
                    self.trainset.to_raw_iid(inner_id) 
                    for inner_id in self.trainset.ur[user_inner_id]
                }
            except ValueError:
                # User not in training set (new user)
                logger.warning(f"User {user_id} not found in training set")
        
        # Generate predictions for all unrated movies
        predictions = []
        for movie_id in all_movie_ids:
            if movie_id not in rated_movies:
                pred = self.predict_rating(user_id, movie_id)
                predictions.append((movie_id, pred))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N
        top_n = predictions[:n]
        
        # Format results with movie metadata
        recommendations = []
        for movie_id, predicted_rating in top_n:
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
            recommendations.append({
                'movieId': int(movie_id),
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'predicted_rating': round(predicted_rating, 2)
            })
        
        return recommendations
    
    def get_similar_movies(
        self, 
        movie_id: int, 
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find similar movies based on latent factors.
        
        Args:
            movie_id: Movie ID to find similar movies for
            n: Number of similar movies to return
            
        Returns:
            List of similar movies with similarity scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            # Get the inner id of the movie
            movie_inner_id = self.trainset.to_inner_iid(movie_id)
            
            # Get movie's latent factors
            movie_factors = self.model.qi[movie_inner_id]
            
            # Calculate similarity with all other movies
            similarities = []
            for other_inner_id, other_factors in enumerate(self.model.qi):
                if other_inner_id != movie_inner_id:
                    # Cosine similarity
                    sim = np.dot(movie_factors, other_factors) / (
                        np.linalg.norm(movie_factors) * np.linalg.norm(other_factors)
                    )
                    other_movie_id = self.trainset.to_raw_iid(other_inner_id)
                    similarities.append((other_movie_id, sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top N
            similar = similarities[:n]
            
            # Format results
            result = []
            for sim_movie_id, similarity in similar:
                movie_info = self.movies_df[self.movies_df['movieId'] == sim_movie_id].iloc[0]
                result.append({
                    'movieId': int(sim_movie_id),
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'similarity_score': round(similarity, 3)
                })
            
            return result
            
        except ValueError:
            logger.error(f"Movie {movie_id} not found in training set")
            return []
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: DataFrame with test ratings
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare test data
        reader = Reader(rating_scale=(0.5, 5.0))
        test_dataset = Dataset.load_from_df(
            test_data[['userId', 'movieId', 'rating']], 
            reader
        )
        
        # Get predictions
        _, testset = train_test_split(test_dataset, test_size=1.0)
        predictions = self.model.test(testset)
        
        # Calculate metrics
        metrics = {
            'rmse': accuracy.rmse(predictions, verbose=False),
            'mae': accuracy.mae(predictions, verbose=False),
            'n_test': len(predictions)
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Nothing to save.")
        
        model_data = {
            'model': self.model,
            'movies_df': self.movies_df,
            'trainset': self.trainset,
            'training_metrics': self.training_metrics,
            'params': {
                'n_factors': self.n_factors,
                'n_epochs': self.n_epochs,
                'lr_all': self.lr_all,
                'reg_all': self.reg_all
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'MovieRecommender':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            MovieRecommender instance with loaded model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance with saved parameters
        recommender = cls(
            n_factors=model_data['params']['n_factors'],
            n_epochs=model_data['params']['n_epochs'],
            lr_all=model_data['params']['lr_all'],
            reg_all=model_data['params']['reg_all']
        )
        
        # Load the trained model and data
        recommender.model = model_data['model']
        recommender.movies_df = model_data['movies_df']
        recommender.trainset = model_data['trainset']
        recommender.training_metrics = model_data['training_metrics']
        recommender.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Model trained on: {model_data['timestamp']}")
        
        return recommender