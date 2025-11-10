"""
Training script for the movie recommender model.
Run this script to train the SVD model with optimized hyperparameters.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from data.bigquery_loader import BigQueryLoader
from data.preprocessing import DataPreprocessor
from models.recommender import MovieRecommender

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(sample_size: int = None, save_dir: str = 'models'):
    """
    Train the movie recommender model.
    
    Args:
        sample_size: Optional sample size for quick testing
        save_dir: Directory to save the trained model
    """
    logger.info("="*50)
    logger.info("Starting Movie Recommender Training")
    logger.info("="*50)
    
    # Step 1: Load data from BigQuery
    logger.info("Step 1: Loading data from BigQuery...")
    loader = BigQueryLoader(project_id='students-groupX')  # Replace with your project
    
    if sample_size:
        logger.info(f"Loading sample of {sample_size} ratings for testing...")
        ratings_df = loader.load_ratings(limit=sample_size)
    else:
        logger.info("Loading full dataset...")
        ratings_df = loader.load_ratings()
    
    movies_df = loader.load_movies()
    
    logger.info(f"Loaded {len(ratings_df)} ratings and {len(movies_df)} movies")
    
    # Step 2: Preprocess data
    logger.info("Step 2: Preprocessing data...")
    preprocessor = DataPreprocessor(ratings_df, movies_df)
    
    # Filter cold start users (optional, since we only have 3.6%)
    ratings_filtered = preprocessor.filter_cold_start_users(min_ratings=5)
    logger.info(f"Filtered dataset: {len(ratings_filtered)} ratings")
    
    # Step 3: Train model with optimal parameters
    logger.info("Step 3: Training SVD model...")
    
    # These are the optimal parameters from hyperparameter tuning
    # You'll update these after running hyperparameter tuning
    optimal_params = {
        'n_factors': 100,
        'n_epochs': 20,
        'lr_all': 0.005,
        'reg_all': 0.02,
        'random_state': 42,
        'verbose': True
    }
    
    recommender = MovieRecommender(**optimal_params)
    
    # Train the model
    metrics = recommender.train(
        ratings_df=ratings_filtered,
        movies_df=movies_df,
        test_size=0.2
    )
    
    # Step 4: Display training results
    logger.info("="*50)
    logger.info("Training Complete!")
    logger.info(f"RMSE: {metrics['rmse']:.4f}")
    logger.info(f"MAE: {metrics['mae']:.4f}")
    logger.info(f"Training samples: {metrics['n_train']}")
    logger.info(f"Test samples: {metrics['n_test']}")
    logger.info(f"Training time: {metrics['training_time']:.2f} seconds")
    logger.info("="*50)
    
    # Step 5: Save the model
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if sample_size:
        model_path = f"{save_dir}/recommender_sample_{sample_size}.pkl"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{save_dir}/recommender_v2_{timestamp}.pkl"
    
    recommender.save_model(model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Save training metrics
    metrics_path = model_path.replace('.pkl', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")
    
    # Step 6: Test recommendations for a sample user
    logger.info("\nTesting recommendations for user 1...")
    recommendations = recommender.get_top_n_recommendations(user_id=1, n=5)
    
    logger.info("Top 5 recommendations:")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"{i}. {rec['title']} (predicted: {rec['predicted_rating']})")
    
    return recommender, metrics


if __name__ == "__main__":
    # Train on sample first for testing
    logger.info("Training on sample data for testing...")
    model, metrics = train_model(sample_size=5000)
    
    # Uncomment below to train on full dataset
    # logger.info("\nTraining on full dataset...")
    # model, metrics = train_model(sample_size=None)