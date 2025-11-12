"""
Simple Phase 2 Evaluation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy

from src.data.bigquery_loader import BigQueryLoader
from src.data.preprocessing import DataPreprocessor


def evaluate_model(sample_size=20000, verbose=True):
    """
    Simple evaluation function that trains and evaluates the model.
    
    Args:
        sample_size: Number of ratings to load (use 20000 to match hyperparameter tuning)
        verbose: Whether to print progress
    
    Returns:
        Dictionary with all key metrics
    """
    
    if verbose:
        print("="*60)
        print(" SIMPLE MODEL EVALUATION")
        print("="*60)
        print()
    
    # ========================================
    # 1. LOAD DATA
    # ========================================
    if verbose:
        print("Loading data...")
    
    loader = BigQueryLoader(project_id='master-ai-cloud')  # Update with your project
    ratings_df = loader.load_ratings(limit=sample_size)
    movies_df = loader.load_movies()
    
    # ========================================
    # 2. PREPROCESS
    # ========================================
    if verbose:
        print("Preprocessing...")
    
    preprocessor = DataPreprocessor(ratings_df, movies_df)
    ratings_filtered = preprocessor.filter_cold_start_users(min_ratings=5)
    
    if verbose:
        print(f"  • Loaded {len(ratings_df)} ratings")
        print(f"  • After filtering: {len(ratings_filtered)} ratings")
    
    # ========================================
    # 3. CALCULATE COLD START PERCENTAGE
    # ========================================
    user_stats = ratings_df.groupby('userId').size().reset_index(name='num_ratings')
    cold_start_users = user_stats[user_stats['num_ratings'] < 5]
    cold_start_percentage = (len(cold_start_users) / len(user_stats)) * 100
    
    # ========================================
    # 4. LOAD OPTIMAL PARAMETERS
    # ========================================
    try:
        # Try to load optimal parameters if they exist
        with open('models/optimal_hyperparameters.json', 'r') as f:
            optimal_params = json.load(f)
            
        model_params = {
            'n_factors': optimal_params['n_factors'],
            'n_epochs': optimal_params['n_epochs'],
            'lr_all': optimal_params['lr_all'],
            'reg_all': optimal_params['reg_all']
        }
    except:
        # Use default if file doesn't exist
        model_params = {
            'n_factors': 100,
            'n_epochs': 20,
            'lr_all': 0.006,
            'reg_all': 0.025
        }
    
    if verbose:
        print(f"\nModel Parameters:")
        for key, value in model_params.items():
            print(f"  • {key}: {value}")
    
    # ========================================
    # 5. PREPARE DATA FOR SURPRISE
    # ========================================
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(
        ratings_filtered[['userId', 'movieId', 'rating']], 
        reader
    )
    
    # ========================================
    # 6. TRAIN WITH 80/20 SPLIT
    # ========================================
    if verbose:
        print(f"\nTraining model with 80/20 split...")
    
    # Create train/test split
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # Initialize and train model
    model = SVD(
        n_factors=model_params['n_factors'],
        n_epochs=model_params['n_epochs'],
        lr_all=model_params['lr_all'],
        reg_all=model_params['reg_all'],
        random_state=42,
        verbose=False
    )
    
    # Measure training time
    start_time = time.time()
    model.fit(trainset)
    training_time = time.time() - start_time
    
    # ========================================
    # 7. EVALUATE
    # ========================================
    if verbose:
        print("Evaluating...")
    
    # Get predictions
    predictions = model.test(testset)
    
    # Calculate metrics
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    
    # ========================================
    # 8. CROSS-VALIDATION (Optional - more robust)
    # ========================================
    if verbose:
        print("Running 5-fold cross-validation...")
    
    cv_model = SVD(**model_params, random_state=42)
    cv_results = cross_validate(
        cv_model, 
        data, 
        measures=['RMSE', 'MAE'],
        cv=5,
        verbose=False
    )
    
    cv_rmse = np.mean(cv_results['test_rmse'])
    cv_mae = np.mean(cv_results['test_mae'])
    
    # ========================================
    # 9. COMPILE RESULTS
    # ========================================
    results = {
        'model_training_time': training_time,
        'test_rmse': rmse,
        'test_mae': mae,
        'cold_start_percentage': cold_start_percentage,
        'cv_rmse': cv_rmse,
        'cv_mae': cv_mae,
        'dataset_size': len(ratings_filtered),
        'train_size': trainset.n_ratings,
        'test_size': len(testset),
        'total_users': len(user_stats),
        'cold_start_users': len(cold_start_users)
    }
    
    # ========================================
    # 10. PRINT RESULTS
    # ========================================
    if verbose:
        print("\n" + "="*60)
        print(" RESULTS")
        print("="*60)
        print()
        print("## Key Results")
        print(f"- Model training time: {results['model_training_time']:.2f} seconds")
        print(f"- Test set RMSE: {results['test_rmse']:.4f}")
        print(f"- Test set MAE: {results['test_mae']:.4f}")
        print(f"- Cold start users handled: {results['cold_start_percentage']:.1f}% of dataset")
        print()
        print("## Cross-Validation Results (5-fold)")
        print(f"- CV RMSE: {results['cv_rmse']:.4f}")
        print(f"- CV MAE: {results['cv_mae']:.4f}")
        print()
        print("## Dataset Info")
        print(f"- Total ratings: {results['dataset_size']}")
        print(f"- Train size: {results['train_size']}")
        print(f"- Test size: {results['test_size']}")
        print(f"- Total users: {results['total_users']}")
        print(f"- Cold start users: {results['cold_start_users']}")
        print()
        print("="*60)
    
    return results


def quick_test():
    """
    Even simpler version - just the 4 key metrics
    """
    print("\nQuick Test - Getting your 4 key metrics...\n")
    
    # Load data
    loader = BigQueryLoader(project_id='students-groupX')
    ratings = loader.load_ratings(limit=20000)
    movies = loader.load_movies()
    
    # Preprocess
    preprocessor = DataPreprocessor(ratings, movies)
    ratings_filtered = preprocessor.filter_cold_start_users(min_ratings=5)
    
    # Cold start percentage
    user_stats = ratings.groupby('userId').size().reset_index(name='num_ratings')
    cold_start_pct = (sum(user_stats['num_ratings'] < 5) / len(user_stats)) * 100
    
    # Prepare data
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_filtered[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # Train model (with your optimal parameters)
    model = SVD(n_factors=100, n_epochs=20, lr_all=0.006, reg_all=0.025, random_state=42)
    
    start = time.time()
    model.fit(trainset)
    train_time = time.time() - start
    
    # Evaluate
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    
    # Print just the 4 metrics you need
    print("## Key Results")
    print(f"- Model training time: {train_time:.2f} seconds")
    print(f"- Test set RMSE: {rmse:.4f}")
    print(f"- Test set MAE: {mae:.4f}")
    print(f"- Cold start users handled: {cold_start_pct:.1f}% of dataset")
    
    return {
        'model_training_time': train_time,
        'test_rmse': rmse,
        'test_mae': mae,
        'cold_start_percentage': cold_start_pct
    }


if __name__ == "__main__":
    # Choose which version to run:
    
    # Option 1: Full evaluation with detailed output
    results = evaluate_model(sample_size=20000, verbose=True)
    
    # Option 2: Quick test - just the 4 metrics
    # results = quick_test()