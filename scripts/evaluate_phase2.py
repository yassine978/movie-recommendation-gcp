"""
Phase 2 Evaluation Script
Runs complete model evaluation and generates all key metrics for documentation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Import your modules
from src.data.bigquery_loader import BigQueryLoader
from src.data.preprocessing import DataPreprocessor
from src.models.recommender import MovieRecommender
from src.models.cold_start import ColdStartHandler

def run_complete_evaluation():
    """
    Run complete evaluation of Phase 2 models and generate metrics.
    """
    
    print("="*60)
    print(" PHASE 2: COMPLETE MODEL EVALUATION")
    print("="*60)
    print()
    
    results = {}
    
    # ========================================
    # STEP 1: LOAD DATA
    # ========================================
    print("Step 1: Loading data from BigQuery...")
    print("-"*40)
    
    loader = BigQueryLoader(project_id='master-ai-cloud')  # Update with your project
    
    # Load full dataset for final evaluation
    data_load_start = time.time()
    ratings_df = loader.load_ratings()  # Full dataset
    movies_df = loader.load_movies()
    data_load_time = time.time() - data_load_start
    
    print(f"✓ Loaded {len(ratings_df)} ratings in {data_load_time:.2f} seconds")
    print(f"✓ Loaded {len(movies_df)} movies")
    print(f"✓ Unique users: {ratings_df['userId'].nunique()}")
    print(f"✓ Unique movies: {ratings_df['movieId'].nunique()}")
    print()
    
    # ========================================
    # STEP 2: ANALYZE COLD START USERS
    # ========================================
    print("Step 2: Analyzing Cold Start Distribution...")
    print("-"*40)
    
    # Calculate user statistics
    user_stats = ratings_df.groupby('userId').size().reset_index(name='num_ratings')
    
    # Categorize users
    cold_start_users = user_stats[user_stats['num_ratings'] < 5]
    moderate_users = user_stats[(user_stats['num_ratings'] >= 5) & (user_stats['num_ratings'] < 20)]
    active_users = user_stats[user_stats['num_ratings'] >= 20]
    
    total_users = len(user_stats)
    cold_start_percentage = (len(cold_start_users) / total_users) * 100
    
    print(f"User Distribution:")
    print(f"  • Cold start (<5 ratings): {len(cold_start_users)} users ({cold_start_percentage:.1f}%)")
    print(f"  • Moderate (5-19 ratings): {len(moderate_users)} users ({len(moderate_users)/total_users*100:.1f}%)")
    print(f"  • Active (20+ ratings): {len(active_users)} users ({len(active_users)/total_users*100:.1f}%)")
    print()
    
    results['cold_start_percentage'] = cold_start_percentage
    results['cold_start_users'] = len(cold_start_users)
    results['total_users'] = total_users
    
    # ========================================
    # STEP 3: PREPROCESS DATA
    # ========================================
    print("Step 3: Preprocessing data...")
    print("-"*40)
    
    preprocessor = DataPreprocessor(ratings_df, movies_df)
    
    # Filter cold start users for training (optional)
    ratings_filtered = preprocessor.filter_cold_start_users(min_ratings=5)
    
    print(f"✓ Original ratings: {len(ratings_df)}")
    print(f"✓ After filtering cold start: {len(ratings_filtered)}")
    print(f"✓ Ratings removed: {len(ratings_df) - len(ratings_filtered)} ({(len(ratings_df) - len(ratings_filtered))/len(ratings_df)*100:.1f}%)")
    print()
    
    # ========================================
    # STEP 4: LOAD OPTIMAL HYPERPARAMETERS
    # ========================================
    print("Step 4: Loading optimal hyperparameters...")
    print("-"*40)
    
    # Check if hyperparameter file exists
    hyperparam_path = 'models/optimal_hyperparameters.json'
    
    if os.path.exists(hyperparam_path):
        with open(hyperparam_path, 'r') as f:
            optimal_params = json.load(f)
        
        model_params = {
            'n_factors': optimal_params['n_factors'],
            'n_epochs': optimal_params['n_epochs'],
            'lr_all': optimal_params['lr_all'],
            'reg_all': optimal_params['reg_all'],
            'random_state': 42,
            'verbose': False
        }
        print("✓ Loaded optimal hyperparameters from file")
    else:
        # Default parameters if file doesn't exist
        model_params = {
            'n_factors': 100,
            'n_epochs': 20,
            'lr_all': 0.005,
            'reg_all': 0.02,
            'random_state': 42,
            'verbose': False
        }
        print("⚠ Using default hyperparameters (optimal file not found)")
    
    print(f"  • n_factors: {model_params['n_factors']}")
    print(f"  • n_epochs: {model_params['n_epochs']}")
    print(f"  • lr_all: {model_params['lr_all']}")
    print(f"  • reg_all: {model_params['reg_all']}")
    print()
    
    # ========================================
    # STEP 5: TRAIN SVD MODEL
    # ========================================
    print("Step 5: Training SVD Model...")
    print("-"*40)
    
    # Initialize recommender
    recommender = MovieRecommender(**model_params)
    
    # Train the model and measure time
    train_start = time.time()
    
    training_metrics = recommender.train(
        ratings_df=ratings_filtered,
        movies_df=movies_df,
        test_size=0.2  # 80/20 split
    )
    
    training_time = time.time() - train_start
    
    print(f"✓ Model training completed!")
    print(f"  • Training time: {training_time:.2f} seconds")
    print(f"  • Training samples: {training_metrics['n_train']}")
    print(f"  • Test samples: {training_metrics['n_test']}")
    print()
    
    results['model_training_time'] = training_time
    results['training_samples'] = training_metrics['n_train']
    results['test_samples'] = training_metrics['n_test']
    
    # ========================================
    # STEP 6: EVALUATE MODEL PERFORMANCE
    # ========================================
    print("Step 6: Model Performance Metrics...")
    print("-"*40)
    
    # Get metrics from training
    test_rmse = training_metrics['rmse']
    test_mae = training_metrics['mae']
    
    print(f"Test Set Performance:")
    print(f"  • RMSE: {test_rmse:.4f}")
    print(f"  • MAE:  {test_mae:.4f}")
    print()
    
    results['test_rmse'] = test_rmse
    results['test_mae'] = test_mae
    
    # Calculate baseline for comparison
    print("Calculating baseline metrics...")
    global_mean = ratings_filtered['rating'].mean()
    
    # Create test predictions with global mean
    from surprise import Dataset, Reader
    from surprise.model_selection import train_test_split
    
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_filtered[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # Calculate baseline RMSE (predict global mean for everything)
    baseline_predictions = []
    for uid, iid, true_r in testset:
        baseline_predictions.append((true_r, global_mean))
    
    baseline_rmse = np.sqrt(np.mean([(true - pred)**2 for true, pred in baseline_predictions]))
    baseline_mae = np.mean([abs(true - pred) for true, pred in baseline_predictions])
    
    print(f"Baseline Performance (Global Mean):")
    print(f"  • RMSE: {baseline_rmse:.4f}")
    print(f"  • MAE:  {baseline_mae:.4f}")
    print()
    
    improvement_rmse = ((baseline_rmse - test_rmse) / baseline_rmse) * 100
    improvement_mae = ((baseline_mae - test_mae) / baseline_mae) * 100
    
    print(f"Improvement over baseline:")
    print(f"  • RMSE improvement: {improvement_rmse:.1f}%")
    print(f"  • MAE improvement:  {improvement_mae:.1f}%")
    print()
    
    results['baseline_rmse'] = baseline_rmse
    results['baseline_mae'] = baseline_mae
    results['rmse_improvement'] = improvement_rmse
    results['mae_improvement'] = improvement_mae
    
    # ========================================
    # STEP 7: TEST PREDICTIONS
    # ========================================
    print("Step 7: Testing Predictions...")
    print("-"*40)
    
    # Test prediction speed
    test_users = [1, 2, 3, 4, 5]  # Sample users
    
    pred_start = time.time()
    for user_id in test_users:
        recs = recommender.get_top_n_recommendations(user_id=user_id, n=10)
    pred_time = (time.time() - pred_start) / len(test_users)
    
    print(f"✓ Average prediction time per user: {pred_time*1000:.2f} ms")
    print()
    
    results['avg_prediction_time_ms'] = pred_time * 1000
    
    # ========================================
    # STEP 8: TEST COLD START HANDLER
    # ========================================
    print("Step 8: Testing Cold Start Handler...")
    print("-"*40)
    
    cold_handler = ColdStartHandler(ratings_df, movies_df)
    
    # Test popular recommendations
    popular_recs = cold_handler.get_popular_recommendations(n=5)
    
    print("Top 5 Popular Movies (for new users):")
    for i, movie in enumerate(popular_recs, 1):
        print(f"  {i}. {movie['title'][:50]}... (score: {movie['popularity_score']:.2f})")
    print()
    
    # Test hybrid recommendations for user with few ratings
    if len(cold_start_users) > 0:
        sample_cold_user = cold_start_users.iloc[0]['userId']
        cold_user_recs = cold_handler.get_recommendations_for_user(sample_cold_user, n=5)
        print(f"Recommendations for cold start user {sample_cold_user}:")
        print(f"  • Strategy: {cold_user_recs['recommendation_type']}")
        print(f"  • User has {cold_user_recs['num_ratings']} ratings")
    print()
    
    # ========================================
    # STEP 9: SAVE MODEL
    # ========================================
    print("Step 9: Saving trained model...")
    print("-"*40)
    
    # Create models directory if it doesn't exist
    Path('models').mkdir(exist_ok=True)
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/recommender_final_{timestamp}.pkl"
    recommender.save_model(model_path)
    
    print(f"✓ Model saved to: {model_path}")
    print()
    
    results['model_path'] = model_path
    
    # ========================================
    # STEP 10: GENERATE SUMMARY
    # ========================================
    print("="*60)
    print(" EVALUATION SUMMARY")
    print("="*60)
    print()
    
    print("## Key Results")
    print(f"- Model training time: {results['model_training_time']:.2f} seconds")
    print(f"- Test set RMSE: {results['test_rmse']:.4f}")
    print(f"- Test set MAE: {results['test_mae']:.4f}")
    print(f"- Cold start users handled: {results['cold_start_percentage']:.1f}% of dataset")
    print()
    
    print("## Additional Metrics")
    print(f"- Total users: {results['total_users']}")
    print(f"- Cold start users: {results['cold_start_users']}")
    print(f"- Training samples: {results['training_samples']}")
    print(f"- Test samples: {results['test_samples']}")
    print(f"- Baseline RMSE: {results['baseline_rmse']:.4f}")
    print(f"- Baseline MAE: {results['baseline_mae']:.4f}")
    print(f"- RMSE improvement: {results['rmse_improvement']:.1f}%")
    print(f"- MAE improvement: {results['mae_improvement']:.1f}%")
    print(f"- Avg prediction time: {results['avg_prediction_time_ms']:.2f} ms")
    print()
    
    # ========================================
    # SAVE RESULTS TO FILE
    # ========================================
    results_path = f"models/evaluation_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {results_path}")
    print()
    
    # ========================================
    # GENERATE MARKDOWN REPORT
    # ========================================
    markdown_report = f"""# Phase 2 Evaluation Results

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Key Results
- **Model training time**: {results['model_training_time']:.2f} seconds
- **Test set RMSE**: {results['test_rmse']:.4f}
- **Test set MAE**: {results['test_mae']:.4f}
- **Cold start users handled**: {results['cold_start_percentage']:.1f}% of dataset

## Model Performance
| Metric | SVD Model | Baseline | Improvement |
|--------|-----------|----------|-------------|
| RMSE | {results['test_rmse']:.4f} | {results['baseline_rmse']:.4f} | {results['rmse_improvement']:.1f}% |
| MAE | {results['test_mae']:.4f} | {results['baseline_mae']:.4f} | {results['mae_improvement']:.1f}% |

## Dataset Statistics
- Total ratings: {len(ratings_df):,}
- Total users: {results['total_users']:,}
- Total movies: {len(movies_df):,}
- Cold start users (<5 ratings): {results['cold_start_users']} ({results['cold_start_percentage']:.1f}%)

## Model Configuration
- Algorithm: SVD (Singular Value Decomposition)
- Factors: {model_params['n_factors']}
- Epochs: {model_params['n_epochs']}
- Learning rate: {model_params['lr_all']}
- Regularization: {model_params['reg_all']}

## Performance Metrics
- Training samples: {results['training_samples']:,}
- Test samples: {results['test_samples']:,}
- Average prediction time: {results['avg_prediction_time_ms']:.2f} ms per user

## Model Files
- Trained model: `{results['model_path']}`
- Results JSON: `{results_path}`
"""
    
    # Save markdown report
    report_path = f"docs/phase2_evaluation_{timestamp}.md"
    Path('docs').mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(markdown_report)
    
    print(f"Markdown report saved to: {report_path}")
    print()
    
    print("="*60)
    print(" EVALUATION COMPLETE!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    # Run the complete evaluation
    results = run_complete_evaluation()