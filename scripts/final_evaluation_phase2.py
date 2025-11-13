"""
Final Model Evaluation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

from src.data.bigquery_loader import BigQueryLoader
from src.data.preprocessing import DataPreprocessor


def get_final_metrics():
    """
    Get final metrics on full dataset for Phase 2 summary
    """
    
    print("="*70)
    print(" FINAL MODEL EVALUATION - FULL DATASET")
    print("="*70)
    print()
    
    # ========================================
    # 1. LOAD FULL DATA
    # ========================================
    print("📊 Loading FULL dataset from BigQuery...")
    
    loader = BigQueryLoader(project_id='master-ai-cloud')
    ratings_df = loader.load_ratings()  # No limit = full dataset
    movies_df = loader.load_movies()
    
    print(f"✓ Loaded {len(ratings_df):,} ratings")
    print(f"✓ Loaded {len(movies_df):,} movies")
    print()
    
    # ========================================
    # 2. PREPROCESS
    # ========================================
    print("🔧 Preprocessing data...")
    
    preprocessor = DataPreprocessor(ratings_df, movies_df)
    ratings_filtered = preprocessor.filter_cold_start_users(min_ratings=5)
    
    print(f"✓ After filtering: {len(ratings_filtered):,} ratings")
    print()
    
    # ========================================
    # 3. CALCULATE COLD START PERCENTAGE
    # ========================================
    user_stats = ratings_df.groupby('userId').size().reset_index(name='num_ratings')
    cold_start_users = user_stats[user_stats['num_ratings'] < 5]
    cold_start_percentage = (len(cold_start_users) / len(user_stats)) * 100
    
    print(f"👥 Total users: {len(user_stats):,}")
    print(f"👥 Cold start users (<5 ratings): {len(cold_start_users):,}")
    print(f"📊 Cold start percentage: {cold_start_percentage:.1f}%")
    print()
    
    # ========================================
    # 4. LOAD OPTIMAL PARAMETERS
    # ========================================
    print("⚙️  Loading optimal hyperparameters...")
    
    try:
        with open('models/optimal_hyperparameters.json', 'r') as f:
            all_params = json.load(f)
        
        # Extract only valid SVD parameters
        model_params = {
            'n_factors': all_params.get('n_factors', 100),
            'n_epochs': all_params.get('n_epochs', 20),
            'lr_all': all_params.get('lr_all', 0.006),
            'reg_all': all_params.get('reg_all', 0.025),
            'random_state': 42
        }
        print("✓ Using optimal parameters:")
    except:
        model_params = {
            'n_factors': 100,
            'n_epochs': 20,
            'lr_all': 0.006,
            'reg_all': 0.025,
            'random_state': 42
        }
        print("⚠️  Using default parameters:")
    
    for key, value in model_params.items():
        if key != 'random_state':
            print(f"   • {key}: {value}")
    print()
    
    # ========================================
    # 5. PREPARE DATA FOR SURPRISE
    # ========================================
    print("📦 Preparing data for training...")
    
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(
        ratings_filtered[['userId', 'movieId', 'rating']], 
        reader
    )
    
    # Create 80/20 train/test split
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    print(f"✓ Training set: {trainset.n_ratings:,} ratings")
    print(f"✓ Test set: {len(testset):,} ratings")
    print()
    
    # ========================================
    # 6. TRAIN MODEL
    # ========================================
    print("🚀 Training SVD model on FULL dataset...")
    print("   (This may take 30-60 seconds...)")
    print()
    
    model = SVD(**model_params, verbose=False)
    
    start_time = time.time()
    model.fit(trainset)
    training_time = time.time() - start_time
    
    print(f"✓ Training completed in {training_time:.2f} seconds")
    print()
    
    # ========================================
    # 7. EVALUATE ON TEST SET
    # ========================================
    print("📈 Evaluating on test set...")
    
    predictions = model.test(testset)
    
    test_rmse = accuracy.rmse(predictions, verbose=False)
    test_mae = accuracy.mae(predictions, verbose=False)
    
    print(f"✓ Test RMSE: {test_rmse:.4f}")
    print(f"✓ Test MAE: {test_mae:.4f}")
    print()
    
    # ========================================
    # 8. COMPILE FINAL RESULTS
    # ========================================
    results = {
        'dataset': {
            'total_ratings': len(ratings_df),
            'filtered_ratings': len(ratings_filtered),
            'total_users': len(user_stats),
            'cold_start_users': len(cold_start_users),
            'cold_start_percentage': cold_start_percentage,
            'num_movies': len(movies_df)
        },
        'training': {
            'train_size': trainset.n_ratings,
            'test_size': len(testset),
            'training_time_seconds': training_time
        },
        'hyperparameters': model_params,
        'performance': {
            'test_rmse': test_rmse,
            'test_mae': test_mae
        }
    }
    
    # ========================================
    # 9. SAVE RESULTS
    # ========================================
    results_file = 'models/final_evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {results_file}")
    print()
    
    # ========================================
    # 10. DISPLAY FORMATTED SUMMARY
    # ========================================
    print("="*70)
    print(" FINAL RESULTS FOR PHASE 2 SUMMARY")
    print("="*70)
    print()
    print("## Key Metrics (Use these in phase2_summary.md):")
    print()
    print(f"- Model training time: {training_time:.2f} seconds")
    print(f"- Test set RMSE: {test_rmse:.4f}")
    print(f"- Test set MAE: {test_mae:.4f}")
    print(f"- Cold start users handled: {cold_start_percentage:.1f}% of dataset")
    print()
    print("## Dataset Statistics:")
    print()
    print(f"- Total ratings: {len(ratings_df):,}")
    print(f"- After filtering: {len(ratings_filtered):,}")
    print(f"- Training samples: {trainset.n_ratings:,}")
    print(f"- Test samples: {len(testset):,}")
    print(f"- Total users: {len(user_stats):,}")
    print(f"- Cold start users: {len(cold_start_users):,}")
    print()
    print("## Optimal Hyperparameters:")
    print()
    print(f"- n_factors: {model_params['n_factors']}")
    print(f"- n_epochs: {model_params['n_epochs']}")
    print(f"- lr_all: {model_params['lr_all']}")
    print(f"- reg_all: {model_params['reg_all']}")
    print()
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = get_final_metrics()
    
    print("\n✅ Evaluation complete!")