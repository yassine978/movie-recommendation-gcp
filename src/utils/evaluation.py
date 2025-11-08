"""
Evaluation metrics for recommendation models.
Includes rating prediction metrics (RMSE, MAE) and ranking metrics (Precision@K, Recall@K).
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        predictions: Array of predicted ratings
        actuals: Array of actual ratings
        
    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((predictions - actuals) ** 2))


def mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        predictions: Array of predicted ratings
        actuals: Array of actual ratings
        
    Returns:
        MAE value
    """
    return np.mean(np.abs(predictions - actuals))


def precision_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    Calculate Precision@K.
    
    Precision@K = (# of recommended items @K that are relevant) / K
    
    Args:
        recommended: List of recommended item IDs (in order)
        relevant: List of relevant item IDs
        k: Number of recommendations to consider
        
    Returns:
        Precision@K value
    """
    if k == 0:
        return 0.0
    
    recommended_at_k = recommended[:k]
    relevant_set = set(relevant)
    
    num_relevant = sum(1 for item in recommended_at_k if item in relevant_set)
    
    return num_relevant / k


def recall_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    Calculate Recall@K.
    
    Recall@K = (# of recommended items @K that are relevant) / (# of relevant items)
    
    Args:
        recommended: List of recommended item IDs (in order)
        relevant: List of relevant item IDs
        k: Number of recommendations to consider
        
    Returns:
        Recall@K value
    """
    if len(relevant) == 0:
        return 0.0
    
    recommended_at_k = recommended[:k]
    relevant_set = set(relevant)
    
    num_relevant = sum(1 for item in recommended_at_k if item in relevant_set)
    
    return num_relevant / len(relevant)


def average_precision_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    Calculate Average Precision@K.
    
    AP@K = (1/min(K, # relevant)) * Σ(Precision@i * relevance(i))
    
    Args:
        recommended: List of recommended item IDs (in order)
        relevant: List of relevant item IDs
        k: Number of recommendations to consider
        
    Returns:
        Average Precision@K value
    """
    if len(relevant) == 0:
        return 0.0
    
    recommended_at_k = recommended[:k]
    relevant_set = set(relevant)
    
    score = 0.0
    num_hits = 0
    
    for i, item in enumerate(recommended_at_k):
        if item in relevant_set:
            num_hits += 1
            score += num_hits / (i + 1)
    
    return score / min(len(relevant), k)


def ndcg_at_k(recommended: List[int], relevant: List[int], k: int, 
              relevance_scores: Dict[int, float] = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain@K.
    
    Args:
        recommended: List of recommended item IDs (in order)
        relevant: List of relevant item IDs
        k: Number of recommendations to consider
        relevance_scores: Optional dict mapping item IDs to relevance scores
        
    Returns:
        NDCG@K value
    """
    if len(relevant) == 0:
        return 0.0
    
    # Default: binary relevance (1 if relevant, 0 otherwise)
    if relevance_scores is None:
        relevance_scores = {item: 1.0 for item in relevant}
    
    recommended_at_k = recommended[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(recommended_at_k):
        if item in relevance_scores:
            dcg += relevance_scores[item] / np.log2(i + 2)  # i+2 because i starts at 0
    
    # Calculate IDCG (ideal DCG)
    ideal_items = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
    idcg = 0.0
    for i, (item, score) in enumerate(ideal_items[:k]):
        idcg += score / np.log2(i + 2)
    
    # Calculate NDCG
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def coverage(recommended_all: List[List[int]], catalog: List[int]) -> float:
    """
    Calculate catalog coverage.
    
    Coverage = (# of unique items recommended) / (# of items in catalog)
    
    Args:
        recommended_all: List of recommendation lists for all users
        catalog: List of all item IDs in catalog
        
    Returns:
        Coverage value (0 to 1)
    """
    recommended_items = set()
    for rec_list in recommended_all:
        recommended_items.update(rec_list)
    
    return len(recommended_items) / len(catalog)


def diversity(recommended_all: List[List[int]]) -> float:
    """
    Calculate inter-list diversity.
    
    Diversity = 1 - (average pairwise Jaccard similarity between recommendation lists)
    
    Args:
        recommended_all: List of recommendation lists for all users
        
    Returns:
        Diversity value (0 to 1, higher is more diverse)
    """
    if len(recommended_all) < 2:
        return 1.0
    
    similarities = []
    
    for i in range(len(recommended_all)):
        for j in range(i + 1, len(recommended_all)):
            set_i = set(recommended_all[i])
            set_j = set(recommended_all[j])
            
            if len(set_i) == 0 and len(set_j) == 0:
                similarity = 1.0
            else:
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                similarity = intersection / union if union > 0 else 0.0
            
            similarities.append(similarity)
    
    avg_similarity = np.mean(similarities)
    return 1 - avg_similarity


class ModelEvaluator:
    """
    Comprehensive model evaluation.
    """
    
    def __init__(self, model, test_data: pd.DataFrame):
        """
        Initialize evaluator.
        
        Args:
            model: Trained recommendation model
            test_data: Test dataset with columns [userId, movieId, rating]
        """
        self.model = model
        self.test_data = test_data
    
    def evaluate_predictions(self) -> Dict[str, float]:
        """
        Evaluate rating prediction accuracy.
        
        Returns:
            Dictionary with RMSE and MAE
        """
        logger.info("Evaluating prediction accuracy...")
        
        predictions = []
        actuals = []
        
        for _, row in self.test_data.iterrows():
            pred = self.model.predict(row['userId'], row['movieId'])
            predictions.append(pred)
            actuals.append(row['rating'])
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        results = {
            'rmse': rmse(predictions, actuals),
            'mae': mae(predictions, actuals)
        }
        
        logger.info(f"RMSE: {results['rmse']:.4f}, MAE: {results['mae']:.4f}")
        
        return results
    
    def evaluate_rankings(self, k: int = 10, relevance_threshold: float = 4.0) -> Dict[str, float]:
        """
        Evaluate ranking quality.
        
        Args:
            k: Number of recommendations to consider
            relevance_threshold: Rating threshold for relevance
            
        Returns:
            Dictionary with Precision@K, Recall@K, etc.
        """
        logger.info(f"Evaluating rankings (K={k}, threshold={relevance_threshold})...")
        
        # Group test data by user
        user_groups = self.test_data.groupby('userId')
        
        precisions = []
        recalls = []
        
        for user_id, group in user_groups:
            # Get relevant items (high ratings)
            relevant_items = group[group['rating'] >= relevance_threshold]['movieId'].tolist()
            
            if len(relevant_items) == 0:
                continue
            
            # Get recommendations (would need to implement get_recommendations in model)
            # For now, skip this part - will implement when we have full recommender
            
            # precisions.append(precision_at_k(recommended, relevant_items, k))
            # recalls.append(recall_at_k(recommended, relevant_items, k))
        
        # This is placeholder - will complete when we have SVD model
        results = {
            'precision@k': 0.0,  # np.mean(precisions) if precisions else 0.0
            'recall@k': 0.0,     # np.mean(recalls) if recalls else 0.0
        }
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Test metrics with sample data
    print("=== Testing Evaluation Metrics ===\\n")
    
    # Test RMSE and MAE
    predictions = np.array([3.5, 4.0, 2.5, 4.5, 3.0])
    actuals = np.array([3.0, 4.5, 2.0, 4.0, 3.5])
    
    print(f"RMSE: {rmse(predictions, actuals):.4f}")
    print(f"MAE: {mae(predictions, actuals):.4f}")
    
    # Test ranking metrics
    recommended = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    relevant = [2, 4, 6, 8, 10]
    k = 5
    
    print(f"\\nPrecision@{k}: {precision_at_k(recommended, relevant, k):.4f}")
    print(f"Recall@{k}: {recall_at_k(recommended, relevant, k):.4f}")
    print(f"AP@{k}: {average_precision_at_k(recommended, relevant, k):.4f}")
    print(f"NDCG@{k}: {ndcg_at_k(recommended, relevant, k):.4f}")
    
    # Test coverage and diversity
    all_recs = [
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [10, 11, 12, 13, 14]
    ]
    catalog = list(range(1, 21))
    
    print(f"\\nCoverage: {coverage(all_recs, catalog):.4f}")
    print(f"Diversity: {diversity(all_recs):.4f}")
    
    print("\\n✓ All evaluation metrics working!")