# Model Selection Rationale

## Problem Statement

Build a recommendation system that:
1. Handles new users (cold start)
2. Provides personalized recommendations
3. Adapts as users provide more ratings
4. Deploys easily on cloud infrastructure

## Model Options Considered

### 1. Collaborative Filtering (Matrix Factorization)

**Algorithms**: SVD, NMF, ALS

**Pros**:
- Proven approach for sparse rating data
- Captures latent factors (user preferences, movie characteristics)
- Fast training and inference
- Easy to deploy

**Cons**:
- Cannot handle completely new users
- Requires cold-start strategy
- Limited to user-item interactions

**Decision**: ✓ **Selected - SVD variant**

**Why SVD?**
- Industry standard (used by Netflix Prize winners)
- scikit-surprise library provides efficient implementation
- Handles 100K+ ratings in seconds
- Model size < 50 MB (easy to deploy)
- Good balance of accuracy and simplicity

### 2. Content-Based Filtering

**Algorithms**: Cosine similarity on genre/metadata features

**Pros**:
- Works for new users
- Explainable recommendations
- No cold-start problem for items

**Cons**:
- Limited by feature quality
- Cannot discover unexpected preferences
- Serendipity problem (only similar items)

**Decision**: ✗ Not selected as primary model

**Usage**: As fallback for cold-start scenarios

### 3. Deep Learning (Neural Collaborative Filtering)

**Algorithms**: NCF, AutoEncoders, Transformers

**Pros**:
- Can capture complex patterns
- State-of-the-art on some benchmarks
- Flexible architecture

**Cons**:
- Requires more data (we have 100K ratings, need millions)
- Longer training time
- Larger model size (harder to deploy)
- More complex to debug and maintain
- Overkill for this problem size

**Decision**: ✗ Not selected

**Reason**: Dataset too small, deployment complexity not justified

### 4. Baseline Models

**Algorithms**: Global average, per-movie average, popularity

**Pros**:
- Simple to implement
- Fast inference
- Good starting point

**Cons**:
- Not personalized
- Limited accuracy

**Decision**: ✓ Implemented as baselines and cold-start fallbacks

## Final Architecture
```
┌─────────────────────────────────────────┐
│  User has 0 ratings?                    │
│  → Popularity-Based (Bayesian average)  │
└─────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  User has 1-4 ratings?                  │
│  → Genre-Based (from rated movies)      │
└─────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  User has 5+ ratings?                   │
│  → SVD Collaborative Filtering          │
└─────────────────────────────────────────┘
```

## SVD Configuration

**Library**: scikit-surprise

**Algorithm**: SVD (Singular Value Decomposition)

**Hyperparameters** (to be tuned):
- n_factors: 50-200 (latent dimensions)
- n_epochs: 10-30 (training iterations)
- lr_all: 0.005-0.02 (learning rate)
- reg_all: 0.02-0.1 (regularization)

**Training Data**:
- Filter users with < 5 ratings
- Train/test split: 80/20
- Validation for hyperparameter tuning

## Evaluation Strategy

### Offline Metrics

**Rating Prediction**:
- RMSE < 1.0 (target: 0.85-0.90)
- MAE < 0.8 (target: 0.65-0.75)

**Ranking Quality**:
- Precision@10 > 0.7
- Recall@10 > 0.15

**System Quality**:
- Coverage > 80% (recommend diverse movies)
- Diversity > 0.6 (varied recommendations per user)

### Baseline Comparison

Must outperform:
1. Global average
2. Movie average
3. User average
4. Popularity-based

## Expected Performance

Based on literature and similar datasets (MovieLens 100K):

**SVD Performance**:
- RMSE: 0.87-0.92
- MAE: 0.68-0.73
- Training time: 10-30 seconds
- Prediction time: < 10ms per user

**Baseline Performance**:
- Global average RMSE: ~1.12
- Movie average RMSE: ~0.98
- Our SVD should be ~10-20% better

## Implementation Plan

**Phase 1** (Current):
- ✓ Baseline models implemented
- ✓ Evaluation metrics ready
- ✓ Data pipeline complete

**Phase 2** (Next):
- Implement SVD recommender
- Hyperparameter tuning
- Model evaluation and comparison

**Phase 3** (After):
- Add cold-start handling
- Integrate all components
- Deploy to Cloud Run

## Alternative Approaches (Future)

If we had more time/data/resources:

1. **Hybrid Models**: Combine collaborative + content-based
2. **Context-Aware**: Use temporal features, time of day
3. **Neural CF**: Deep learning for larger datasets
4. **Real-time Learning**: Online learning with new ratings
5. **Multi-Armed Bandits**: Exploration vs exploitation

## References

- Koren, Y. (2009). "Matrix Factorization Techniques for Recommender Systems"
- Ricci, F. et al. (2015). "Recommender Systems Handbook"
- Netflix Prize competition results
- scikit-surprise documentation