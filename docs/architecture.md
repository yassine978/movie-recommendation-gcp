# System Architecture

## Overview

The Movie Recommendation System is a cloud-native application deployed on Google Cloud Platform (GCP). It follows a modular architecture with clear separation of concerns:

1. **Data Layer**: BigQuery for data storage
2. **Processing Layer**: Python modules for data processing and ML
3. **Model Layer**: Trained recommendation models
4. **API Layer**: FastAPI REST service
5. **Deployment Layer**: Docker + Cloud Run

## Architecture Diagram
```
┌────────────────────────────────────────────┐
│           BIGQUERY (Data Source)           │
│      master-ai-cloud.MoviePlatform         │
│                                            │
│    ┌─────────────┐    ┌──────────────┐    │
│    │   movies    │    │   ratings    │    │
│    └─────────────┘    └──────────────┘    │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│         PYTHON APPLICATION                 │
│                                            │
│  ┌────────────────────────────────────┐   │
│  │  Data Processing                   │   │
│  │  - BigQueryLoader                  │   │
│  │  - DataPreprocessor                │   │
│  └────────────────────────────────────┘   │
│                                            │
│  ┌────────────────────────────────────┐   │
│  │  Models                            │   │
│  │  - Baseline Models                 │   │
│  │  - SVD Recommender                 │   │
│  │  - Cold Start Handler              │   │
│  └────────────────────────────────────┘   │
│                                            │
│  ┌────────────────────────────────────┐   │
│  │  API (FastAPI)                     │   │
│  │  - Recommendation endpoints        │   │
│  │  - Rating submission               │   │
│  └────────────────────────────────────┘   │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│      DOCKER CONTAINER                      │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│      GOOGLE CLOUD RUN                      │
│      (Serverless Deployment)               │
└────────────────────────────────────────────┘
```

## Component Details

### 1. Data Layer

**BigQuery Dataset**: `master-ai-cloud.MoviePlatform`

Tables:
- `movies`: Movie metadata (~9K movies)
- `ratings`: User ratings (~100K ratings)

Access Pattern:
- Read-only access via service account
- Query optimization with LIMIT and column selection
- Sample data for development, full data for production

### 2. Data Processing

**Module**: `src/data/`

Components:
- `bigquery_loader.py`: Handles BigQuery connections and queries
- `preprocessing.py`: Data cleaning, splitting, feature engineering

Key Functions:
- Load movies and ratings with optional limits
- Create train/test splits
- Filter cold-start users/movies
- Extract genre features

### 3. Model Layer

**Module**: `src/models/`

Baseline Models (`baseline.py`):
- GlobalAverageBaseline: Predicts global average rating
- MovieAverageBaseline: Predicts per-movie average
- UserAverageBaseline: Predicts per-user average
- PopularityBasedRecommender: Recommends popular movies

Main Recommender (to be implemented):
- SVD-based collaborative filtering
- Handles personalized recommendations
- Incremental learning support

Cold Start (`cold_start.py`):
- Popularity-based recommendations
- Genre-based filtering
- Hybrid approach

### 4. Evaluation

**Module**: `src/utils/evaluation.py`

Metrics:
- Rating Prediction: RMSE, MAE
- Ranking Quality: Precision@K, Recall@K, NDCG@K
- System Quality: Coverage, Diversity

### 5. API Layer

**Module**: `src/api/main.py`

Framework: FastAPI

Endpoints:
- `GET /health`: Health check
- `GET /movies/popular`: Popular movies
- `GET /user/{id}/recommendations`: Personalized recommendations
- `POST /user/{id}/rate`: Submit ratings
- `GET /movies/search`: Search movies

### 6. Deployment

**Container**: Docker
**Platform**: Google Cloud Run

Configuration:
- Memory: 2 GiB
- CPU: 2 vCPUs
- Timeout: 300s
- Auto-scaling: 0-10 instances

## Data Flow

### Training Flow
```
BigQuery → Load Data → Preprocess → Train Model → Save Model → Cloud Storage
```

### Inference Flow
```
User Request → API → Load Model → Generate Recommendations → Return JSON
```

### Cold Start Flow
```
New User → Check Ratings → No Ratings → Popular Movies
                         → Few Ratings → Genre-based
                         → Many Ratings → Personalized (SVD)
```

## Technology Stack

- **Language**: Python 3.11+
- **Data**: BigQuery, pandas, numpy
- **ML**: scikit-surprise, scikit-learn
- **API**: FastAPI, Pydantic, Uvicorn
- **Testing**: pytest
- **Deployment**: Docker, Cloud Run
- **Monitoring**: Cloud Logging

## Design Decisions

### Why SVD for Collaborative Filtering?

1. **Proven**: Industry-standard approach
2. **Scalable**: Handles 100K+ ratings efficiently
3. **Handles Sparsity**: Works with >99% missing data
4. **Simple**: Easier to deploy than deep learning

### Why Popularity for Cold Start?

1. **Fast**: No computation needed
2. **Effective**: Works well for new users
3. **Explainable**: Users understand why they see popular movies

### Why FastAPI?

1. **Modern**: Async support, type hints
2. **Fast**: High performance for ML serving
3. **Auto-docs**: Swagger UI generated automatically

### Why Cloud Run?

1. **Serverless**: No server management
2. **Cost-effective**: Pay per request, scale to zero
3. **Simple**: Easy deployment from containers

## Security Considerations

- Service account with minimal permissions (BigQuery Viewer)
- No sensitive data in code (environment variables)
- API authentication (optional for production)
- Rate limiting (optional for production)

## Scalability

Current Design:
- Handles 100 requests/second
- Model fits in memory (< 50 MB)
- Stateless API (easy to scale horizontally)

Future Improvements:
- Model caching (Redis)
- Batch prediction optimization
- Model versioning and A/B testing

## Monitoring & Observability

Metrics:
- Request latency (p50, p95, p99)
- Error rate
- Request count
- Model prediction latency

Logging:
- Request logs
- Error logs
- Model prediction logs

## Next Steps

Phase 2:
- Implement SVD recommender
- Add cold start handler
- Model hyperparameter tuning

Phase 3:
- Build FastAPI application
- Create Docker container
- Deploy to Cloud Run

Phase 4:
- Create demo notebook
- Performance benchmarking
- Final documentation