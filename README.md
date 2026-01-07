# Movie Recommendation System

A production-ready, cloud-native movie recommendation system built on Google Cloud Platform (GCP) using collaborative filtering, deployed with Docker and Cloud Run. This project demonstrates end-to-end ML engineering from data exploration to production deployment with intelligent cold-start handling.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Development](#development)
- [Documentation](#documentation)

---

## Overview

This movie recommendation system provides personalized movie suggestions using **SVD-based collaborative filtering**. It handles various user scenarios through a progressive recommendation strategy:

- **New users (0 ratings)**: Popular movies using Bayesian averaging
- **Casual users (1-4 ratings)**: Genre-based recommendations
- **Active users (5+ ratings)**: Fully personalized SVD recommendations

The system is built for production deployment on GCP, with robust fallback mechanisms, comprehensive testing, and complete documentation.

### Key Achievements

- ✅ **RMSE: 0.88** (26.7% improvement over baseline)
- ✅ **Fast training**: 1.47 seconds for 105K ratings
- ✅ **Real-time inference**: <100ms per user
- ✅ **Lightweight model**: ~15MB (optimized for cloud deployment)
- ✅ **Scalable architecture**: Serverless deployment with auto-scaling
- ✅ **Progressive personalization**: Seamless cold-start to personalized transition

---

## Architecture

The system follows a modular, cloud-native architecture:

```
┌─────────────────────────────────────────────────┐
│         BigQuery (Data Source)                  │
│    master-ai-cloud.MoviePlatform                │
│   • movies (~10K movies)                        │
│   • ratings (~105K ratings from 668 users)      │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│         Python Application                      │
│                                                 │
│  ┌───────────────────────────────────────┐     │
│  │  Data Layer                           │     │
│  │  • BigQueryLoader (cloud)             │     │
│  │  • LocalDataLoader (fallback)         │     │
│  │  • DataPreprocessor                   │     │
│  └───────────────────────────────────────┘     │
│                                                 │
│  ┌───────────────────────────────────────┐     │
│  │  Model Layer                          │     │
│  │  • SVD Recommender (collaborative)    │     │
│  │  • Cold Start Handler (3-tier)        │     │
│  │  • Baseline Models                    │     │
│  └───────────────────────────────────────┘     │
│                                                 │
│  ┌───────────────────────────────────────┐     │
│  │  API Layer (FastAPI)                  │     │
│  │  • REST endpoints                     │     │
│  │  • Request validation                 │     │
│  │  • Auto-documentation                 │     │
│  └───────────────────────────────────────┘     │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│         Docker Container                        │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│         Google Cloud Run                        │
│    (Serverless Container Platform)              │
│    • Auto-scaling: 0-10 instances               │
│    • Pay-per-request pricing                    │
└─────────────────────────────────────────────────┘
```

### Data Flow

**Training Pipeline:**
```
BigQuery → Load Data → Preprocess → Train SVD → Save Model → Local/Cloud Storage
```

**Inference Pipeline:**
```
User Request → FastAPI → Load Model → Generate Recs → Return JSON
```

**Progressive Personalization:**
```
User → Check Rating Count → 0 ratings → Popular Movies
                          → 1-4 ratings → Genre-based Recs
                          → 5+ ratings → Personalized SVD Recs
```

---

## Key Features

### Intelligent Recommendation Strategies

1. **Cold Start Handling**: Bayesian-averaged popular movies for new users
2. **Genre-Based Recommendations**: Hybrid approach for users with 1-4 ratings
3. **Collaborative Filtering**: SVD-based personalization for active users
4. **Similar Movie Discovery**: Content-based similarity using genre features

### Production-Ready Design

- **Robust Fallbacks**: BigQuery → Local data → Graceful error handling
- **Stateless API**: Horizontal scalability on Cloud Run
- **Health Checks**: Startup validation and continuous monitoring
- **Comprehensive Logging**: Request, error, and performance logs
- **CORS Support**: Cross-origin requests for frontend integration

### Developer Experience

- **Interactive UI**: Gradio web interface for testing and demos
- **Auto-Generated Docs**: Swagger UI at `/docs` endpoint
- **Jupyter Notebooks**: Data exploration, training, and evaluation
- **Complete Documentation**: Architecture, API, and performance reports

---

## Technology Stack

### Core Technologies

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.10+ |
| **Data Processing** | pandas 2.1.3, numpy 1.26.2 |
| **Machine Learning** | scikit-surprise 1.1.4, scikit-learn 1.3.2 |
| **API Framework** | FastAPI 0.104.1, Uvicorn 0.24.0, Pydantic 2.5.0 |
| **Cloud Services** | Google Cloud BigQuery, Google Cloud Storage |
| **Frontend** | Gradio (interactive UI), Streamlit (dashboard) |
| **Testing** | pytest 7.4.3, pytest-cov 4.1.0 |
| **Containerization** | Docker, Google Cloud Run |
| **Development** | Vertex AI Workbench |

### Dependencies

See [requirements.txt](requirements.txt) for complete dependency list.

---

## Project Structure

```
movie-recommendation-gcp/
├── src/                              # Core application code
│   ├── api/                          # FastAPI REST service
│   │   ├── main.py                   # API application
│   │   └── models.py                 # Request/response schemas
│   ├── data/                         # Data loading & preprocessing
│   │   ├── bigquery_loader.py        # GCP BigQuery integration
│   │   ├── local_loader.py           # Local fallback loader
│   │   └── preprocessing.py          # Data preprocessing pipeline
│   ├── models/                       # ML models
│   │   ├── recommender.py            # SVD collaborative filtering
│   │   ├── cold_start.py             # Cold start handler
│   │   ├── baseline.py               # Baseline models
│   │   ├── integrated_recommender.py # Model orchestration
│   │   └── train.py                  # Training script
│   └── utils/                        # Utilities
│       └── evaluation.py             # Evaluation metrics
│
├── frontend/                         # Frontend applications
│   ├── gradio_app.py                 # Gradio interactive UI
│   ├── app.py                        # Streamlit dashboard
│   ├── api_client.py                 # API client library
│   └── config.py                     # Configuration
│
├── notebooks/                        # Jupyter notebooks
│   ├── 00_integration_test.ipynb     # Integration testing
│   ├── 01_data_exploration.ipynb     # EDA and data analysis
│   └── 02b_hyperparameter_tuning.ipynb # Model optimization
│
├── docs/                             # Documentation
│   ├── architecture.md               # System architecture
│   ├── api_documentation.md          # API reference
│   ├── model_selection.md            # Model selection rationale
│   ├── performance_report.md         # Performance metrics
│   └── eda_summary.md                # EDA summary
│
├── models/                           # Trained models
│   └── recommender_v2_final.pkl      # Production SVD model
│
├── tests/                            # Test suite
│   ├── test_api.py                   # API endpoint tests
│   ├── test_models.py                # Model tests
│   ├── test_cold_start.py            # Cold start tests
│   └── test_evaluation.py            # Evaluation tests
│
├── deployment/                       # Deployment configs
│   ├── Dockerfile                    # Production Docker image
│   └── requirements.txt              # Production dependencies
│
├── Dockerfile                        # Root Docker configuration
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
├── .dockerignore                     # Docker ignore rules
└── README.md                         # This file
```

---

## Setup & Installation

### Prerequisites

- Python 3.10 or higher
- Google Cloud SDK (for BigQuery access)
- Docker (for containerized deployment)
- GCP project with BigQuery enabled

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yassine978/movie-recommendation-gcp.git
   cd movie-recommendation-gcp
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running the API Server

Start the FastAPI server locally:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### Running the Gradio UI

Launch the interactive web interface:

```bash
python frontend/gradio_app.py
```

The Gradio interface will open in your browser with tabs for:
- **Explorer**: Search movies, view details, find similar movies
- **Cold Start / Popular**: Browse popular movies
- **Rate Movies**: Submit ratings and manage your profile
- **Recommendations**: Get personalized recommendations

### Using Docker

Build and run the Docker container:

```bash
# Build the image
docker build -t movie-recommender .

# Run the container
docker run -p 8080:8080 movie-recommender
```

Access the API at `http://localhost:8080`

### Training the Model

Train or retrain the recommendation model:

```bash
python src/models/train.py
```

This will:
1. Load data from BigQuery (or local fallback)
2. Preprocess and split data
3. Train SVD model with optimized hyperparameters
4. Evaluate on test set
5. Save model to `models/recommender_v2_final.pkl`

### Running Tests

Execute the test suite:

```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v

# Run with detailed output
pytest tests/ -v -s
```

### Exploring with Notebooks

Launch Jupyter and explore the notebooks:

```bash
jupyter notebook notebooks/
```

Available notebooks:
- **00_integration_test.ipynb**: End-to-end system testing
- **01_data_exploration.ipynb**: Data analysis and visualization
- **02b_hyperparameter_tuning.ipynb**: Model optimization

---

## Model Performance

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Ratings** | 105,339 |
| **Training Set** | 84,271 (80%) |
| **Test Set** | 21,068 (20%) |
| **Total Users** | 668 |
| **Total Movies** | 10,329 |
| **Avg Ratings/User** | 157.7 |
| **Rating Scale** | 0.5 - 5.0 (0.5 increments) |
| **Data Sparsity** | >99% |

### Model Configuration

- **Algorithm**: Singular Value Decomposition (SVD)
- **Latent Factors**: 100
- **Training Epochs**: 20
- **Learning Rate**: 0.006
- **Regularization**: 0.025
- **Model Size**: ~15MB

### Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **RMSE** | 0.8802 | 26.7% improvement over baseline |
| **MAE** | 0.6783 | Mean absolute error |
| **Training Time** | 1.47 seconds | Full dataset (105K ratings) |
| **Inference Time** | <100ms | Per user recommendation |
| **Model Size** | ~15MB | Optimized for cloud deployment |

### Baseline Comparison

| Model | RMSE | MAE | Improvement |
|-------|------|-----|-------------|
| Global Average | 1.20 | 0.95 | Baseline |
| Per-Movie Average | 1.05 | 0.82 | +12.5% |
| **SVD (Optimized)** | **0.8802** | **0.6783** | **+26.7%** |

### Cold Start Strategy

| User Rating Count | Strategy | Method | Coverage |
|------------------|----------|--------|----------|
| **0 ratings** | Cold Start | Popular movies (Bayesian avg) | 100% |
| **1-4 ratings** | Genre-based | Hybrid: 0.6×popularity + 0.4×genre | 100% |
| **5+ ratings** | Personalized | SVD collaborative filtering | >95% |

See [docs/performance_report.md](docs/performance_report.md) for detailed analysis.

---

## API Documentation

### Core Endpoints

#### Health & Metadata

```http
GET /health
GET /meta
```

#### Movie Discovery

```http
GET /movies/popular?n=10&genre=Action
GET /movies/search?query=matrix&limit=10
GET /movies/{movie_id}
GET /movies/{movie_id}/similar?n=5
```

#### User Recommendations

```http
GET /user/{user_id}/recommendations?n=10&mode=auto
POST /user/{user_id}/rate
GET /user/{user_id}/ratings
POST /user/{user_id}/reset
```

### Example Requests

**Get Popular Movies**
```bash
curl "http://localhost:8000/movies/popular?n=5"
```

**Submit Rating**
```bash
curl -X POST "http://localhost:8000/user/999/rate" \
  -H "Content-Type: application/json" \
  -d '{
    "movie_id": 1,
    "rating": 4.5
  }'
```

**Get Personalized Recommendations**
```bash
curl "http://localhost:8000/user/999/recommendations?n=10"
```

### Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

See [docs/api_documentation.md](docs/api_documentation.md) for complete API reference.

---

## Development

### Known Issues & Limitations

1. **Cloud Run Deployment**: Initial deployment had configuration issues, resolved with local Gradio/FastAPI setup
2. **BigQuery Dependency**: System requires GCP access or falls back to local data
3. **Cold Start Coverage**: Limited movie metadata for advanced content-based filtering
4. **Real-time Updates**: Model requires retraining for new ratings (not incremental)

### Future Enhancements

- [ ] Implement incremental learning for real-time model updates
- [ ] Add A/B testing framework for model experimentation
- [ ] Integrate advanced NLP for movie descriptions
- [ ] Add user authentication and personalization storage
- [ ] Implement caching layer (Redis) for improved performance
- [ ] Add monitoring dashboard (Grafana/Cloud Monitoring)
- [ ] Support for implicit feedback (views, clicks)
- [ ] Multi-modal recommendations (posters, trailers)

---

## Documentation

Comprehensive documentation is available in the [docs/](docs/) directory:

| Document | Description |
|----------|-------------|
| [architecture.md](docs/architecture.md) | System architecture and design decisions |
| [api_documentation.md](docs/api_documentation.md) | Complete API endpoint reference |
| [model_selection.md](docs/model_selection.md) | ML model selection rationale and comparison |
| [performance_report.md](docs/performance_report.md) | Model performance metrics and analysis |
| [eda_summary.md](docs/eda_summary.md) | Exploratory data analysis summary |

### Additional Resources

- **Notebooks**: Interactive exploration and analysis in [notebooks/](notebooks/)
- **Tests**: Test suite with examples in [tests/](tests/)
- **API Docs**: Auto-generated Swagger UI at `/docs` endpoint

---

## Contributors

This project was developed as part of a cloud-based machine learning course focused on building production-ready recommendation systems on Google Cloud Platform.

**Development Environment**: GCP Vertex AI Workbench
**Target Deployment**: Google Cloud Run (with local fallback)
**Architecture**: Serverless, containerized microservices

### Acknowledgments

- **Dataset**: MovieLens dataset via GCP BigQuery
- **ML Framework**: scikit-surprise library
- **API Framework**: FastAPI
- **Cloud Platform**: Google Cloud Platform

---

## Contact & Support

For questions, issues, or contributions:

1. **GitHub Issues**: Report bugs or request features
2. **Documentation**: Check [docs/](docs/) for detailed guides
3. **API Documentation**: Access interactive docs at `/docs` endpoint
