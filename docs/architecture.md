# System Architecture

## Overview

[Diagram will be added here]

## Components

### Data Layer
- BigQuery: MovieLens dataset storage

### Processing Layer
- Vertex AI Workbench: Development environment
- Python: Data processing and model training

### Model Layer
- SVD Algorithm: Collaborative filtering
- Cloud Storage: Model storage

### API Layer
- FastAPI: REST API
- Docker: Containerization

### Deployment Layer
- Cloud Run: Serverless deployment

## Data Flow

1. BigQuery → Data Loading → Preprocessing
2. Preprocessing → Model Training → Model Storage
3. Model Storage → API → Cloud Run
4. User → API → Recommendations

---

**Status**: In Development  
**Last Updated**: 2025-11-07
