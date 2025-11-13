"""
FastAPI application for movie recommendations
"""
import os
import logging
from datetime import datetime
from typing import List, Optional
import pickle

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (
    Rating, RatingRequest, Movie, RecommendationResponse, 
    RatingResponse, HealthResponse
)
from src.data.bigquery_loader import BigQueryLoader
from src.models.recommender import MovieRecommender
from src.models.cold_start import ColdStartHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommendation API",
    description="Personalized movie recommendations using collaborative filtering",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware (allows web browsers to access API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and data
recommender = None
cold_start_handler = None
bigquery_loader = None
movies_df = None
user_ratings_cache = {}  # Cache for user ratings: {user_id: [ratings]}

# Configuration
PROJECT_ID = os.getenv('PROJECT_ID', 'master-ai-cloud')
MODEL_PATH = os.getenv('MODEL_PATH', 'models/recommender_v2_final.pkl')


@app.on_event("startup")
async def startup_event():
    """
    Load model and initialize connections on startup
    """
    global recommender, cold_start_handler, bigquery_loader, movies_df
    
    logger.info("Starting up API...")
    
    try:
        # Load the trained model
        logger.info(f"Loading model from {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            recommender = pickle.load(f)
        logger.info("✓ Model loaded successfully")
        
        # Initialize BigQuery loader
        logger.info("Initializing BigQuery connection...")
        bigquery_loader = BigQueryLoader(project_id=PROJECT_ID)
        logger.info("✓ BigQuery connected")
        
        # Load movies dataframe
        logger.info("Loading movies data...")
        movies_df = bigquery_loader.load_movies()
        logger.info(f"✓ Loaded {len(movies_df)} movies")
        
        # Initialize cold start handler
        logger.info("Initializing cold start handler...")
        ratings_df = bigquery_loader.load_ratings(limit=100000)
        cold_start_handler = ColdStartHandler(ratings_df, movies_df)
        logger.info("✓ Cold start handler initialized")
        
        logger.info("🚀 API startup complete!")
        
    except Exception as e:
        logger.error(f"Failed to start up: {str(e)}")
        raise


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - API information
    """
    return {
        "message": "Movie Recommendation API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "popular_movies": "/movies/popular",
            "search_movies": "/movies/search",
            "movie_details": "/movies/{movie_id}",
            "recommendations": "/user/{user_id}/recommendations",
            "rate_movies": "/user/{user_id}/rate",
            "similar_movies": "/movies/{movie_id}/similar"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    """
    return HealthResponse(
        status="healthy",
        model_loaded=recommender is not None,
        bigquery_connected=bigquery_loader is not None,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )