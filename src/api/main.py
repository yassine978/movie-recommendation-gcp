"""
FastAPI application for movie recommendations
"""
import os
import logging
from datetime import datetime
from typing import List, Optional
import pickle
import pandas as pd

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

@app.get("/movies/popular", response_model=List[Movie], tags=["Movies"])
async def get_popular_movies(
    n: int = Query(10, ge=1, le=50, description="Number of movies to return"),
    genre: Optional[str] = Query(None, description="Filter by genre")
):
    """Get popular movies (for cold start users)"""
    try:
        if cold_start_handler is None:
            raise HTTPException(status_code=503, detail="Cold start handler not initialized")
        
        # Get popular recommendations (returns LIST, not DataFrame)
        popular_list = cold_start_handler.get_popular_recommendations(n=n, genre=genre)
        
        # Convert list of dicts to Movie objects
        movies = []
        for movie_dict in popular_list:
            movies.append(Movie(
                movieId=movie_dict['movieId'],
                title=movie_dict['title'],
                genres=movie_dict['genres'],
                popularity_score=movie_dict['popularity_score'],
                num_ratings=movie_dict['num_ratings']
            ))
        
        logger.info(f"Returned {len(movies)} popular movies (genre={genre})")
        return movies
        
    except Exception as e:
        logger.error(f"Error getting popular movies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
        
@app.get("/movies/search", response_model=List[Movie], tags=["Movies"])
async def search_movies(
    query: str = Query(..., min_length=1, description="Search term"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results")
):
    """
    Search for movies by title
    
    Args:
        query: Search term
        limit: Maximum number of results
    
    Returns:
        List of matching movies
    """
    try:
        if movies_df is None:
            raise HTTPException(status_code=503, detail="Movies data not loaded")
        
        # Case-insensitive search
        mask = movies_df['title'].str.contains(query, case=False, na=False)
        results = movies_df[mask].head(limit)
        
        # Convert to Movie objects
        movies = []
        for _, row in results.iterrows():
            movies.append(Movie(
                movieId=int(row['movieId']),
                title=row['title'],
                genres=row['genres']
            ))
        
        logger.info(f"Search '{query}' returned {len(movies)} results")
        return movies
        
    except Exception as e:
        logger.error(f"Error searching movies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/movies/{movie_id}", response_model=Movie, tags=["Movies"])
async def get_movie_details(movie_id: int):
    """
    Get details for a specific movie
    
    Args:
        movie_id: Movie ID
    
    Returns:
        Movie details
    """
    try:
        if movies_df is None:
            raise HTTPException(status_code=503, detail="Movies data not loaded")
        
        # Find movie
        movie_row = movies_df[movies_df['movieId'] == movie_id]
        
        if len(movie_row) == 0:
            raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")
        
        row = movie_row.iloc[0]
        return Movie(
            movieId=int(row['movieId']),
            title=row['title'],
            genres=row['genres']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting movie details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user/{user_id}/recommendations", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(
    user_id: int,
    n: int = Query(10, ge=1, le=50, description="Number of recommendations")
):
    """Get personalized recommendations for a user"""
    try:
        # Get user's rating count from cache
        num_ratings = len(user_ratings_cache.get(user_id, []))
        
        # Determine recommendation strategy
        if num_ratings == 0:
            # Cold start: popular movies (returns LIST)
            popular_list = cold_start_handler.get_popular_recommendations(n=n)
            recommendations = []
            for movie_dict in popular_list:
                recommendations.append(Movie(
                    movieId=movie_dict['movieId'],
                    title=movie_dict['title'],
                    genres=movie_dict['genres'],
                    popularity_score=movie_dict.get('popularity_score')
                ))
            rec_type = "cold_start"
            
        elif num_ratings < 5:
            # Genre-based recommendations
            user_ratings = user_ratings_cache[user_id]
            rated_movie_ids = [r['movieId'] for r in user_ratings]
            
            # Extract genres from rated movies
            preferred_genres = []
            for rating in user_ratings:
                movie_id = rating['movieId']
                movie_row = movies_df[movies_df['movieId'] == movie_id]
                if not movie_row.empty and pd.notna(movie_row.iloc[0]['genres']):
                    # If user rated >= 3.5, they liked it
                    if rating['rating'] >= 3.5:
                        genres = movie_row.iloc[0]['genres'].split('|')
                        preferred_genres.extend(genres)
            
            # Get genre-based recommendations (returns LIST)
            if preferred_genres:
                genre_list = cold_start_handler.get_genre_based_recommendations(
                    preferred_genres=list(set(preferred_genres)),
                    n=n,
                    exclude_movie_ids=rated_movie_ids
                )
            else:
                # Fallback to popular
                genre_list = cold_start_handler.get_popular_recommendations(
                    n=n,
                    exclude_movie_ids=rated_movie_ids
                )
            
            recommendations = []
            for movie_dict in genre_list:
                recommendations.append(Movie(
                    movieId=movie_dict['movieId'],
                    title=movie_dict['title'],
                    genres=movie_dict['genres']
                ))
            rec_type = "genre_based"
            
        else:
            # Personalized SVD recommendations
            recs = recommender.get_top_n_recommendations(
                user_id=user_id,
                n=n,
                rated_movies=set([r['movieId'] for r in user_ratings_cache.get(user_id, [])])
            )
            recommendations = []
            for rec in recs:
                movie_row = movies_df[movies_df['movieId'] == rec['movieId']]
                if not movie_row.empty:
                    recommendations.append(Movie(
                        movieId=rec['movieId'],
                        title=movie_row.iloc[0]['title'],
                        genres=movie_row.iloc[0]['genres'],
                        predicted_rating=rec['predicted_rating']
                    ))
            rec_type = "personalized"
        
        logger.info(f"Generated {len(recommendations)} {rec_type} recommendations for user {user_id}")
        
        return RecommendationResponse(
            user_id=user_id,
            num_ratings=num_ratings,
            recommendations=recommendations,
            recommendation_type=rec_type
        )
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
        

@app.post("/user/{user_id}/rate", response_model=RatingResponse, tags=["Ratings"])
async def submit_ratings(user_id: int, rating_request: RatingRequest):
    """
    Submit movie ratings for a user
    
    Args:
        user_id: User ID
        rating_request: List of ratings
    
    Returns:
        Confirmation response
    """
    try:
        # Initialize user's ratings if not exists
        if user_id not in user_ratings_cache:
            user_ratings_cache[user_id] = []
        
        # Add new ratings to cache
        for rating in rating_request.ratings:
            user_ratings_cache[user_id].append({
                'movieId': rating.movieId,
                'rating': rating.rating
            })
        
        total_ratings = len(user_ratings_cache[user_id])
        
        logger.info(f"User {user_id} submitted {len(rating_request.ratings)} ratings (total: {total_ratings})")
        
        return RatingResponse(
            status="success",
            user_id=user_id,
            ratings_submitted=len(rating_request.ratings),
            total_ratings=total_ratings,
            message=f"Successfully recorded {len(rating_request.ratings)} ratings"
        )
        
    except Exception as e:
        logger.error(f"Error submitting ratings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/movies/{movie_id}/similar", response_model=dict, tags=["Movies"])
async def get_similar_movies(
    movie_id: int,
    n: int = Query(10, ge=1, le=50, description="Number of similar movies")
):
    """
    Get similar movies based on genre
    
    Args:
        movie_id: Movie ID
        n: Number of similar movies
    
    Returns:
        List of similar movies
    """
    try:
        if movies_df is None:
            raise HTTPException(status_code=503, detail="Movies data not loaded")
        
        # Get the target movie
        movie_row = movies_df[movies_df['movieId'] == movie_id]
        if len(movie_row) == 0:
            raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")
        
        target_movie = movie_row.iloc[0]
        target_genres = set(target_movie['genres'].split('|'))
        
        # Find similar movies by genre overlap
        def genre_similarity(genres_str):
            genres = set(genres_str.split('|'))
            intersection = len(target_genres & genres)
            union = len(target_genres | genres)
            return intersection / union if union > 0 else 0
        
        movies_df['similarity'] = movies_df['genres'].apply(genre_similarity)
        similar = movies_df[movies_df['movieId'] != movie_id].nlargest(n, 'similarity')
        
        # Convert to response
        similar_movies = []
        for _, row in similar.iterrows():
            similar_movies.append(Movie(
                movieId=int(row['movieId']),
                title=row['title'],
                genres=row['genres'],
                similarity_score=float(row['similarity'])
            ))
        
        return {
            "movie_id": movie_id,
            "movie_title": target_movie['title'],
            "similar_movies": similar_movies
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting similar movies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))