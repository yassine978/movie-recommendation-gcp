"""
FastAPI application for movie recommendations
- BigQuery optional (may be unavailable on student project)
- Robust local fallback using movies_df stored inside the trained model (.pkl)
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
    RatingRequest,
    Movie,
    RecommendationResponse,
    RatingResponse,
    HealthResponse,
)
from src.data.bigquery_loader import BigQueryLoader
from src.models.recommender import MovieRecommender
from src.models.cold_start import ColdStartHandler

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Movie Recommendation API",
    description="Personalized movie recommendations using collaborative filtering",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
recommender: Optional[MovieRecommender] = None
cold_start_handler: Optional[ColdStartHandler] = None
bigquery_loader: Optional[BigQueryLoader] = None
movies_df: Optional[pd.DataFrame] = None

user_ratings_cache = {}  # {user_id: [{"movieId": int, "rating": float}, ...]}

DATA_SOURCE = "unknown"
BIGQUERY_OK = False

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
PROJECT_ID = os.getenv("PROJECT_ID", "students-group2")
DATASET_ID = os.getenv("DATASET_ID", "MoviePlatform")
MODEL_PATH = os.getenv("MODEL_PATH", "models/recommender_v2_final.pkl")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _normalize_movies_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure movies_df has required columns and sane types."""
    if df is None:
        raise RuntimeError("movies_df is None")

    df = df.copy()

    # common alternate names (just in case)
    rename_map = {}
    if "movie_id" in df.columns and "movieId" not in df.columns:
        rename_map["movie_id"] = "movieId"
    if "name" in df.columns and "title" not in df.columns:
        rename_map["name"] = "title"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"movieId", "title", "genres"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"movies_df missing columns {missing}. Found={list(df.columns)}")

    # types
    try:
        df["movieId"] = df["movieId"].astype(int)
    except Exception:
        # keep as-is if conversion fails
        pass

    df["title"] = df["title"].fillna("").astype(str)
    df["genres"] = df["genres"].fillna("").astype(str)
    return df


def _local_popular_movies(
    n: int = 10,
    genre: Optional[str] = None,
    exclude_movie_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Local fallback for "popular" movies when we don't have ratings.
    Strategy: deterministic sample from movies_df (optionally genre filtered).
    """
    if movies_df is None or len(movies_df) == 0:
        return pd.DataFrame(columns=["movieId", "title", "genres"])

    df = movies_df.copy()

    if exclude_movie_ids:
        df = df[~df["movieId"].isin(exclude_movie_ids)]

    if genre:
        df = df[df["genres"].str.contains(genre, case=False, na=False)]

    # deterministic sample so your demo is stable
    if len(df) > n:
        df = df.sample(n=n, random_state=42)
    else:
        df = df.head(n)

    return df[["movieId", "title", "genres"]]


def _local_genre_based_movies(
    preferred_genres: List[str],
    n: int = 10,
    exclude_movie_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Local fallback genre-based recs (no ratings needed)."""
    if movies_df is None or len(movies_df) == 0:
        return pd.DataFrame(columns=["movieId", "title", "genres"])

    df = movies_df.copy()

    if exclude_movie_ids:
        df = df[~df["movieId"].isin(exclude_movie_ids)]

    if preferred_genres:
        pattern = "|".join([g.replace("|", "") for g in preferred_genres])
        df = df[df["genres"].str.contains(pattern, case=False, na=False)]

    # deterministic sample
    if len(df) > n:
        df = df.sample(n=n, random_state=7)
    else:
        df = df.head(n)

    return df[["movieId", "title", "genres"]]


# -----------------------------------------------------------------------------
# Startup
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """
    Load model and initialize connections on startup.
    BigQuery is optional; fallback to local data if unavailable.
    """
    global recommender, cold_start_handler, bigquery_loader, movies_df, DATA_SOURCE, BIGQUERY_OK

    logger.info("Starting up API...")

    try:
        # 1) Load trained model
        logger.info(f"Loading model from {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)

        # Handle both dict format (old) and MovieRecommender format (new)
        if isinstance(model_data, dict):
            logger.info("Loading model from dictionary format...")
            recommender = MovieRecommender()
            recommender.model = model_data["model"]
            recommender.movies_df = model_data.get("movies_df")
            recommender.trainset = model_data.get("trainset")
            recommender.training_metrics = model_data.get("training_metrics", {})
            recommender.is_trained = True
            logger.info("✓ Model reconstructed from dictionary")
        else:
            recommender = model_data
            logger.info("✓ Model loaded directly")

        rmse_val = getattr(recommender, "training_metrics", {}).get("rmse", "N/A")
        logger.info(f"✓ Model loaded successfully (RMSE: {rmse_val})")

        # 2) Try BigQuery
        BIGQUERY_OK = False
        try:
            logger.info(
                f"Initializing BigQuery connection (project={PROJECT_ID}, dataset={DATASET_ID})..."
            )
            bigquery_loader = BigQueryLoader(project_id=PROJECT_ID, dataset_id=DATASET_ID)

            logger.info("Loading movies data from BigQuery...")
            movies_df_bq = bigquery_loader.load_movies()
            movies_df = _normalize_movies_df(movies_df_bq)
            logger.info(f"✓ Loaded {len(movies_df)} movies from BigQuery")

            logger.info("Initializing cold start handler from BigQuery ratings...")
            ratings_df = bigquery_loader.load_ratings(limit=100000)
            cold_start_handler = ColdStartHandler(ratings_df, movies_df)
            logger.info("✓ Cold start handler initialized from BigQuery")

            BIGQUERY_OK = True
            DATA_SOURCE = "bigquery"

        except Exception as e:
            logger.warning(f"BigQuery unavailable, falling back to local model data. Reason: {e}")
            BIGQUERY_OK = False
            bigquery_loader = None
            cold_start_handler = None

            # 3) Local fallback: use movies_df stored inside the model
            local_movies = None
            if isinstance(model_data, dict):
                local_movies = model_data.get("movies_df")

            if local_movies is None and getattr(recommender, "movies_df", None) is not None:
                local_movies = recommender.movies_df

            if local_movies is None:
                raise RuntimeError(
                    "Local fallback failed: no movies_df found in the model. "
                    "Ensure the trained model was saved with movies_df."
                )

            movies_df = _normalize_movies_df(local_movies)
            logger.info(f"✓ Movies ready from local model fallback: {len(movies_df)} rows")

            # Ratings are usually not available locally -> we will do local cold-start fallback
            cold_start_handler = None
            logger.warning("Cold start handler disabled (no ratings locally). Using local popular fallback.")
            DATA_SOURCE = "local"

        logger.info(f"🚀 API startup complete! (data_source={DATA_SOURCE})")

    except Exception as e:
        logger.error(f"Failed to start up: {str(e)}")
        raise


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Movie Recommendation API",
        "version": "1.0.0",
        "status": "running",
        "data_source": DATA_SOURCE,
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "popular_movies": "/movies/popular",
            "search_movies": "/movies/search",
            "movie_details": "/movies/{movie_id}",
            "recommendations": "/user/{user_id}/recommendations",
            "rate_movies": "/user/{user_id}/rate",
            "similar_movies": "/movies/{movie_id}/similar",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=recommender is not None,
        bigquery_connected=bool(BIGQUERY_OK),
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.get("/movies/popular", response_model=List[Movie], tags=["Movies"])
async def get_popular_movies(
    n: int = Query(10, ge=1, le=50, description="Number of movies to return"),
    genre: Optional[str] = Query(None, description="Filter by genre"),
):
    """
    Popular movies for cold-start.
    - If ratings-based ColdStartHandler exists => use it
    - Else => local deterministic fallback from movies_df
    """
    try:
        if movies_df is None:
            raise HTTPException(status_code=503, detail="Movies data not loaded")

        if cold_start_handler is not None:
            popular_list = cold_start_handler.get_popular_recommendations(n=n, genre=genre)
            return [
                Movie(
                    movieId=int(m["movieId"]),
                    title=m["title"],
                    genres=m["genres"],
                    popularity_score=m.get("popularity_score"),
                    num_ratings=m.get("num_ratings"),
                )
                for m in popular_list
            ]

        df = _local_popular_movies(n=n, genre=genre)
        return [
            Movie(movieId=int(r["movieId"]), title=r["title"], genres=r["genres"])
            for _, r in df.iterrows()
        ]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting popular movies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/movies/search", response_model=List[Movie], tags=["Movies"])
async def search_movies(
    query: str = Query(..., min_length=1, description="Search term"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
):
    try:
        if movies_df is None:
            raise HTTPException(status_code=503, detail="Movies data not loaded")

        mask = movies_df["title"].str.contains(query, case=False, na=False)
        results = movies_df[mask].head(limit)

        return [
            Movie(movieId=int(row["movieId"]), title=row["title"], genres=row["genres"])
            for _, row in results.iterrows()
        ]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching movies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/movies/{movie_id}", response_model=Movie, tags=["Movies"])
async def get_movie_details(movie_id: int):
    try:
        if movies_df is None:
            raise HTTPException(status_code=503, detail="Movies data not loaded")

        movie_row = movies_df[movies_df["movieId"] == movie_id]
        if len(movie_row) == 0:
            raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")

        row = movie_row.iloc[0]
        return Movie(movieId=int(row["movieId"]), title=row["title"], genres=row["genres"])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting movie details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user/{user_id}/recommendations", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(
    user_id: int,
    n: int = Query(10, ge=1, le=50, description="Number of recommendations"),
):
    """
    Recommendation strategy:
    - 0 ratings: cold_start (ratings-based if available else local fallback)
    - 1..4 ratings: genre_based (handler if available else local fallback)
    - >=5 ratings: personalized SVD (from trained model)
    """
    try:
        if movies_df is None:
            raise HTTPException(status_code=503, detail="Movies data not loaded")
        if recommender is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        num_ratings = len(user_ratings_cache.get(user_id, []))

        # -------------------------
        # A) Cold start
        # -------------------------
        if num_ratings == 0:
            if cold_start_handler is not None:
                popular_list = cold_start_handler.get_popular_recommendations(n=n)
                recommendations = [
                    Movie(
                        movieId=int(m["movieId"]),
                        title=m["title"],
                        genres=m["genres"],
                        popularity_score=m.get("popularity_score"),
                        num_ratings=m.get("num_ratings"),
                    )
                    for m in popular_list
                ]
                rec_type = "cold_start"
            else:
                df = _local_popular_movies(n=n)
                recommendations = [
                    Movie(movieId=int(r["movieId"]), title=r["title"], genres=r["genres"])
                    for _, r in df.iterrows()
                ]
                rec_type = "cold_start_local"

        # -------------------------
        # B) Few ratings -> genre based
        # -------------------------
        elif num_ratings < 5:
            user_ratings = user_ratings_cache[user_id]
            rated_movie_ids = [r["movieId"] for r in user_ratings]

            preferred_genres: List[str] = []
            for rating in user_ratings:
                if rating["rating"] >= 3.5:
                    movie_row = movies_df[movies_df["movieId"] == rating["movieId"]]
                    if not movie_row.empty and pd.notna(movie_row.iloc[0]["genres"]):
                        preferred_genres.extend(movie_row.iloc[0]["genres"].split("|"))
            preferred_genres = list(sorted(set([g for g in preferred_genres if g])))

            if cold_start_handler is not None:
                if preferred_genres:
                    genre_list = cold_start_handler.get_genre_based_recommendations(
                        preferred_genres=preferred_genres,
                        n=n,
                        exclude_movie_ids=rated_movie_ids,
                    )
                else:
                    genre_list = cold_start_handler.get_popular_recommendations(
                        n=n,
                        exclude_movie_ids=rated_movie_ids,
                    )

                recommendations = [
                    Movie(movieId=int(m["movieId"]), title=m["title"], genres=m["genres"])
                    for m in genre_list
                ]
                rec_type = "genre_based"
            else:
                df = _local_genre_based_movies(preferred_genres, n=n, exclude_movie_ids=rated_movie_ids)
                if len(df) == 0:
                    df = _local_popular_movies(n=n, exclude_movie_ids=rated_movie_ids)
                recommendations = [
                    Movie(movieId=int(r["movieId"]), title=r["title"], genres=r["genres"])
                    for _, r in df.iterrows()
                ]
                rec_type = "genre_based_local"

        # -------------------------
        # C) Personalized (SVD)
        # -------------------------
        else:
            recs = recommender.get_top_n_recommendations(
                user_id=user_id,
                n=n,
                exclude_rated=True,
            )

            recommendations: List[Movie] = []
            for rec in recs:
                movie_row = movies_df[movies_df["movieId"] == rec["movieId"]]
                if not movie_row.empty:
                    recommendations.append(
                        Movie(
                            movieId=int(rec["movieId"]),
                            title=movie_row.iloc[0]["title"],
                            genres=movie_row.iloc[0]["genres"],
                            predicted_rating=float(rec["predicted_rating"]),
                        )
                    )
            rec_type = "personalized"

        logger.info(f"Generated {len(recommendations)} {rec_type} recommendations for user {user_id}")

        return RecommendationResponse(
            user_id=user_id,
            num_ratings=num_ratings,
            recommendations=recommendations,
            recommendation_type=rec_type,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/user/{user_id}/rate", response_model=RatingResponse, tags=["Ratings"])
async def submit_ratings(user_id: int, rating_request: RatingRequest):
    try:
        if user_id not in user_ratings_cache:
            user_ratings_cache[user_id] = []

        for rating in rating_request.ratings:
            user_ratings_cache[user_id].append({"movieId": rating.movieId, "rating": rating.rating})

        total_ratings = len(user_ratings_cache[user_id])
        logger.info(f"User {user_id} submitted {len(rating_request.ratings)} ratings (total: {total_ratings})")

        return RatingResponse(
            status="success",
            user_id=user_id,
            ratings_submitted=len(rating_request.ratings),
            total_ratings=total_ratings,
            message=f"Successfully recorded {len(rating_request.ratings)} ratings",
        )

    except Exception as e:
        logger.error(f"Error submitting ratings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/meta", tags=["Meta"])
async def meta():
    """Return runtime metadata for the UI."""
    return {
        "data_source": DATA_SOURCE,
        "bigquery_ok": bool(BIGQUERY_OK),
        "num_movies": int(len(movies_df)) if movies_df is not None else 0,
        "model_loaded": recommender is not None,
        "cold_start_handler": cold_start_handler is not None,
    }


@app.get("/user/{user_id}/ratings", tags=["Ratings"])
async def get_user_ratings(user_id: int):
    """Debug endpoint: show cached ratings for a user."""
    return {
        "user_id": user_id,
        "ratings": user_ratings_cache.get(user_id, []),
        "count": len(user_ratings_cache.get(user_id, [])),
    }


@app.post("/user/{user_id}/reset", tags=["Ratings"])
async def reset_user(user_id: int):
    """Reset a user's cached ratings (useful for demos)."""
    user_ratings_cache[user_id] = []
    return {"status": "ok", "user_id": user_id, "message": "user ratings reset"}


@app.get("/movies/{movie_id}/similar", response_model=dict, tags=["Movies"])
async def get_similar_movies(
    movie_id: int,
    n: int = Query(10, ge=1, le=50, description="Number of similar movies"),
):
    try:
        if movies_df is None:
            raise HTTPException(status_code=503, detail="Movies data not loaded")

        movie_row = movies_df[movies_df["movieId"] == movie_id]
        if len(movie_row) == 0:
            raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")

        target_movie = movie_row.iloc[0]
        target_genres = set(str(target_movie["genres"]).split("|"))

        def genre_similarity(genres_str: str) -> float:
            genres = set(str(genres_str).split("|"))
            inter = len(target_genres & genres)
            union = len(target_genres | genres)
            return inter / union if union > 0 else 0.0

        df = movies_df.copy()
        df["similarity"] = df["genres"].apply(genre_similarity)
        similar = df[df["movieId"] != movie_id].nlargest(n, "similarity")

        similar_movies = [
            Movie(
                movieId=int(row["movieId"]),
                title=row["title"],
                genres=row["genres"],
                similarity_score=float(row["similarity"]),
            )
            for _, row in similar.iterrows()
        ]

        return {
            "movie_id": movie_id,
            "movie_title": target_movie["title"],
            "similar_movies": similar_movies,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting similar movies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
