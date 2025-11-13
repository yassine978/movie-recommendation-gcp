"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional


class Rating(BaseModel):
    """Single movie rating"""
    movieId: int = Field(..., gt=0, description="Movie ID")
    rating: float = Field(..., ge=0.5, le=5.0, description="Rating value")
    
    @validator('rating')
    def validate_rating(cls, v):
        """Ensure rating is in 0.5 increments"""
        valid_ratings = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        if v not in valid_ratings:
            raise ValueError('Rating must be in 0.5 increments (0.5, 1.0, ..., 5.0)')
        return v


class RatingRequest(BaseModel):
    """Request body for submitting ratings"""
    ratings: List[Rating] = Field(..., min_items=1, max_items=50)


class Movie(BaseModel):
    """Movie information"""
    movieId: int
    title: str
    genres: str
    predicted_rating: Optional[float] = None
    popularity_score: Optional[float] = None
    num_ratings: Optional[int] = None
    similarity_score: Optional[float] = None


class RecommendationResponse(BaseModel):
    """Response for recommendation requests"""
    user_id: int
    num_ratings: int
    recommendations: List[Movie]
    recommendation_type: str  # "personalized", "cold_start", "genre_based"


class RatingResponse(BaseModel):
    """Response after submitting ratings"""
    status: str
    user_id: int
    ratings_submitted: int
    total_ratings: int
    message: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    bigquery_connected: bool
    timestamp: str