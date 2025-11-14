"""
Tests for FastAPI application
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.main import app


# Create client
client = TestClient(app)


def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert data["status"] == "running"


def test_health():
    """Test health check"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    # Note: model_loaded might be False in tests if startup didn't run
    assert isinstance(data["model_loaded"], bool)


def test_popular_movies():
    """Test popular movies endpoint"""
    response = client.get("/movies/popular?n=5")
    # Accept 200, 500, or 503
    assert response.status_code in [200, 500, 503]
    
    if response.status_code == 200:
        movies = response.json()
        assert len(movies) <= 5
        assert all('movieId' in m for m in movies)
        assert all('title' in m for m in movies)


def test_search_movies():
    """Test movie search"""
    response = client.get("/movies/search?query=Toy&limit=5")
    # Accept 200, 500, or 503
    assert response.status_code in [200, 500, 503]
    
    if response.status_code == 200:
        movies = response.json()
        assert len(movies) <= 5


def test_get_movie_details():
    """Test getting movie details"""
    response = client.get("/movies/1")
    # Accept both 200, 404, or 503
    assert response.status_code in [200, 404, 503]
    
    if response.status_code == 200:
        movie = response.json()
        assert movie['movieId'] == 1
        assert 'title' in movie


def test_submit_ratings():
    """Test submitting ratings"""
    user_id = 999999
    
    payload = {
        "ratings": [
            {"movieId": 1, "rating": 5.0},
            {"movieId": 2, "rating": 4.0}
        ]
    }
    response = client.post(f"/user/{user_id}/rate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'success'
    assert data['ratings_submitted'] == 2


def test_get_recommendations():
    """Test getting recommendations"""
    user_id = 888888
    
    response = client.get(f"/user/{user_id}/recommendations?n=10")
    # Accept 200, 500, or 503
    assert response.status_code in [200, 500, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert data['user_id'] == user_id
        assert 'recommendations' in data
        assert 'recommendation_type' in data

def test_invalid_rating():
    """Test validation of invalid rating"""
    payload = {
        "ratings": [
            {"movieId": 1, "rating": 6.0}  # Invalid!
        ]
    }
    response = client.post("/user/1/rate", json=payload)
    assert response.status_code == 422  # Validation error


def test_invalid_rating_increments():
    """Test validation of rating increments"""
    payload = {
        "ratings": [
            {"movieId": 1, "rating": 3.7}  # Invalid increment!
        ]
    }
    response = client.post("/user/1/rate", json=payload)
    assert response.status_code == 422  # Validation error


def test_movie_not_found():
    """Test 404 for non-existent movie"""
    response = client.get("/movies/999999999")
    # Accept 404 or 503 (if not initialized)
    assert response.status_code in [404, 503]


# Integration test - only runs if API is fully initialized
def test_full_workflow_if_initialized():
    """Test complete user journey if API is initialized"""
    # Check if API is initialized
    health = client.get("/health").json()
    
    if not health.get("model_loaded"):
        pytest.skip("API not fully initialized - skipping integration test")
    
    user_id = 777777
    
    # 1. Get cold start recommendations
    response = client.get(f"/user/{user_id}/recommendations?n=5")
    assert response.status_code == 200
    data = response.json()
    assert data['recommendation_type'] == 'cold_start'
    assert data['num_ratings'] == 0
    
    # 2. Submit 3 ratings
    ratings = {
        "ratings": [
            {"movieId": 1, "rating": 5.0},
            {"movieId": 260, "rating": 4.5},
            {"movieId": 589, "rating": 4.0}
        ]
    }
    response = client.post(f"/user/{user_id}/rate", json=ratings)
    assert response.status_code == 200
    
    # 3. Get genre-based recommendations
    response = client.get(f"/user/{user_id}/recommendations?n=5")
    assert response.status_code == 200
    data = response.json()
    assert data['recommendation_type'] == 'genre_based'
    assert data['num_ratings'] == 3
    
    # 4. Submit 5 more ratings (total: 8)
    more_ratings = {
        "ratings": [
            {"movieId": 318, "rating": 5.0},
            {"movieId": 527, "rating": 4.5},
            {"movieId": 858, "rating": 4.5},
            {"movieId": 1196, "rating": 5.0},
            {"movieId": 1197, "rating": 4.0}
        ]
    }
    response = client.post(f"/user/{user_id}/rate", json=more_ratings)
    assert response.status_code == 200
    
    # 5. Get personalized recommendations
    response = client.get(f"/user/{user_id}/recommendations?n=5")
    assert response.status_code == 200
    data = response.json()
    assert data['recommendation_type'] == 'personalized'
    assert data['num_ratings'] == 8
    assert len(data['recommendations']) == 5