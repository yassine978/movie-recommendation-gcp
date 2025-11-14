# API Documentation

## Base URL
- **Local Development**: `http://localhost:8000`
- **Production**: `https://[your-service].run.app` (to be deployed)

## Authentication
No authentication required (public API for demonstration)

## Endpoints

### 1. GET `/`
Get API information and available endpoints.

**Response:**
```json
{
  "message": "Movie Recommendation API",
  "version": "1.0.0",
  "status": "running",
  "endpoints": {...}
}
```

### 2. GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "bigquery_connected": true,
  "timestamp": "2025-11-14T10:30:00Z"
}
```

### 3. GET `/movies/popular`
Get popular movies (for cold start users).

**Query Parameters:**
- `n` (optional): Number of movies (1-50, default: 10)
- `genre` (optional): Filter by genre

**Example:**
```bash
curl "http://localhost:8000/movies/popular?n=5&genre=Action"
```

**Response:**
```json
[
  {
    "movieId": 94466,
    "title": "Black Mirror (2011)",
    "genres": "Drama|Sci-Fi",
    "popularity_score": 4.49,
    "num_ratings": 6
  },
  ...
]
```

### 4. GET `/movies/search`
Search for movies by title.

**Query Parameters:**
- `query` (required): Search term
- `limit` (optional): Max results (1-50, default: 10)

**Example:**
```bash
curl "http://localhost:8000/movies/search?query=Toy%20Story"
```

### 5. GET `/movies/{movie_id}`
Get details for a specific movie.

**Example:**
```bash
curl "http://localhost:8000/movies/1"
```

### 6. GET `/user/{user_id}/recommendations`
Get personalized recommendations for a user.

**Query Parameters:**
- `n` (optional): Number of recommendations (1-50, default: 10)

**Recommendation Strategy:**
- 0 ratings: Popular movies (cold_start)
- 1-4 ratings: Genre-based recommendations
- 5+ ratings: Personalized SVD recommendations

**Example:**
```bash
curl "http://localhost:8000/user/999999/recommendations?n=10"
```

**Response:**
```json
{
  "user_id": 999999,
  "num_ratings": 3,
  "recommendation_type": "genre_based",
  "recommendations": [...]
}
```

### 7. POST `/user/{user_id}/rate`
Submit movie ratings for a user.

**Request Body:**
```json
{
  "ratings": [
    {"movieId": 1, "rating": 5.0},
    {"movieId": 260, "rating": 4.5}
  ]
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/user/999999/rate" \
  -H "Content-Type: application/json" \
  -d '{"ratings":[{"movieId":1,"rating":5.0}]}'
```

### 8. GET `/movies/{movie_id}/similar`
Get similar movies based on genre.

**Query Parameters:**
- `n` (optional): Number of similar movies (1-50, default: 10)

**Example:**
```bash
curl "http://localhost:8000/movies/1/similar?n=5"
```

## Interactive Documentation

FastAPI provides auto-generated interactive documentation:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Error Responses

### 404 Not Found
```json
{
  "detail": "Movie not found"
}
```

### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "ratings", 0, "rating"],
      "msg": "Rating must be between 0.5 and 5.0"
    }
  ]
}
```

### 500 Internal Server Error
```json
{
  "detail": "Error message"
}
```

## Rating Validation

Ratings must:
- Be between 0.5 and 5.0
- Be in 0.5 increments (0.5, 1.0, 1.5, ..., 5.0)
