from __future__ import annotations
import requests
from typing import Any, Dict, List, Optional
from .config import API_BASE_URL


class APIError(RuntimeError):
    pass


def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = f"{API_BASE_URL}{path}"
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise APIError(f"GET {url} failed: {e}")


def health() -> Dict[str, Any]:
    return _get("/health")


def search_movies(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    return _get("/movies/search", params={"query": query, "limit": limit})


def popular_movies(n: int = 10, genre: Optional[str] = None) -> List[Dict[str, Any]]:
    params = {"n": n}
    if genre:
        params["genre"] = genre
    return _get("/movies/popular", params=params)

def get_recommendations(user_id: int, n: int = 10) -> Dict[str, Any]:
    return _get(f"/user/{user_id}/recommendations", params={"n": n})
