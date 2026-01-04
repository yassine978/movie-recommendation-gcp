"""
Local data loader (fallback) for movies/ratings.
Priority:
1) from the loaded model object (recommender.movies_df if present)
2) from local files in data/processed/
3) empty DataFrame (API still starts)
"""
from __future__ import annotations

import os
import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


class LocalDataLoader:
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = data_dir

    def load_movies(self, recommender=None) -> pd.DataFrame:
        # 1) from model
        if recommender is not None and getattr(recommender, "movies_df", None) is not None:
            df = recommender.movies_df.copy()
            logger.info(f"Loaded movies from model object: {len(df)} rows")
            return self._normalize_movies(df)

        # 2) from local file
        movies_path = os.path.join(self.data_dir, "movies.csv")
        if os.path.exists(movies_path):
            df = pd.read_csv(movies_path)
            logger.info(f"Loaded movies from {movies_path}: {len(df)} rows")
            return self._normalize_movies(df)

        # 3) empty fallback
        logger.warning("No local movies source found. Using empty movies DataFrame.")
        return pd.DataFrame(columns=["movieId", "title", "genres"])

    def load_ratings(self) -> pd.DataFrame:
        ratings_path = os.path.join(self.data_dir, "ratings.csv")
        if os.path.exists(ratings_path):
            df = pd.read_csv(ratings_path)
            logger.info(f"Loaded ratings from {ratings_path}: {len(df)} rows")
            return df
        logger.warning("No local ratings source found. Using empty ratings DataFrame.")
        return pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])

    @staticmethod
    def _normalize_movies(df: pd.DataFrame) -> pd.DataFrame:
        # Ensure required columns exist
        for col in ["movieId", "title", "genres"]:
            if col not in df.columns:
                df[col] = None
        # Clean types
        try:
            df["movieId"] = df["movieId"].astype(int)
        except Exception:
            pass
        return df[["movieId", "title", "genres"]]