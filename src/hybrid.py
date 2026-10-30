"""
Hybrid Movie Recommendation Module.

Combines Content-Based Filtering (genre TF-IDF semantics) with Collaborative Filtering
(item-item co-rating patterns) using min-max score normalization and a configurable
weight parameter alpha in [0.0, 1.0].

Formula:
  Hybrid_Score = alpha * Normalized_Content_Score + (1 - alpha) * Normalized_CF_Score
"""

import sys
from pathlib import Path
from typing import Optional

# Add project root to sys.path if not present
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd

from src.content_based import ContentRecommender, get_content_recommender
from src.collaborative import ItemBasedCF, get_item_cf_recommender
from src.preprocess import RANDOM_SEED

np.random.seed(RANDOM_SEED)


class HybridRecommender:
    """
    Hybrid Recommender combining Content-Based and Item-Based Collaborative Filtering.
    """

    def __init__(
        self,
        content_recommender: Optional[ContentRecommender] = None,
        item_cf_recommender: Optional[ItemBasedCF] = None,
    ):
        self.content_rec = (
            content_recommender if content_recommender is not None else get_content_recommender()
        )
        self.item_cf_rec = (
            item_cf_recommender if item_cf_recommender is not None else get_item_cf_recommender()
        )
        self.movies_df = self.content_rec.movies_df

    def _min_max_scale(self, scores: np.ndarray) -> np.ndarray:
        """Min-max scales an array of scores into [0, 1]."""
        min_v = np.min(scores)
        max_v = np.max(scores)
        if max_v - min_v < 1e-9:
            return np.zeros_like(scores)
        return (scores - min_v) / (max_v - min_v)

    def recommend(
        self,
        movie_title: str,
        n: int = 10,
        alpha: float = 0.5,
    ) -> pd.DataFrame:
        """
        Recommends top-N movies using a weighted combination of content and collaborative signals.

        Args:
            movie_title: Name of the movie query (e.g. 'Toy Story')
            n: Number of recommendations (default: 10)
            alpha: Weight for Content-Based score between 0.0 (100% CF) and 1.0 (100% Content).
                   Default is 0.5 (balanced 50/50 blend).

        Returns:
            DataFrame containing ['movieId', 'title', 'genres', 'similarity_score', 'content_score', 'cf_score']
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha parameter must be between 0.0 and 1.0, got {alpha}")

        # Resolve index in both recommenders
        target_idx, matched_title = self.content_rec.find_movie_index(movie_title)

        # 1. Content raw similarity vector across all movies
        content_raw = self.content_rec.similarity_matrix[target_idx].copy()

        # 2. CF raw similarity vector across all movies
        cf_raw = self.item_cf_rec.similarity_matrix[target_idx].copy()

        # Normalize both vectors across catalog to put them on an identical [0, 1] scale
        content_norm = self._min_max_scale(content_raw)
        cf_norm = self._min_max_scale(cf_raw)

        # 3. Weighted hybrid fusion
        hybrid_scores = alpha * content_norm + (1.0 - alpha) * cf_norm

        # Exclude query movie itself
        hybrid_scores[target_idx] = -1.0

        # Select top-N indices
        top_indices = np.argsort(hybrid_scores)[::-1][:n]

        results = []
        for idx in top_indices:
            row = self.movies_df.iloc[idx]
            results.append({
                "movieId": int(row["movieId"]),
                "title": row["clean_title"],
                "raw_title": row["raw_title"],
                "genres": row["genres"],
                "similarity_score": round(float(hybrid_scores[idx]), 4),
                "content_score": round(float(content_norm[idx]), 4),
                "cf_score": round(float(cf_norm[idx]), 4),
            })

        return pd.DataFrame(results)


# Global singleton instance
_GLOBAL_HYBRID_RECOMMENDER: Optional[HybridRecommender] = None


def get_hybrid_recommender() -> HybridRecommender:
    """Returns or initializes the HybridRecommender singleton."""
    global _GLOBAL_HYBRID_RECOMMENDER
    if _GLOBAL_HYBRID_RECOMMENDER is None:
        _GLOBAL_HYBRID_RECOMMENDER = HybridRecommender()
    return _GLOBAL_HYBRID_RECOMMENDER


def recommend_hybrid(movie_title: str, n: int = 10, alpha: float = 0.5) -> pd.DataFrame:
    """
    Public API: Generates hybrid recommendations for a given movie title.
    """
    recommender = get_hybrid_recommender()
    return recommender.recommend(movie_title, n=n, alpha=alpha)


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 6: HYBRID RECOMMENDER TEST")
    print("=" * 60)

    test_movie = "Toy Story"
    print(f"Hybrid Recommendations for: '{test_movie}' (alpha=0.5)\n")

    recs = recommend_hybrid(test_movie, n=10, alpha=0.5)
    print(recs[["title", "genres", "similarity_score", "content_score", "cf_score"]].to_string(index=False))
    print("=" * 60)
