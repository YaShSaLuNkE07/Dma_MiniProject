"""
Content-Based Movie Recommendation Module.

Uses TF-IDF Vectorization on movie genres and metadata combined with Cosine Similarity
to discover movies with identical or conceptually related thematic compositions.
"""

import sys
from difflib import get_close_matches
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root to sys.path if not present
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocess import load_and_preprocess_data, RANDOM_SEED

np.random.seed(RANDOM_SEED)


class ContentRecommender:
    """
    Content-Based Recommender using TF-IDF on movie genres and Cosine Similarity.
    """

    def __init__(self, movies_df: pd.DataFrame):
        """
        Initializes the recommender with cleaned movie metadata and builds the similarity matrix.
        """
        self.movies_df = movies_df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b[a-zA-Z0-9_-]+\b",  # Keeps composite genres like 'Sci-Fi' intact
            stop_words="english",
        )

        # Fit TF-IDF on space-separated genre strings
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies_df["genres_str"])

        # Compute pairwise Cosine Similarity (shape: N_movies x N_movies)
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        # Create fast lookup tables
        self._build_lookup_indices()

    def _build_lookup_indices(self):
        """Builds lookup structures for exact and case-insensitive matching."""
        self.clean_title_to_idx = {
            title.lower(): idx for idx, title in enumerate(self.movies_df["clean_title"])
        }
        self.raw_title_to_idx = {
            title.lower(): idx for idx, title in enumerate(self.movies_df["raw_title"])
        }
        self.movie_id_to_idx = {
            mid: idx for idx, mid in enumerate(self.movies_df["movieId"])
        }
        self.all_search_titles = list(self.movies_df["clean_title"].values)

    def find_movie_index(self, query: str) -> Tuple[int, str]:
        """
        Resolves a movie title query to its internal catalog index using:
        1. Exact match (case-insensitive) on clean title or raw title
        2. Fuzzy matching via difflib.get_close_matches

        Returns:
            Tuple of (index, matched_movie_title)
        Raises:
            KeyError if no satisfactory match is found.
        """
        q_clean = query.strip().lower()

        # 1. Exact match lookup
        if q_clean in self.clean_title_to_idx:
            idx = self.clean_title_to_idx[q_clean]
            return idx, self.movies_df.loc[idx, "clean_title"]

        if q_clean in self.raw_title_to_idx:
            idx = self.raw_title_to_idx[q_clean]
            return idx, self.movies_df.loc[idx, "clean_title"]

        # 2. Fuzzy match lookup
        close_matches = get_close_matches(query.strip(), self.all_search_titles, n=1, cutoff=0.55)
        if close_matches:
            matched_title = close_matches[0]
            idx = self.clean_title_to_idx[matched_title.lower()]
            return idx, matched_title

        # Check raw titles for fuzzy match as fallback
        close_raw = get_close_matches(query.strip(), list(self.movies_df["raw_title"].values), n=1, cutoff=0.55)
        if close_raw:
            matched_raw = close_raw[0]
            idx = self.raw_title_to_idx[matched_raw.lower()]
            return idx, self.movies_df.loc[idx, "clean_title"]

        raise KeyError(
            f"Movie '{query}' not found in the preprocessed catalog. "
            "Please check the spelling or select a movie from the catalog list."
        )

    def recommend(self, movie_title: str, n: int = 10) -> pd.DataFrame:
        """
        Returns top-N content-based recommendations for a target movie.

        Args:
            movie_title: Name of the movie to query (e.g. 'Toy Story')
            n: Number of recommendations to return (default: 10)

        Returns:
            DataFrame with columns ['movieId', 'title', 'genres', 'similarity_score']
        """
        target_idx, matched_title = self.find_movie_index(movie_title)

        # Retrieve pairwise similarity vector for target movie
        sim_scores = self.similarity_matrix[target_idx].copy()

        # Set target movie's self-similarity to -1 so it is excluded from results
        sim_scores[target_idx] = -1.0

        # Sort indices by similarity descending
        top_indices = np.argsort(sim_scores)[::-1][:n]

        results = []
        for idx in top_indices:
            row = self.movies_df.iloc[idx]
            results.append({
                "movieId": int(row["movieId"]),
                "title": row["clean_title"],
                "raw_title": row["raw_title"],
                "genres": row["genres"],
                "similarity_score": round(float(sim_scores[idx]), 4),
            })

        return pd.DataFrame(results)


# Global singleton instance for module-level function
_GLOBAL_CONTENT_RECOMMENDER: Optional[ContentRecommender] = None


def get_content_recommender(data_dir: Optional[Path | str] = None) -> ContentRecommender:
    """Returns or creates the cached ContentRecommender singleton."""
    global _GLOBAL_CONTENT_RECOMMENDER
    if _GLOBAL_CONTENT_RECOMMENDER is None:
        data = load_and_preprocess_data(data_dir)
        _GLOBAL_CONTENT_RECOMMENDER = ContentRecommender(data.movies_clean)
    return _GLOBAL_CONTENT_RECOMMENDER


def recommend_content(movie_title: str, n: int = 10) -> pd.DataFrame:
    """
    Public API function to get content-based recommendations.

    Args:
        movie_title: Name of the movie (e.g. 'Toy Story')
        n: Number of recommendations (default: 10)

    Returns:
        DataFrame containing ['movieId', 'title', 'genres', 'similarity_score']
    """
    recommender = get_content_recommender()
    return recommender.recommend(movie_title, n=n)


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 4: CONTENT-BASED FILTERING TEST")
    print("=" * 60)

    test_movie = "Toy Story"
    print(f"Generating Top-10 Content-Based Recommendations for: '{test_movie}'\n")

    recs = recommend_content(test_movie, n=10)
    print(recs[["title", "genres", "similarity_score"]].to_string(index=False))
    print("=" * 60)
