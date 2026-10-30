"""
Collaborative Filtering Movie Recommendation Module.

Includes:
1. Item-Based Collaborative Filtering (Cosine similarity over item-user interaction vectors)
2. SVD Matrix Factorization (Latent factor decomposition using scikit-surprise, with scipy.sparse.linalg.svds fallback)
3. Robust fuzzy title matching for unknown or slightly misspelled queries
"""

import sys
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Add project root to sys.path if not present
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocess import PreprocessedData, load_and_preprocess_data, RANDOM_SEED

np.random.seed(RANDOM_SEED)

# Attempt to import scikit-surprise; if unavailable, rely on scipy fallback
try:
    from surprise import Dataset, Reader, SVD
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    from scipy.sparse.linalg import svds


class ItemBasedCF:
    """
    Item-Based Collaborative Filtering Recommender.
    Measures item-item similarity based on user rating patterns across the catalog.
    """

    def __init__(self, data: PreprocessedData):
        self.data = data
        self.movies_df = data.movies_clean.reset_index(drop=True)
        self.user_item_df = data.user_item_df

        # Item-user matrix: movies as rows, users as columns
        # Shape: (N_movies, N_users)
        self.item_user_matrix = self.user_item_df.T.values

        # Compute pairwise Cosine Similarity between movies
        # Movies rated similarly by the same cohorts of users will have high cosine similarity
        self.similarity_matrix = cosine_similarity(self.item_user_matrix, self.item_user_matrix)

        # Lookup dictionaries
        self.clean_title_to_idx = {
            title.lower(): idx for idx, title in enumerate(self.movies_df["clean_title"])
        }
        self.raw_title_to_idx = {
            title.lower(): idx for idx, title in enumerate(self.movies_df["raw_title"])
        }
        self.all_clean_titles = list(self.movies_df["clean_title"].values)
        self.all_raw_titles = list(self.movies_df["raw_title"].values)

    def find_movie_index(self, query: str) -> Tuple[int, str]:
        """
        Resolves a movie title to its index in the matrix using exact and fuzzy matching.
        """
        q_clean = query.strip().lower()

        # 1. Exact match
        if q_clean in self.clean_title_to_idx:
            idx = self.clean_title_to_idx[q_clean]
            return idx, self.movies_df.loc[idx, "clean_title"]

        if q_clean in self.raw_title_to_idx:
            idx = self.raw_title_to_idx[q_clean]
            return idx, self.movies_df.loc[idx, "clean_title"]

        # 2. Fuzzy match
        close_clean = get_close_matches(query.strip(), self.all_clean_titles, n=1, cutoff=0.55)
        if close_clean:
            matched_title = close_clean[0]
            idx = self.clean_title_to_idx[matched_title.lower()]
            return idx, matched_title

        close_raw = get_close_matches(query.strip(), self.all_raw_titles, n=1, cutoff=0.55)
        if close_raw:
            matched_raw = close_raw[0]
            idx = self.raw_title_to_idx[matched_raw.lower()]
            return idx, self.movies_df.loc[idx, "clean_title"]

        raise KeyError(
            f"Movie '{query}' not found. Try searching with a closer title or release year."
        )

    def recommend(self, movie_title: str, n: int = 10) -> pd.DataFrame:
        """
        Recommends top-N movies similar to the target movie based on user rating patterns.

        Args:
            movie_title: Name of the movie (e.g. 'Toy Story')
            n: Number of recommendations (default: 10)

        Returns:
            DataFrame containing ['movieId', 'title', 'genres', 'similarity_score']
        """
        target_idx, matched_title = self.find_movie_index(movie_title)

        sim_scores = self.similarity_matrix[target_idx].copy()
        sim_scores[target_idx] = -1.0  # Exclude target movie itself

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


class SVDRecommender:
    """
    SVD Matrix Factorization Recommender.
    Decomposes the sparse rating matrix into low-rank latent user and item representations.
    Supports user rating prediction as well as item latent factor similarity matching.
    """

    def __init__(self, data: PreprocessedData, n_factors: int = 50):
        self.data = data
        self.movies_df = data.movies_clean.reset_index(drop=True)
        self.ratings_df = data.ratings_clean
        self.n_factors = n_factors
        self.use_surprise = SURPRISE_AVAILABLE

        # Lookup structures for titles
        self.clean_title_to_idx = {
            title.lower(): idx for idx, title in enumerate(self.movies_df["clean_title"])
        }
        self.raw_title_to_idx = {
            title.lower(): idx for idx, title in enumerate(self.movies_df["raw_title"])
        }
        self.movie_id_to_row = {
            int(mid): idx for idx, mid in enumerate(self.movies_df["movieId"])
        }
        self.row_to_movie_id = {
            idx: int(mid) for idx, mid in enumerate(self.movies_df["movieId"])
        }
        self.all_clean_titles = list(self.movies_df["clean_title"].values)

        self._train_model()

    def _train_model(self):
        """Trains SVD either via scikit-surprise or via scipy.sparse.linalg.svds fallback."""
        if self.use_surprise:
            reader = Reader(rating_scale=(0.5, 5.0))
            surprise_data = Dataset.load_from_df(
                self.ratings_df[["userId", "movieId", "rating"]], reader
            )
            self.trainset = surprise_data.build_full_trainset()
            self.model = SVD(n_factors=self.n_factors, random_state=RANDOM_SEED, n_epochs=20)
            self.model.fit(self.trainset)

            # Build latent item factor matrix (N_movies, n_factors) aligned with self.movies_df
            item_factors = []
            for mid in self.movies_df["movieId"]:
                try:
                    inner_id = self.trainset.to_inner_iid(mid)
                    item_factors.append(self.model.qi[inner_id])
                except ValueError:
                    item_factors.append(np.zeros(self.n_factors))
            self.item_factors = np.array(item_factors)
        else:
            # Fallback using scipy svds
            user_item_arr = self.data.user_item_matrix.toarray().astype(float)
            # Subtract user mean rating for unrated zeros handling
            user_means = np.zeros(user_item_arr.shape[0])
            for i in range(user_item_arr.shape[0]):
                rated = user_item_arr[i][user_item_arr[i] > 0]
                user_means[i] = rated.mean() if len(rated) > 0 else 0.0

            centered_matrix = user_item_arr - user_means[:, np.newaxis]
            u, s, vt = svds(centered_matrix, k=min(self.n_factors, centered_matrix.shape[0] - 1))
            self.u = u
            self.s = np.diag(s)
            self.vt = vt
            self.user_means = user_means
            # Item factor representation: vt transposed (N_movies, n_factors)
            self.item_factors = self.vt.T

        # Normalize item factors for cosine similarity calculations
        norms = np.linalg.norm(self.item_factors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.normalized_item_factors = self.item_factors / norms

    def find_movie_index(self, query: str) -> Tuple[int, str]:
        """Resolves movie title to index via exact and fuzzy matching."""
        q_clean = query.strip().lower()

        if q_clean in self.clean_title_to_idx:
            idx = self.clean_title_to_idx[q_clean]
            return idx, self.movies_df.loc[idx, "clean_title"]

        if q_clean in self.raw_title_to_idx:
            idx = self.raw_title_to_idx[q_clean]
            return idx, self.movies_df.loc[idx, "clean_title"]

        close_clean = get_close_matches(query.strip(), self.all_clean_titles, n=1, cutoff=0.55)
        if close_clean:
            matched_title = close_clean[0]
            idx = self.clean_title_to_idx[matched_title.lower()]
            return idx, matched_title

        raise KeyError(f"Movie '{query}' not found for SVD recommendations.")

    def recommend_for_user(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """
        Recommends top-N unrated movies for a given user_id based on predicted ratings.

        Args:
            user_id: Target user ID
            n: Number of recommendations (default: 10)
        """
        # Find movies already rated by user
        user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
        rated_movie_ids = set(user_ratings["movieId"])

        unrated_movies = self.movies_df[~self.movies_df["movieId"].isin(rated_movie_ids)]

        predictions = []
        if self.use_surprise:
            for _, row in unrated_movies.iterrows():
                mid = int(row["movieId"])
                pred = self.model.predict(uid=user_id, iid=mid)
                predictions.append((mid, row["clean_title"], row["raw_title"], row["genres"], pred.est))
        else:
            if user_id in self.data.user_id_to_idx:
                u_idx = self.data.user_id_to_idx[user_id]
                pred_ratings = self.user_means[u_idx] + np.dot(
                    np.dot(self.u[u_idx, :], self.s), self.vt
                )
                for _, row in unrated_movies.iterrows():
                    mid = int(row["movieId"])
                    col_idx = self.data.movie_id_to_idx[mid]
                    score = pred_ratings[col_idx]
                    predictions.append((mid, row["clean_title"], row["raw_title"], row["genres"], score))
            else:
                global_mean = self.ratings_df["rating"].mean()
                for _, row in unrated_movies.iterrows():
                    predictions.append((int(row["movieId"]), row["clean_title"], row["raw_title"], row["genres"], global_mean))

        # Sort by predicted rating descending
        predictions.sort(key=lambda x: x[4], reverse=True)
        top_preds = predictions[:n]

        results = []
        for mid, title, raw_title, genres, est_rating in top_preds:
            results.append({
                "movieId": mid,
                "title": title,
                "raw_title": raw_title,
                "genres": genres,
                "similarity_score": round(float(est_rating), 4),
            })

        return pd.DataFrame(results)

    def recommend_by_movie(self, movie_title: str, n: int = 10) -> pd.DataFrame:
        """
        Recommends top-N movies similar in latent feature space to target movie.
        Allows SVD to directly compare alongside Content and Item-CF for a selected movie.
        """
        target_idx, matched_title = self.find_movie_index(movie_title)

        target_vec = self.normalized_item_factors[target_idx]
        sim_scores = np.dot(self.normalized_item_factors, target_vec)
        sim_scores[target_idx] = -1.0  # Exclude self

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


# Global singletons
_GLOBAL_ITEM_CF: Optional[ItemBasedCF] = None
_GLOBAL_SVD: Optional[SVDRecommender] = None


def get_item_cf_recommender(data_dir: Optional[Path | str] = None) -> ItemBasedCF:
    """Returns or initializes ItemBasedCF singleton."""
    global _GLOBAL_ITEM_CF
    if _GLOBAL_ITEM_CF is None:
        data = load_and_preprocess_data(data_dir)
        _GLOBAL_ITEM_CF = ItemBasedCF(data)
    return _GLOBAL_ITEM_CF


def get_svd_recommender(data_dir: Optional[Path | str] = None) -> SVDRecommender:
    """Returns or initializes SVDRecommender singleton."""
    global _GLOBAL_SVD
    if _GLOBAL_SVD is None:
        data = load_and_preprocess_data(data_dir)
        _GLOBAL_SVD = SVDRecommender(data)
    return _GLOBAL_SVD


def recommend_item_cf(movie_title: str, n: int = 10) -> pd.DataFrame:
    """
    Public API: Item-based CF recommendations for a movie title.
    """
    recommender = get_item_cf_recommender()
    return recommender.recommend(movie_title, n=n)


def recommend_svd(query: Union[int, str], n: int = 10) -> pd.DataFrame:
    """
    Public API: SVD matrix factorization recommendations.
    Accepts:
      - user_id (int or numeric string): returns top-N predicted unrated movies for the user.
      - movie_title (str): returns top-N latent factor similar movies for the target movie.
    """
    recommender = get_svd_recommender()

    # If query is integer or string representing integer -> recommend for user
    if isinstance(query, int) or (isinstance(query, str) and query.strip().isdigit()):
        user_id = int(query)
        return recommender.recommend_for_user(user_id, n=n)
    else:
        # Recommending by movie title in latent feature space
        return recommender.recommend_by_movie(str(query), n=n)


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 5: COLLABORATIVE FILTERING TEST")
    print("=" * 60)

    test_movie = "Toy Story"
    print(f"1. Item-Based CF Recommendations for: '{test_movie}'")
    item_recs = recommend_item_cf(test_movie, n=5)
    print(item_recs[["title", "genres", "similarity_score"]].to_string(index=False))
    print()

    print(f"2. SVD Item-Latent Recommendations for: '{test_movie}'")
    svd_item_recs = recommend_svd(test_movie, n=5)
    print(svd_item_recs[["title", "genres", "similarity_score"]].to_string(index=False))
    print()

    test_user_id = 1
    print(f"3. SVD User Rating Predictions for User ID: {test_user_id}")
    svd_user_recs = recommend_svd(test_user_id, n=5)
    print(svd_user_recs[["title", "genres", "similarity_score"]].to_string(index=False))
    print("=" * 60)
