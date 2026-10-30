"""
Data Preprocessing Module for Movie Recommendation System.

Handles:
- Title cleaning and release year extraction
- Genre parsing and tokenization
- Filtering long-tail movies (< 5 ratings)
- Constructing user-item matrix and calculating sparsity
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Fixed random seed for reproducibility across the project
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


@dataclass
class PreprocessedData:
    """Container holding cleaned datasets, matrices, and metadata."""
    movies_raw: pd.DataFrame
    ratings_raw: pd.DataFrame
    movies_clean: pd.DataFrame
    ratings_clean: pd.DataFrame
    user_item_df: pd.DataFrame
    user_item_matrix: csr_matrix
    sparsity_pct: float
    user_id_to_idx: dict[int, int]
    movie_id_to_idx: dict[int, int]
    idx_to_movie_id: dict[int, int]


def clean_titles_and_genres(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans movie titles, extracts 4-digit release year, and formats genres.

    Args:
        movies_df: Raw movies dataframe containing ['movieId', 'title', 'genres']

    Returns:
        DataFrame with added columns: 'year', 'clean_title', 'genres_list', 'genres_str'
    """
    df = movies_df.copy()

    # Extract 4-digit release year from title (e.g. "Toy Story (1995)" -> 1995)
    # Using regex that captures 4 digits inside the final parentheses
    year_extracted = df["title"].str.extract(r"\((\d{4})\)\s*$", expand=False)
    df["year"] = pd.to_numeric(year_extracted, errors="coerce")

    # Remove the trailing (YYYY) from title for clean display and fuzzy matching
    df["clean_title"] = df["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True).str.strip()

    # Store raw title for exact lookup
    df["raw_title"] = df["title"].str.strip()

    # Process genres: replace pipe '|' with space for TF-IDF, and store as list
    # Handle "(no genres listed)" by substituting an empty string
    clean_genres_str = df["genres"].replace("(no genres listed)", "").str.replace("|", " ", regex=False)
    df["genres_str"] = clean_genres_str.str.strip()
    df["genres_list"] = df["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )

    return df


def filter_by_minimum_ratings(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    min_ratings: int = 5,
    exempt_movie_ids: Optional[set] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filters out movies with fewer than min_ratings reviews to reduce extreme sparsity.
    Custom added movies in exempt_movie_ids are preserved regardless of review count.

    Args:
        ratings_df: DataFrame of ratings (userId, movieId, rating, timestamp)
        movies_df: DataFrame of movies
        min_ratings: Minimum number of ratings required per movie (default: 5)
        exempt_movie_ids: Set of movie IDs to always keep (e.g. user-added custom movies)

    Returns:
        Tuple of (filtered_ratings_df, filtered_movies_df)
    """
    movie_counts = ratings_df["movieId"].value_counts()
    valid_movie_ids = set(movie_counts[movie_counts >= min_ratings].index)
    if exempt_movie_ids:
        valid_movie_ids = valid_movie_ids.union(exempt_movie_ids)

    filtered_ratings = ratings_df[ratings_df["movieId"].isin(valid_movie_ids)].copy()
    filtered_movies = movies_df[movies_df["movieId"].isin(valid_movie_ids)].copy()

    return filtered_ratings.reset_index(drop=True), filtered_movies.reset_index(drop=True)


def build_user_item_matrices(
    ratings_df: pd.DataFrame,
    all_movies_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, csr_matrix, float, dict, dict, dict]:
    """
    Builds the user-item interaction matrix as both a DataFrame and a Scipy CSR sparse matrix.
    Also calculates the dataset sparsity percentage.
    """
    # Pivot table with users as rows and movies as columns
    user_item_df = ratings_df.pivot(index="userId", columns="movieId", values="rating").fillna(0.0)

    # Ensure all movies in all_movies_df exist as columns (e.g. cold-start zero vectors)
    if all_movies_df is not None:
        for mid in all_movies_df["movieId"]:
            if mid not in user_item_df.columns:
                user_item_df[mid] = 0.0
        user_item_df = user_item_df.reindex(columns=sorted(user_item_df.columns), fill_value=0.0)

    # Scipy sparse matrix (CSR format for fast row slicing)
    user_item_csr = csr_matrix(user_item_df.values)

    n_users = user_item_df.shape[0]
    n_movies = user_item_df.shape[1]
    n_ratings = len(ratings_df)

    total_possible_ratings = n_users * n_movies
    sparsity_pct = (1.0 - (n_ratings / total_possible_ratings)) * 100.0

    # Index mappings for fast array indexing
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_item_df.index)}
    movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(user_item_df.columns)}
    idx_to_movie_id = {idx: movie_id for idx, movie_id in enumerate(user_item_df.columns)}

    return (
        user_item_df,
        user_item_csr,
        sparsity_pct,
        user_id_to_idx,
        movie_id_to_idx,
        idx_to_movie_id,
    )


def load_and_preprocess_data(
    data_dir: Path | str | None = None,
    min_ratings: int = 5,
) -> PreprocessedData:
    """
    Complete pipeline to load raw MovieLens CSVs, clean attributes, filter noise,
    incorporate custom user-added movies, and construct sparse user-item matrices.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"
    else:
        data_dir = Path(data_dir)

    movies_path = data_dir / "movies.csv"
    ratings_path = data_dir / "ratings.csv"

    if not movies_path.exists() or not ratings_path.exists():
        raise FileNotFoundError(
            f"Missing required data files in {data_dir}. Run data/download_data.py first."
        )

    # 1. Load raw data
    raw_movies = pd.read_csv(movies_path)
    raw_ratings = pd.read_csv(ratings_path)

    # Check for custom user-added movies & ratings
    custom_movies_path = data_dir / "custom_movies.csv"
    custom_ratings_path = data_dir / "custom_ratings.csv"
    custom_movie_ids = set()

    if custom_movies_path.exists():
        try:
            custom_movies = pd.read_csv(custom_movies_path)
            if not custom_movies.empty:
                raw_movies = pd.concat([raw_movies, custom_movies], ignore_index=True)
                custom_movie_ids = set(custom_movies["movieId"].unique())
        except Exception:
            pass

    if custom_ratings_path.exists():
        try:
            custom_ratings = pd.read_csv(custom_ratings_path)
            if not custom_ratings.empty:
                raw_ratings = pd.concat([raw_ratings, custom_ratings], ignore_index=True)
        except Exception:
            pass

    # 2. Clean titles, extract years, tokenize genres
    movies_cleaned = clean_titles_and_genres(raw_movies)

    # 3. Filter movies with < min_ratings (preserving custom movies)
    ratings_filtered, movies_filtered = filter_by_minimum_ratings(
        raw_ratings, movies_cleaned, min_ratings=min_ratings, exempt_movie_ids=custom_movie_ids
    )

    # 4. Build user-item sparse matrix & compute sparsity
    (
        user_item_df,
        user_item_csr,
        sparsity_pct,
        user_id_to_idx,
        movie_id_to_idx,
        idx_to_movie_id,
    ) = build_user_item_matrices(ratings_filtered, movies_filtered)
    return PreprocessedData(
        movies_raw=raw_movies,
        ratings_raw=raw_ratings,
        movies_clean=movies_filtered,
        ratings_clean=ratings_filtered,
        user_item_df=user_item_df,
        user_item_matrix=user_item_csr,
        sparsity_pct=sparsity_pct,
        user_id_to_idx=user_id_to_idx,
        movie_id_to_idx=movie_id_to_idx,
        idx_to_movie_id=idx_to_movie_id,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 2: PREPROCESSING CHECKPOINT")
    print("=" * 60)

    data = load_and_preprocess_data()

    print(f"Dataset Shape BEFORE Cleaning:")
    print(f"  - Movies DataFrame:  {data.movies_raw.shape[0]:,} rows x {data.movies_raw.shape[1]} cols")
    print(f"  - Ratings DataFrame: {data.ratings_raw.shape[0]:,} rows x {data.ratings_raw.shape[1]} cols")
    print()
    print(f"Dataset Shape AFTER Cleaning (min_ratings >= 5):")
    print(f"  - Movies DataFrame:  {data.movies_clean.shape[0]:,} rows x {data.movies_clean.shape[1]} cols")
    print(f"  - Ratings DataFrame: {data.ratings_clean.shape[0]:,} rows x {data.ratings_clean.shape[1]} cols")
    print()
    print(f"User-Item Interaction Matrix:")
    print(f"  - Dimensions:        {data.user_item_df.shape[0]} unique users x {data.user_item_df.shape[1]} unique movies")
    print(f"  - Non-zero ratings:  {len(data.ratings_clean):,}")
    print(f"  - Matrix Sparsity:   {data.sparsity_pct:.2f}%")
    print("=" * 60)
