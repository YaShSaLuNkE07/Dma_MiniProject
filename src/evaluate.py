"""
Evaluation Module for Movie Recommendation System.

Conducts an 80/20 train/test evaluation (fixed random seed 42) across all 4 methods:
1. Content-Based Filtering
2. Item-Based Collaborative Filtering
3. SVD Matrix Factorization
4. Hybrid Recommender (alpha = 0.5)

Metrics computed:
- Rating Prediction: RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error)
- Top-10 Ranking: Precision@10 and Recall@10 (Threshold: rating >= 4.0)

Generates:
- results/metrics.csv
- results/comparison.png (Grouped bar chart)
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to sys.path if not present
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from src.preprocess import load_and_preprocess_data, RANDOM_SEED

# Surprise import with fallback
try:
    from surprise import Dataset, Reader, SVD
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    from scipy.sparse.linalg import svds

np.random.seed(RANDOM_SEED)


def compute_rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Calculates RMSE and MAE between true and predicted ratings."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return round(rmse, 4), round(mae, 4)


def run_evaluation(
    data_dir: Path | str | None = None,
    results_dir: Path | str | None = None,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """
    Executes train/test benchmark on Content-based, Item-CF, SVD, and Hybrid.
    Saves metrics.csv and comparison.png.
    """
    if results_dir is None:
        results_dir = _project_root / "results"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading preprocessed dataset...")
    data = load_and_preprocess_data(data_dir)
    ratings_df = data.ratings_clean
    movies_df = data.movies_clean.reset_index(drop=True)

    # 1. 80/20 Train-Test Split (fixed seed 42)
    print("Performing 80/20 train/test split (seed=42)...")
    train_ratings, test_ratings = train_test_split(
        ratings_df, test_size=0.20, random_state=RANDOM_SEED, stratify=None
    )
    print(f"Train set: {len(train_ratings):,} ratings | Test set: {len(test_ratings):,} ratings")

    # Catalog & Matrix Dimensions
    all_movie_ids = list(movies_df["movieId"].values)
    movie_id_to_idx = {mid: i for i, mid in enumerate(all_movie_ids)}
    n_movies = len(all_movie_ids)

    # Compute Global & User Training Means
    global_mean = float(train_ratings["rating"].mean())
    user_train_means = train_ratings.groupby("userId")["rating"].mean().to_dict()

    # Pre-build Content Similarity Matrix (Genres TF-IDF)
    print("Building Content similarity matrix...")
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b[a-zA-Z0-9_-]+\b", stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_df["genres_str"])
    content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Pre-build Item-Item CF Matrix on TRAIN ratings only
    print("Building Item-Item CF similarity matrix on train split...")
    train_pivot = train_ratings.pivot(index="userId", columns="movieId", values="rating")
    # Reindex columns to match movies_df
    train_pivot = train_pivot.reindex(columns=all_movie_ids, fill_value=0.0).fillna(0.0)
    item_user_train = train_pivot.T.values
    item_cf_sim = cosine_similarity(item_user_train, item_user_train)
    # Zero out diagonal to avoid self-similarity in rating aggregation
    np.fill_diagonal(item_cf_sim, 0.0)
    # Zero out diagonal in content_sim copy for prediction
    content_sim_pred = content_sim.copy()
    np.fill_diagonal(content_sim_pred, 0.0)

    # Train SVD Model on TRAIN split only
    print("Fitting SVD on train split...")
    if SURPRISE_AVAILABLE:
        reader = Reader(rating_scale=(0.5, 5.0))
        sp_train_data = Dataset.load_from_df(train_ratings[["userId", "movieId", "rating"]], reader)
        sp_trainset = sp_train_data.build_full_trainset()
        svd_model = SVD(n_factors=50, random_state=RANDOM_SEED, n_epochs=20)
        svd_model.fit(sp_trainset)
    else:
        # Scipy fallback on train matrix
        u_arr = train_pivot.values.astype(float)
        u_means = np.array([row[row > 0].mean() if np.any(row > 0) else global_mean for row in u_arr])
        centered_train = u_arr - u_means[:, np.newaxis]
        svd_u, svd_s, svd_vt = svds(centered_train, k=min(50, centered_train.shape[0] - 1))
        train_user_to_row = {uid: idx for idx, uid in enumerate(train_pivot.index)}

    # Map user ratings in training set for fast item-CF and content rating prediction
    user_train_ratings: Dict[int, Dict[int, float]] = {}
    for _, row in train_ratings.iterrows():
        u = int(row["userId"])
        m = int(row["movieId"])
        r = float(row["rating"])
        if u not in user_train_ratings:
            user_train_ratings[u] = {}
        user_train_ratings[u][m] = r

    # -------------------------------------------------------------
    # 2. Rating Prediction on Test Set (RMSE & MAE)
    # -------------------------------------------------------------
    print("Evaluating rating predictions on test set...")
    y_true = []
    y_pred_svd = []
    y_pred_item_cf = []
    y_pred_content = []
    y_pred_hybrid = []

    for _, row in test_ratings.iterrows():
        u = int(row["userId"])
        m = int(row["movieId"])
        r_actual = float(row["rating"])
        y_true.append(r_actual)

        u_mean = user_train_means.get(u, global_mean)
        rated_dict = user_train_ratings.get(u, {})

        # SVD prediction
        if SURPRISE_AVAILABLE:
            svd_est = svd_model.predict(u, m).est
        else:
            if u in train_user_to_row and m in movie_id_to_idx:
                r_idx = train_user_to_row[u]
                c_idx = movie_id_to_idx[m]
                svd_est = u_means[r_idx] + np.dot(np.dot(svd_u[r_idx, :], np.diag(svd_s)), svd_vt[:, c_idx])
            else:
                svd_est = u_mean
        svd_est = float(np.clip(svd_est, 0.5, 5.0))
        y_pred_svd.append(svd_est)

        # Target movie index in item matrices
        target_idx = movie_id_to_idx.get(m, None)

        if target_idx is not None and len(rated_dict) > 0:
            rated_indices = [movie_id_to_idx[rm] for rm in rated_dict if rm in movie_id_to_idx]
            rated_vals = np.array([rated_dict[rm] for rm in rated_dict if rm in movie_id_to_idx])
            rated_diffs = rated_vals - u_mean

            # Item CF prediction
            cf_sims = item_cf_sim[target_idx, rated_indices]
            pos_mask_cf = cf_sims > 0
            if np.any(pos_mask_cf):
                sim_weights = cf_sims[pos_mask_cf]
                cf_pred = u_mean + (np.sum(sim_weights * rated_diffs[pos_mask_cf]) / np.sum(sim_weights))
            else:
                cf_pred = u_mean

            # Content prediction
            cont_sims = content_sim_pred[target_idx, rated_indices]
            pos_mask_cont = cont_sims > 0
            if np.any(pos_mask_cont):
                sim_weights_c = cont_sims[pos_mask_cont]
                cont_pred = u_mean + (np.sum(sim_weights_c * rated_diffs[pos_mask_cont]) / np.sum(sim_weights_c))
            else:
                cont_pred = u_mean
        else:
            cf_pred = u_mean
            cont_pred = u_mean

        cf_pred = float(np.clip(cf_pred, 0.5, 5.0))
        cont_pred = float(np.clip(cont_pred, 0.5, 5.0))
        hyb_pred = float(np.clip(alpha * cont_pred + (1.0 - alpha) * cf_pred, 0.5, 5.0))

        y_pred_item_cf.append(cf_pred)
        y_pred_content.append(cont_pred)
        y_pred_hybrid.append(hyb_pred)

    y_true_arr = np.array(y_true)
    rmse_svd, mae_svd = compute_rmse_mae(y_true_arr, np.array(y_pred_svd))
    rmse_cf, mae_cf = compute_rmse_mae(y_true_arr, np.array(y_pred_item_cf))
    rmse_cont, mae_cont = compute_rmse_mae(y_true_arr, np.array(y_pred_content))
    rmse_hyb, mae_hyb = compute_rmse_mae(y_true_arr, np.array(y_pred_hybrid))

    # -------------------------------------------------------------
    # 3. Top-10 Ranking Evaluation (Precision@10, Recall@10)
    # Relevant item = rating in test set >= 4.0
    # -------------------------------------------------------------
    print("Evaluating Top-10 Ranking (Precision@10, Recall@10)...")
    test_user_groups = test_ratings.groupby("userId")

    prec_cont_list, rec_cont_list = [], []
    prec_cf_list, rec_cf_list = [], []
    prec_svd_list, rec_svd_list = [], []
    prec_hyb_list, rec_hyb_list = [], []

    for u, user_test_df in test_user_groups:
        relevant_test_movies = set(user_test_df[user_test_df["rating"] >= 4.0]["movieId"])
        if len(relevant_test_movies) == 0:
            continue  # Only evaluate ranking on users with at least 1 ground truth relevant item

        test_movie_ids = list(user_test_df["movieId"].values)
        if len(test_movie_ids) < 10:
            # Need candidate pool of at least 10 items for Precision@10 evaluation
            # Augment with unrated items from catalog
            train_rated = set(user_train_ratings.get(u, {}).keys())
            unrated = [mid for mid in all_movie_ids if mid not in train_rated and mid not in test_movie_ids]
            candidate_pool = test_movie_ids + unrated[: (100 - len(test_movie_ids))]
        else:
            candidate_pool = test_movie_ids

        cand_indices = [movie_id_to_idx[mid] for mid in candidate_pool if mid in movie_id_to_idx]
        user_rated_dict = user_train_ratings.get(u, {})

        if len(user_rated_dict) == 0:
            continue

        rated_indices = [movie_id_to_idx[m] for m in user_rated_dict if m in movie_id_to_idx]
        rated_vals = np.array([user_rated_dict[m] for m in user_rated_dict if m in movie_id_to_idx])

        # A. SVD Scores for Candidate Pool
        if SURPRISE_AVAILABLE:
            svd_scores = [svd_model.predict(u, all_movie_ids[ci]).est for ci in cand_indices]
        else:
            u_row = train_user_to_row.get(u, None)
            if u_row is not None:
                svd_scores = [
                    u_means[u_row] + np.dot(np.dot(svd_u[u_row, :], np.diag(svd_s)), svd_vt[:, ci])
                    for ci in cand_indices
                ]
            else:
                svd_scores = [global_mean] * len(cand_indices)

        # B. Content Scores for Candidate Pool (similarity to user's highly rated items)
        # Weight by user's positive rating (>= 3.5)
        pos_rated_indices = [
            movie_id_to_idx[m] for m, r in user_rated_dict.items() if m in movie_id_to_idx and r >= 3.5
        ]
        if len(pos_rated_indices) > 0:
            content_profile_scores = np.mean(content_sim[cand_indices][:, pos_rated_indices], axis=1)
            cf_profile_scores = np.mean(item_cf_sim[cand_indices][:, pos_rated_indices], axis=1)
        else:
            content_profile_scores = np.mean(content_sim[cand_indices][:, rated_indices], axis=1)
            cf_profile_scores = np.mean(item_cf_sim[cand_indices][:, rated_indices], axis=1)

        # C. Hybrid Scores
        # Min-max scale profile scores
        c_denom = np.ptp(content_profile_scores) + 1e-9
        c_norm = (content_profile_scores - np.min(content_profile_scores)) / c_denom

        cf_denom = np.ptp(cf_profile_scores) + 1e-9
        cf_norm = (cf_profile_scores - np.min(cf_profile_scores)) / cf_denom

        hybrid_profile_scores = alpha * c_norm + (1.0 - alpha) * cf_norm

        # Helper to compute Precision@10 and Recall@10
        def evaluate_top10(scores: np.ndarray | List[float]) -> Tuple[float, float]:
            top10_cand_idx = np.argsort(scores)[::-1][:10]
            rec_movies = [all_movie_ids[cand_indices[i]] for i in top10_cand_idx]
            hits = len(set(rec_movies) & relevant_test_movies)
            prec = hits / 10.0
            rec = hits / float(len(relevant_test_movies))
            return prec, rec

        p_cont, r_cont = evaluate_top10(content_profile_scores)
        p_cf, r_cf = evaluate_top10(cf_profile_scores)
        p_svd, r_svd = evaluate_top10(svd_scores)
        p_hyb, r_hyb = evaluate_top10(hybrid_profile_scores)

        prec_cont_list.append(p_cont)
        rec_cont_list.append(r_cont)
        prec_cf_list.append(p_cf)
        rec_cf_list.append(r_cf)
        prec_svd_list.append(p_svd)
        rec_svd_list.append(r_svd)
        prec_hyb_list.append(p_hyb)
        rec_hyb_list.append(r_hyb)

    # 4. Construct Benchmark Summary Table
    metrics_data = [
        {
            "Method": "Content-Based",
            "RMSE": rmse_cont,
            "MAE": mae_cont,
            "Precision@10": round(float(np.mean(prec_cont_list)), 4),
            "Recall@10": round(float(np.mean(rec_cont_list)), 4),
        },
        {
            "Method": "Item-Based CF",
            "RMSE": rmse_cf,
            "MAE": mae_cf,
            "Precision@10": round(float(np.mean(prec_cf_list)), 4),
            "Recall@10": round(float(np.mean(rec_cf_list)), 4),
        },
        {
            "Method": "SVD",
            "RMSE": rmse_svd,
            "MAE": mae_svd,
            "Precision@10": round(float(np.mean(prec_svd_list)), 4),
            "Recall@10": round(float(np.mean(rec_svd_list)), 4),
        },
        {
            "Method": "Hybrid",
            "RMSE": rmse_hyb,
            "MAE": mae_hyb,
            "Precision@10": round(float(np.mean(prec_hyb_list)), 4),
            "Recall@10": round(float(np.mean(rec_hyb_list)), 4),
        },
    ]

    metrics_df = pd.DataFrame(metrics_data)
    csv_path = results_dir / "metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark metrics to: {csv_path}")

    # 5. Generate Grouped Bar Chart
    print("Generating grouped bar chart...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    methods = metrics_df["Method"]
    x = np.arange(len(methods))
    width = 0.35

    # Subplot 1: Error Metrics (Lower is better)
    rects1 = ax1.bar(x - width / 2, metrics_df["RMSE"], width, label="RMSE", color="#2b5c8f")
    rects2 = ax1.bar(x + width / 2, metrics_df["MAE"], width, label="MAE", color="#6baed6")
    ax1.set_title("Rating Prediction Error (Lower is Better)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Error Score", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=10, fontweight="semibold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_ylim(0, max(metrics_df["RMSE"]) * 1.25)

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.4f}", (rect.get_x() + rect.get_width() / 2, h + 0.02),
                     ha="center", va="bottom", fontsize=8.5, fontweight="semibold")

    # Subplot 2: Ranking Metrics (Higher is better)
    rects3 = ax2.bar(x - width / 2, metrics_df["Precision@10"], width, label="Precision@10", color="#238b45")
    rects4 = ax2.bar(x + width / 2, metrics_df["Recall@10"], width, label="Recall@10", color="#74c476")
    ax2.set_title("Top-10 Ranking Performance (Higher is Better)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_ylabel("Metric Score", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=10, fontweight="semibold")
    ax2.legend(loc="upper right", frameon=True)
    ax2.set_ylim(0, max(metrics_df["Recall@10"]) * 1.35)

    for rect in rects3 + rects4:
        h = rect.get_height()
        ax2.annotate(f"{h:.4f}", (rect.get_x() + rect.get_width() / 2, h + 0.005),
                     ha="center", va="bottom", fontsize=8.5, fontweight="semibold")

    plt.tight_layout()
    chart_path = results_dir / "comparison.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"Saved grouped bar chart to: {chart_path}")

    return metrics_df


if __name__ == "__main__":
    print("=" * 65)
    print("PHASE 7: EVALUATION & BENCHMARKING (80/20 Train/Test Split, Seed 42)")
    print("=" * 65)

    results_table = run_evaluation()

    print("\nFINAL COMPARISON TABLE:")
    print("=" * 65)
    print(results_table.to_string(index=False))
    print("=" * 65)
