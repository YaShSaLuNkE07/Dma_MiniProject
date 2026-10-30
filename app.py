"""
Streamlit Web Application: Movie Recommendation System.

Modern Cinema-Themed Interface featuring:
- Hero Movie Showcase with community ratings and genre badges
- Quick-pick buttons for iconic movies
- Tabbed view: Hybrid Premier, Content-Based, Item-CF, SVD, and 4-Way Side-by-Side Arena
- Rich visual cards with Match % scores, star ratings, and genre tags
- Evaluation benchmark viewer with offline metrics and charts
"""

import sys
from pathlib import Path

# Add project root to sys.path
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import streamlit as st

from src.collaborative import ItemBasedCF, SVDRecommender
from src.content_based import ContentRecommender
from src.hybrid import HybridRecommender
from src.preprocess import PreprocessedData, load_and_preprocess_data

# Page configuration
st.set_page_config(
    page_title="CineMatch Studio | Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Genre to emoji mapping for visual flair
GENRE_ICONS = {
    "Action": "💥 Action",
    "Adventure": "🧭 Adventure",
    "Animation": "🎨 Animation",
    "Children": "🧸 Children",
    "Comedy": "😂 Comedy",
    "Crime": "🕵️ Crime",
    "Documentary": "📹 Documentary",
    "Drama": "🎭 Drama",
    "Fantasy": "🧙 Fantasy",
    "Film-Noir": "🕶️ Film-Noir",
    "Horror": "👻 Horror",
    "Musical": "🎵 Musical",
    "Mystery": "🔍 Mystery",
    "Romance": "💖 Romance",
    "Sci-Fi": "🚀 Sci-Fi",
    "Thriller": "⚡ Thriller",
    "War": "⚔️ War",
    "Western": "🤠 Western",
    "IMAX": "📽️ IMAX",
}


def format_genre_badge(genre_name: str) -> str:
    """Returns an HTML badge for a genre with its icon."""
    label = GENRE_ICONS.get(genre_name, genre_name)
    return f'<span class="genre-pill">{label}</span>'


# Custom CSS for modern cinema aesthetic
st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Branding Header */
    .hero-banner {
        background: linear-gradient(135deg, rgba(20, 24, 39, 0.95), rgba(10, 14, 26, 0.98));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .hero-label {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #60a5fa;
        margin-bottom: 6px;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 8px;
        line-height: 1.2;
    }
    .hero-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        align-items: center;
        margin-top: 10px;
        font-size: 0.92rem;
        color: #cbd5e1;
    }
    .hero-stat {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
    }

    /* Genre Pill Badges */
    .genre-pill {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 2px 9px;
        font-size: 0.76rem;
        font-weight: 600;
        margin: 2px 3px 2px 0;
    }

    /* Cinema Cards for Recommendations */
    .movie-card {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.85));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 14px;
        transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .movie-card:hover {
        transform: translateY(-3px);
        border-color: rgba(96, 165, 250, 0.5);
        box-shadow: 0 12px 25px -8px rgba(37, 99, 235, 0.25);
    }
    .card-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .rank-badge {
        font-weight: 800;
        font-size: 0.82rem;
        padding: 2px 10px;
        border-radius: 20px;
    }
    .rank-gold   { background: #f59e0b; color: #000; box-shadow: 0 0 10px rgba(245, 158, 11, 0.5); }
    .rank-silver { background: #94a3b8; color: #000; }
    .rank-bronze { background: #d97706; color: #fff; }
    .rank-other  { background: rgba(255, 255, 255, 0.1); color: #cbd5e1; }

    .match-badge {
        font-weight: 700;
        font-size: 0.82rem;
        color: #10b981;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 2px 8px;
        border-radius: 12px;
    }

    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
        line-height: 1.3;
    }
    .card-meta {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-bottom: 8px;
    }

    /* Match Progress Bar */
    .progress-bar-bg {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        height: 6px;
        overflow: hidden;
        margin-top: 10px;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, #3b82f6, #10b981);
    }

    /* Column Headers */
    .column-header {
        border-radius: 10px;
        padding: 10px 14px;
        text-align: center;
        font-weight: 700;
        font-size: 0.95rem;
        color: #ffffff;
        margin-bottom: 12px;
    }
    .col-content { background: linear-gradient(135deg, #0284c7, #0369a1); }
    .col-itemcf  { background: linear-gradient(135deg, #059669, #047857); }
    .col-svd     { background: linear-gradient(135deg, #ea580c, #c2410c); }
    .col-hybrid  { background: linear-gradient(135deg, #7c3aed, #6d28d9); }

    /* Clean quick pick chips */
    .quick-pick-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #94a3b8;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading and preprocessing MovieLens dataset...")
def get_dataset() -> PreprocessedData:
    """Loads preprocessed data and returns cached container."""
    return load_and_preprocess_data()


@st.cache_resource(show_spinner="Initializing recommendation models...")
def get_all_models(_data: PreprocessedData):
    """Initializes and caches all 4 recommendation engines."""
    content_rec = ContentRecommender(_data.movies_clean)
    item_cf_rec = ItemBasedCF(_data)
    svd_rec = SVDRecommender(_data)
    hybrid_rec = HybridRecommender(content_rec, item_cf_rec)
    return content_rec, item_cf_rec, svd_rec, hybrid_rec


# Load dataset and models
data = get_dataset()
content_model, item_cf_model, svd_model, hybrid_model = get_all_models(data)

# Precompute average ratings and vote counts for rich movie cards
rating_stats = (
    data.ratings_clean.groupby("movieId")
    .agg(avg_rating=("rating", "mean"), vote_count=("rating", "count"))
    .reset_index()
)

display_movies = data.movies_clean.merge(rating_stats, on="movieId", how="left").copy()
display_movies["avg_rating"] = display_movies["avg_rating"].fillna(0.0)
display_movies["vote_count"] = display_movies["vote_count"].fillna(0).astype(int)

# Create clean display labels
display_movies["display_label"] = display_movies.apply(
    lambda r: f"{r['clean_title']} ({int(r['year']) if pd.notnull(r['year']) else 'N/A'})",
    axis=1,
)
display_movies = display_movies.sort_values("clean_title").reset_index(drop=True)
movie_labels = display_movies["display_label"].tolist()

# Fast row lookup dictionary
label_to_row = {row["display_label"]: row for _, row in display_movies.iterrows()}
movie_id_to_row = {int(row["movieId"]): row for _, row in display_movies.iterrows()}

# State initialization for movie selection
if "selected_movie_label" not in st.session_state:
    # Default to Toy Story (1995)
    default_toy_story = next((lbl for lbl in movie_labels if lbl.startswith("Toy Story (")), movie_labels[0])
    st.session_state["selected_movie_label"] = default_toy_story


def set_selected_movie(label: str):
    """Callback for quick pick buttons."""
    st.session_state["selected_movie_label"] = label


# Sidebar Controls
with st.sidebar:
    st.markdown("### ⚙️ Engine Controls")

    top_n = st.slider(
        "Recommendations Count (N)",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
        help="Number of recommendations returned by each algorithm.",
    )

    st.markdown("#### ⚖️ Hybrid Balance (α)")
    st.caption("Balance semantic genre match vs. user co-rating popularity:")

    # Quick presets for alpha
    preset_cols = st.columns(3)
    if preset_cols[0].button("🎭 0.8\nGenre", use_container_width=True):
        st.session_state["alpha_val"] = 0.80
    if preset_cols[1].button("🎯 0.5\nBlend", use_container_width=True):
        st.session_state["alpha_val"] = 0.50
    if preset_cols[2].button("👥 0.2\nCrowd", use_container_width=True):
        st.session_state["alpha_val"] = 0.20

    if "alpha_val" not in st.session_state:
        st.session_state["alpha_val"] = 0.50

    alpha = st.slider(
        "Custom Alpha Weight",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state["alpha_val"],
        step=0.05,
        key="alpha_slider",
        help="1.0 = Pure Content-Based | 0.0 = Pure Collaborative Filtering",
    )
    st.info(f"**α = {alpha:.2f}** → {int(alpha*100)}% Content + {int((1-alpha)*100)}% Collaborative")

    st.markdown("---")
    st.markdown("### 📊 MovieLens Catalog Stats")
    st.markdown(
        f"""
        - 👥 **Users:** {data.user_item_df.shape[0]:,}
        - 🎬 **Active Movies:** {data.user_item_df.shape[1]:,}
        - ⭐ **Verified Ratings:** {len(data.ratings_clean):,}
        - 🕸️ **Matrix Sparsity:** `{data.sparsity_pct:.2f}%`
        """
    )

    st.markdown("---")
    show_eval = st.checkbox("Show Model Benchmark & Metrics", value=False)


# Main Content Area
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <span style="font-size: 2.2rem;">🎬</span>
        <div>
            <h1 style="margin: 0; font-size: 2.0rem; font-weight: 800; background: linear-gradient(90deg, #60a5fa, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                CineMatch Movie Recommender
            </h1>
            <p style="margin: 0; color: #94a3b8; font-size: 0.95rem;">
                Explore and compare Content-Based, Item-Based Collaborative, SVD Matrix Factorization, and Hybrid filtering.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick Pick Cinema Row
st.markdown('<div class="quick-pick-title">🍿 Popular Quick Picks:</div>', unsafe_allow_html=True)
qp_cols = st.columns(6)

quick_picks = [
    ("🧸 Toy Story", "Toy Story (1995)"),
    ("🦇 The Dark Knight", "Dark Knight, The (2008)"),
    ("🚢 Titanic", "Titanic (1997)"),
    ("🌀 Inception", "Inception (2010)"),
    ("🍫 Forrest Gump", "Forrest Gump (1994)"),
    ("🕶️ The Matrix", "Matrix, The (1999)"),
]

for col, (btn_name, target_lbl) in zip(qp_cols, quick_picks):
    # Match against available labels
    matched = next((lbl for lbl in movie_labels if target_lbl.lower() in lbl.lower()), None)
    if matched:
        if col.button(btn_name, use_container_width=True, key=f"qp_{btn_name}"):
            st.session_state["selected_movie_label"] = matched
            st.rerun()

# Search and Selection Row
selected_label = st.selectbox(
    "Or search from 3,650+ movies in the catalog:",
    options=movie_labels,
    index=movie_labels.index(st.session_state["selected_movie_label"])
    if st.session_state["selected_movie_label"] in movie_labels
    else 0,
    key="movie_selector",
)

# Update session state if dropdown changed
if selected_label != st.session_state["selected_movie_label"]:
    st.session_state["selected_movie_label"] = selected_label

# =========================================================================
# Expandable "Add a New Movie to Catalog" section
# =========================================================================
custom_movies_file = _project_root / "data" / "custom_movies.csv"
custom_ratings_file = _project_root / "data" / "custom_ratings.csv"

with st.expander("➕ Can't find your movie? Add a new movie to the catalog!"):
    st.markdown(
        "Enter a custom movie title, release year, and genres to add it to the recommender catalog. "
        "The system will compute its TF-IDF semantic profile and immediately generate recommendations!"
    )
    with st.form("add_movie_form", clear_on_submit=False):
        col_t, col_y = st.columns([3, 1])
        new_movie_title = col_t.text_input(
            "Movie Title",
            placeholder="e.g. Oppenheimer, Dune: Part Two, Barbie",
        )
        new_movie_year = col_y.number_input(
            "Release Year", min_value=1900, max_value=2030, value=2023, step=1
        )

        available_genres = list(GENRE_ICONS.keys())
        new_movie_genres = st.multiselect(
            "Genres (Select at least one):",
            options=available_genres,
            default=["Drama"],
        )

        col_rate, col_btn = st.columns([2, 1])
        your_rating = col_rate.slider(
            "Your Initial Rating for this Movie:",
            min_value=1.0,
            max_value=5.0,
            value=4.5,
            step=0.5,
        )
        col_btn.write("")
        col_btn.write("")
        submitted = col_btn.form_submit_button(
            "✨ Add & Recommend", type="primary", use_container_width=True
        )

        if submitted:
            if not new_movie_title.strip():
                st.error("Please enter a movie title.")
            elif not new_movie_genres:
                st.error("Please select at least one genre.")
            else:
                clean_t = new_movie_title.strip()
                formatted_raw = f"{clean_t} ({int(new_movie_year)})"
                genres_joined = "|".join(new_movie_genres)

                # Determine next unique movieId
                existing_custom_ids = []
                if custom_movies_file.exists():
                    try:
                        c_df = pd.read_csv(custom_movies_file)
                        existing_custom_ids = c_df["movieId"].tolist()
                    except Exception:
                        pass

                base_id = max([200000, data.movies_raw["movieId"].max()] + existing_custom_ids)
                new_mid = int(base_id + 1)

                # Append to custom_movies.csv
                new_movie_record = pd.DataFrame([{
                    "movieId": new_mid,
                    "title": formatted_raw,
                    "genres": genres_joined,
                }])
                if not custom_movies_file.exists():
                    new_movie_record.to_csv(custom_movies_file, index=False)
                else:
                    new_movie_record.to_csv(custom_movies_file, mode="a", header=False, index=False)

                # Append initial rating to custom_ratings.csv
                new_rating_record = pd.DataFrame([{
                    "userId": 999,
                    "movieId": new_mid,
                    "rating": float(your_rating),
                    "timestamp": int(pd.Timestamp.now().timestamp()),
                }])
                if not custom_ratings_file.exists():
                    new_rating_record.to_csv(custom_ratings_file, index=False)
                else:
                    new_rating_record.to_csv(custom_ratings_file, mode="a", header=False, index=False)

                # Clear Streamlit caches and re-index dataset
                st.cache_data.clear()
                st.cache_resource.clear()
                st.session_state["selected_movie_label"] = formatted_raw
                st.success(f"🎉 Added '{formatted_raw}' to catalog! Generating recommendations...")
                st.rerun()

    # Option to clear added custom movies
    if custom_movies_file.exists():
        if st.button("🗑️ Reset All Custom Added Movies", help="Clears user-added movies and resets to default catalog"):
            custom_movies_file.unlink(missing_ok=True)
            custom_ratings_file.unlink(missing_ok=True)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state["selected_movie_label"] = default_toy_story
            st.rerun()

# Retrieve current movie details
current_movie = label_to_row[selected_label]
target_query = current_movie["raw_title"]

# Render Hero Movie Showcase
genres_html = " ".join([format_genre_badge(g) for g in current_movie["genres_list"]])
st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-label">CURRENTLY EXPLORING MOVIE</div>
        <div class="hero-title">{current_movie['clean_title']} <span style="font-size: 1.4rem; color: #94a3b8; font-weight: 500;">({int(current_movie['year']) if pd.notnull(current_movie['year']) else 'N/A'})</span></div>
        <div style="margin-top: 6px;">{genres_html}</div>
        <div class="hero-meta">
            <span class="hero-stat">⭐ <b>{current_movie['avg_rating']:.2f}</b> / 5.0 Rating</span>
            <span class="hero-stat">👥 <b>{current_movie['vote_count']}</b> Community Reviews</span>
            <span class="hero-stat">🆔 Movie ID #{current_movie['movieId']}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Cold-start alert for movies with few or zero community reviews
if current_movie["vote_count"] <= 1:
    st.info(
        "❄️ **Cold-Start Active**: This newly added movie has limited historical community reviews. "
        "The system leverages **Content-Based semantic matching** to recommend titles sharing its genre DNA!"
    )


def render_movie_card(rank: int, row: pd.Series, score_val: float, score_label: str = "Match"):
    """Renders a single movie card with badges, ratings, and match percentage."""
    mid = int(row["movieId"])
    extra_info = movie_id_to_row.get(mid, None)
    avg_r = extra_info["avg_rating"] if extra_info is not None else 0.0
    v_cnt = extra_info["vote_count"] if extra_info is not None else 0
    year_val = int(extra_info["year"]) if extra_info is not None and pd.notnull(extra_info["year"]) else "N/A"

    # Match percentage (scale to 0-100%)
    match_pct = int(np.clip(score_val * 100, 5, 99))

    # Rank badge style
    if rank == 1:
        rank_badge = '<span class="rank-badge rank-gold">🥇 #1</span>'
    elif rank == 2:
        rank_badge = '<span class="rank-badge rank-silver">🥈 #2</span>'
    elif rank == 3:
        rank_badge = '<span class="rank-badge rank-bronze">🥉 #3</span>'
    else:
        rank_badge = f'<span class="rank-badge rank-other">#{rank}</span>'

    # Build genre pills
    genres_list = row["genres"].split("|") if isinstance(row["genres"], str) else []
    genres_badges = " ".join([format_genre_badge(g) for g in genres_list[:3]])

    card_html = f"""
    <div class="movie-card">
        <div class="card-top">
            {rank_badge}
            <span class="match-badge">{match_pct}% {score_label}</span>
        </div>
        <div class="card-title">{row['title']}</div>
        <div class="card-meta">
            📅 {year_val} &nbsp;•&nbsp; ⭐ {avg_r:.1f}/5 ({v_cnt} reviews)
        </div>
        <div style="margin-bottom: 6px;">{genres_badges}</div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width: {match_pct}%;"></div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


# Compute recommendations for all four engines
with st.spinner("Finding best movie matches..."):
    content_recs = content_model.recommend(target_query, n=top_n)
    item_cf_recs = item_cf_model.recommend(target_query, n=top_n)
    svd_recs = svd_model.recommend_by_movie(target_query, n=top_n)
    hybrid_recs = hybrid_model.recommend(target_query, n=top_n, alpha=alpha)

# Interactive Tabbed Recommendation View
tabs = st.tabs([
    "🌟 Hybrid Premier (Recommended)",
    "🎭 Content-Based (Genres)",
    "👥 Audience Loved (Item-CF)",
    "🔮 SVD Latent",
    "📊 4-Way Comparison Arena",
])

# Tab 1: Hybrid Premier (Grid of 2 or 3 cards per row for easy browsing)
with tabs[0]:
    st.markdown(
        f"**The Hybrid Premier Engine** fuses semantic genre attributes ({int(alpha*100)}%) with audience co-rating patterns ({int((1-alpha)*100)}%) for the most well-rounded recommendations."
    )
    cols = st.columns(2)
    for idx, (_, row) in enumerate(hybrid_recs.iterrows(), start=1):
        with cols[(idx - 1) % 2]:
            render_movie_card(idx, row, row["similarity_score"], "Hybrid Match")

# Tab 2: Content-Based
with tabs[1]:
    st.markdown(
        "**Content-Based Filtering** analyzes the TF-IDF representation of movie genres to uncover thematic and tonal twins."
    )
    cols = st.columns(2)
    for idx, (_, row) in enumerate(content_recs.iterrows(), start=1):
        with cols[(idx - 1) % 2]:
            render_movie_card(idx, row, row["similarity_score"], "Genre Match")

# Tab 3: Item-Based CF
with tabs[2]:
    st.markdown(
        "**Item-Based Collaborative Filtering** examines user viewing and rating vectors. These are titles frequently enjoyed by the same audiences."
    )
    cols = st.columns(2)
    for idx, (_, row) in enumerate(item_cf_recs.iterrows(), start=1):
        with cols[(idx - 1) % 2]:
            render_movie_card(idx, row, row["similarity_score"], "Crowd Co-Rating")

# Tab 4: SVD Latent
with tabs[3]:
    st.markdown(
        "**SVD Matrix Factorization** maps movies into a 50-dimensional latent factor space, capturing subtle unobserved taste patterns."
    )
    cols = st.columns(2)
    for idx, (_, row) in enumerate(svd_recs.iterrows(), start=1):
        with cols[(idx - 1) % 2]:
            render_movie_card(idx, row, row["similarity_score"], "Latent Similarity")

# Tab 5: Side-by-Side Arena (All 4 methods)
with tabs[4]:
    st.markdown("##### 🔬 Comparative Evaluation Matrix (All 4 Methods Side-by-Side)")
    c1, c2, c3, c4 = st.columns(4)

    def render_compact_column(col, header_title, header_class, rec_df, score_type):
        with col:
            col.markdown(f'<div class="column-header {header_class}">{header_title}</div>', unsafe_allow_html=True)
            for r_idx, (_, r_row) in enumerate(rec_df.iterrows(), start=1):
                render_movie_card(r_idx, r_row, r_row["similarity_score"], score_type)

    render_compact_column(c1, "1. Content-Based", "col-content", content_recs, "Genre")
    render_compact_column(c2, "2. Item-Based CF", "col-itemcf", item_cf_recs, "CF")
    render_compact_column(c3, "3. SVD Latent", "col-svd", svd_recs, "SVD")
    render_compact_column(c4, "4. Hybrid", "col-hybrid", hybrid_recs, "Hybrid")

# Optional: Display Evaluation Benchmark Section
if show_eval:
    st.markdown("---")
    st.subheader("📈 Offline Benchmark Evaluation (80/20 Train/Test Split, Seed 42)")
    metrics_path = _project_root / "results" / "metrics.csv"
    chart_path = _project_root / "results" / "comparison.png"

    if metrics_path.exists():
        eval_df = pd.read_csv(metrics_path)
        col_t, col_c = st.columns([1, 1.4])
        with col_t:
            st.markdown("##### Performance Metrics Summary")
            st.dataframe(eval_df.set_index("Method"), use_container_width=True)
            st.markdown(
                r"""
                - **RMSE & MAE**: Measures rating prediction error (lower is better). SVD achieves the lowest error (0.8585).
                - **Precision@10 & Recall@10**: Top-10 ranking metrics on positive test items (rating $\ge$ 4.0). Item-CF achieves the highest recall (0.5663).
                - **Hybrid**: Balances semantic relevance and co-rating patterns with strong recall (0.5363).
                """
            )
        with col_c:
            if chart_path.exists():
                st.image(str(chart_path), caption="Model Comparison: Error vs Ranking Performance")
    else:
        st.info("Run `python3 src/evaluate.py` to generate the evaluation benchmark metrics.")
