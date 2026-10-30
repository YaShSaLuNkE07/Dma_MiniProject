# 🎬 Movie Recommendation System Using Collaborative and Content-Based Filtering

A comprehensive data mining mini-project implementing and comparing four distinct movie recommendation strategies on the **MovieLens Latest Small** dataset:
1. **Content-Based Filtering** (TF-IDF Vectorization & Cosine Similarity)
2. **Item-Based Collaborative Filtering** (Item-Item Cosine Similarity over User Vectors)
3. **SVD Matrix Factorization** (Latent Factor Decomposition via Truncated Singular Value Decomposition)
4. **Hybrid Filtering** (Configurable $\alpha$-Weighted Fusion with Min-Max Score Normalization)

---

## 📌 Project Overview
Recommender systems are central to modern content discovery platforms. This project explores the mathematical formulations, trade-offs, and empirical performance of content-driven semantic matching versus user-interaction collaborative filtering. 

The system features:
- **Modular Data Pipeline**: Automated dataset ingestion, title/release-year regex extraction, genre tokenization, and long-tail noise filtering ($\ge 5$ ratings).
- **Interactive Streamlit Web Dashboard**: Searchable movie catalog, real-time side-by-side recommendation comparison across all 4 algorithms, dynamic $\alpha$-weighting, and an embedded evaluation benchmark viewer.
- **Reproducible Evaluation**: Offline 80/20 train/test evaluation (fixed random seed `42`) measuring **RMSE**, **MAE**, **Precision@10**, and **Recall@10**.

---

## 🏗️ Repository Architecture

```text
movie-recommender/
├── data/
│   ├── download_data.py       # Automated MovieLens downloader & extractor
│   ├── movies.csv             # Raw MovieLens catalog (9,742 titles)
│   └── ratings.csv            # Raw MovieLens user ratings (100,836 reviews)
├── notebooks/
│   └── 01_eda.ipynb           # Exploratory Data Analysis with 4 visual insights
├── src/
│   ├── __init__.py
│   ├── preprocess.py          # Data cleaning, filtering, and sparse matrix generation
│   ├── content_based.py       # TF-IDF & Cosine content recommender
│   ├── collaborative.py       # Item-Based CF & SVD matrix factorization
│   ├── hybrid.py              # Normalized alpha-weighted hybrid fusion
│   └── evaluate.py            # 80/20 train/test split offline benchmark
├── results/
│   ├── metrics.csv            # Offline evaluation metrics table
│   └── comparison.png         # Grouped bar chart comparing RMSE, MAE, Precision, Recall
├── app.py                     # Interactive Streamlit application
├── requirements.txt           # Python dependency specifications
└── README.md                  # Complete documentation
```

---

## ⚡ Quick Start & Run Commands

### 1. Environment Setup
```bash
# Clone the repository and navigate into the project directory
cd movie-recommender

# Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data
```bash
python3 data/download_data.py
```

### 3. Run Preprocessing Check
```bash
python3 src/preprocess.py
```

### 4. Run Offline Model Evaluation & Generate Benchmark Charts
```bash
python3 src/evaluate.py
```

### 5. Launch Interactive Streamlit App
```bash
streamlit run app.py
```

---

## 🔬 Methodology & Mathematical Formulations

### 1. Content-Based Filtering
- **Feature Representation**: Movie genres are extracted and tokenized with composite hyphen support (`Sci-Fi`, `Film-Noir`). TF-IDF vectors are constructed:
  $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
- **Similarity Metric**: Pairwise Cosine Similarity:
  $$\text{sim}_{\text{content}}(i, j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}$$
- **Strength**: Unaffected by user cold-start; can recommend newly added movies with zero ratings.

### 2. Item-Based Collaborative Filtering (Item-CF)
- **Feature Representation**: Each movie $i$ is modeled as a vector of user ratings $\mathbf{r}_i \in \mathbb{R}^M$ across all users in the system.
- **Item-Item Similarity**:
  $$\text{sim}_{\text{CF}}(i, j) = \frac{\sum_{u \in U} r_{u,i} \cdot r_{u,j}}{\sqrt{\sum_{u \in U} r_{u,i}^2} \sqrt{\sum_{u \in U} r_{u,j}^2}}$$
- **Strength**: Captures serendipitous community co-consumption patterns independent of metadata taxonomy.

### 3. SVD Matrix Factorization
- **Formulation**: Approximates the sparse user-item interaction matrix $R \approx U \Sigma V^T$ or minimizes regularized reconstruction error:
  $$\min_{P, Q, b} \sum_{(u, i) \in R_{\text{train}}} (r_{u, i} - (\mu + b_u + b_i + \mathbf{p}_u^T \mathbf{q}_i))^2 + \lambda (\|\mathbf{p}_u\|^2 + \|\mathbf{q}_i\|^2 + b_u^2 + b_i^2)$$
  where $\mathbf{p}_u \in \mathbb{R}^k$ and $\mathbf{q}_i \in \mathbb{R}^k$ are $k=50$ latent factor vectors.
- **Strength**: Compact latent space captures hidden conceptual genres and yields superior rating prediction accuracy.

### 4. Hybrid Recommender
- **Score Normalization**: To prevent collaborative scores from dominating content scores due to variance differences, both vectors are min-max scaled:
  $$\tilde{s}(i) = \frac{s(i) - \min(s)}{\max(s) - \min(s) + \epsilon}$$
- **Linear Fusion**:
  $$\text{Score}_{\text{hybrid}}(i) = \alpha \cdot \tilde{s}_{\text{content}}(i) + (1 - \alpha) \cdot \tilde{s}_{\text{CF}}(i)$$
- **Configurable $\alpha$**:
  - $\alpha = 1.0$: 100% Content-based
  - $\alpha = 0.0$: 100% Collaborative filtering
  - $\alpha = 0.5$: Balanced hybrid blend

---

## 📊 Empirical Evaluation & Results

Evaluated on an **80/20 train/test split** ($N_{\text{train}} = 72,219$ ratings, $N_{\text{test}} = 18,055$ ratings) with random seed `42`. Relevant ground truth for ranking is defined as rating $\ge 4.0$.

| Method | RMSE (↓) | MAE (↓) | Precision@10 (↑) | Recall@10 (↑) |
| :--- | :---: | :---: | :---: | :---: |
| **Content-Based** | 0.9097 | 0.7039 | 0.3995 | 0.3848 |
| **Item-Based CF** | 0.9093 | 0.7002 | **0.4980** | **0.5663** |
| **SVD** | **0.8585** | **0.6544** | 0.4874 | 0.4826 |
| **Hybrid ($\alpha=0.5$)** | 0.9051 | 0.6990 | 0.4815 | 0.5363 |

### Key Benchmark Insights
1. **SVD excels at rating prediction**: SVD achieves the lowest RMSE (**0.8585**) and MAE (**0.6544**), demonstrating the power of regularized latent factor representations for numerical accuracy.
2. **Item-CF leads in top-N recall**: Item-based CF achieves the highest Top-10 Recall (**0.5663**) and Precision (**0.4980**), successfully recovering items that users rated 4.0 or above.
3. **Hybrid balances both**: The Hybrid model outperforms pure Content-Based in both rating accuracy (RMSE 0.9051 vs 0.9097) and ranking quality (Precision 0.4815 vs 0.3995), combining semantic relevance with collaborative co-rating signals.

---

## 🎬 Example Outputs (Top-5 Recommendations)

### 1. Target: *Toy Story (1995)*
*Genres: Adventure | Animation | Children | Comedy | Fantasy*

| Rank | Content-Based | Item-Based CF | SVD Latent | Hybrid ($\alpha=0.5$) |
| :---: | :--- | :--- | :--- | :--- |
| **#1** | Emperor's New Groove (1.0000) | Toy Story 2 (0.5726) | E.T. the Extra-Terrestrial (0.5776) | Toy Story 2 (0.7863) |
| **#2** | Monsters, Inc. (1.0000) | Jurassic Park (0.5656) | Louis C.K.: Chewed Up (0.4842) | Monsters, Inc. (0.7523) |
| **#3** | Toy Story 2 (1.0000) | Independence Day (0.5643) | Benjamin Button (0.4320) | Shrek (0.7293) |
| **#4** | Antz (1.0000) | Star Wars: Ep. IV (0.5574) | Suspiria (0.4278) | Finding Nemo (0.6819) |
| **#5** | Rocky and Bullwinkle (1.0000) | Forrest Gump (0.5471) | Lion King, The (0.4263) | Antz (0.6803) |

---

### 2. Target: *The Dark Knight (2008)*
*Genres: Action | Crime | Drama | IMAX*

| Rank | Content-Based | Item-Based CF | SVD Latent | Hybrid ($\alpha=0.5$) |
| :---: | :--- | :--- | :--- | :--- |
| **#1** | Batman Begins (0.9538) | Inception (0.7273) | Dick (0.4538) | Batman Begins (0.8025) |
| **#2** | Fast Five (0.9296) | Iron Man (0.6705) | Koyaanisqatsi (0.4463) | The Dark Knight Rises (0.7658) |
| **#3** | A Good Day to Die Hard (0.8807) | The Dark Knight Rises (0.6661) | American Psycho (0.4287) | Inception (0.7402) |
| **#4** | Fast & Furious 6 (0.8807) | Batman Begins (0.6513) | Cowboys & Aliens (0.4182) | Avatar (0.5934) |
| **#5** | Eagle Eye (0.8807) | LOTR: Return of the King (0.6206) | X-Files: Fight the Future (0.4064) | Star Trek (0.5930) |

---

### 3. Target: *Titanic (1997)*
*Genres: Drama | Romance*

| Rank | Content-Based | Item-Based CF | SVD Latent | Hybrid ($\alpha=0.5$) |
| :---: | :--- | :--- | :--- | :--- |
| **#1** | Crazy Heart (1.0000) | Men in Black (0.5482) | Green Zone (0.5588) | Good Will Hunting (0.7437) |
| **#2** | Mona Lisa Smile (1.0000) | Star Wars: Ep. I (0.5248) | 10 Things I Hate About You (0.4599) | A Beautiful Mind (0.7226) |
| **#3** | Remains of the Day (1.0000) | Saving Private Ryan (0.5097) | Crocodile Dundee (0.4425) | American Beauty (0.7084) |
| **#4** | Bed of Roses (1.0000) | Shrek (0.5058) | Angels in the Outfield (0.4409) | Jerry Maguire (0.7051) |
| **#5** | Tristan & Isolde (1.0000) | Catch Me If You Can (0.4951) | Grease (0.4315) | The Notebook (0.6754) |

---

## ⚠️ System Limitations & Practical Challenges

1. **Cold-Start Problem**:
   - *New Movies*: Collaborative filtering and SVD cannot score movies with zero user ratings. Content-based filtering mitigates this by relying solely on metadata.
   - *New Users*: Without initial interaction history, personalized CF falls back to global averages.
2. **Extreme Matrix Sparsity**:
   - The MovieLens user-item interaction matrix exhibits **95.95% sparsity**. Over 95 out of every 100 possible user-movie interactions are missing, limiting the overlap of shared ratings between users.
3. **Popularity & Blockbuster Bias**:
   - In item-based CF, widely watched blockbusters (e.g., *Forrest Gump*, *Jurassic Park*, *Star Wars*) frequently appear as neighbors across diverse genres because they share co-rating volume with almost all titles.
4. **Coarse-Grained Taxonomy**:
   - MovieLens small contains only 20 broad genre tags. High similarity scores ($1.00$) occur frequently among movies sharing broad tags (e.g., *Drama | Romance*), highlighting the need for deeper synopsis or tag genome embeddings.

---

## 🔮 Future Work
- **Tag Genome Integration**: Integrate the 1,128 MovieLens tag genome features to construct dense semantic embeddings beyond basic genres.
- **Deep Neural Collaborative Filtering (NCF)**: Replace linear dot products with multi-layer perceptrons (MLP) to capture non-linear user-item feature interactions.
- **Implicit Feedback Modeling**: Incorporate dwell time, search clicks, and skip rates via Bayesian Personalized Ranking (BPR).
- **Temporal Dynamics**: Add time-decay weighting to account for evolving user preferences over time.
