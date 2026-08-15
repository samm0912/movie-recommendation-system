# 📽️ Movie Recommendation System

A Machine Learning web application built with Python, Flask, and scikit-learn that provides personalized movie recommendations across a dataset of **10,000 TMDB movies**, featuring live TMDB posters, official YouTube trailer playback, interactive quizzes, 5-star rating collaborative learning, and instant "Surprise Me" recommendations.

---

## 🌟 Key Features

1. **10,000 Movies Dataset**: Powered by the top 10K TMDB dataset with accurate titles, genres, ratings, overviews, and release years.
2. **Official YouTube Trailers**: Automatic metadata fetching and responsive in-app modal video player.
3. **High-Definition Movie Posters**: Direct TMDB CDN poster and backdrop image integration with reliable fallbacks.
4. **Surprise Me**: Instant single-movie recommendation button with rich details and direct trailer launch.
5. **Multi-Vector ML Algorithms**:
   - **Content-Based Filtering**: TF-IDF vectorization across genres, titles, and plot summaries with fast sparse cosine similarity.
   - **Collaborative Filtering**: Dynamic user-movie interaction matrix updated live when users rate movies.
   - **Hybrid Model**: Blends collaborative taste signals with content-based features.
6. **Interactive ML Tools**:
   - **Personalized Quiz**: 2-step genre and mood affinity scorer.
   - **Why This?**: Explainable AI modal detailing TF-IDF match scores and rating weights.
   - **Compare Algorithms**: Side-by-side comparison of Content, Collaborative, and Hybrid recommendation sets.
   - **ML Status**: Real-time stats on vocabulary size, active titles, and precision metrics.
7. **Dark Cinematic Theme**: Modern charcoal, deep slate, and electric cyan/indigo aesthetic.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Fetch/Update Trailer & Metadata Cache (Optional/Pre-cached)
```bash
python fetch_trailers.py
```

### 3. Start the Flask Application
```bash
python app.py
```

### 4. Open in Browser
Visit: [http://localhost:5000](http://localhost:5000)

---

## 🔑 Environment Configuration

Create a `.env` file in the root directory (or use the provided default):
```env
TMDB_API_KEY=your_tmdb_api_key_here
FLASK_SECRET_KEY=your_secret_key_here
```

---

## 👤 Demo User Accounts

| Username | Password | Profile Description |
|---|---|---|
| `demo` | `demo123` | Action, Sci-Fi, and Crime enthusiast |
| `alice` | `alice123` | Drama, Animation, and Romance fan |
| `bob` | `bob123` | Thriller, Mystery, and Sci-Fi buff |

---

## 📁 Project Architecture

```text
netflix-recommender/
├── app.py                  ← Flask web server & API endpoints
├── recommender.py          ← Core ML Engine (TF-IDF, Collaborative, Hybrid)
├── fetch_trailers.py       ← TMDB trailer/poster fetcher and cache engine
├── test_app.py             ← Comprehensive automated test suite
├── requirements.txt        ← Python dependencies
├── .env                    ← Environment variables (gitignored)
├── data/
│   ├── top10K-TMDB-movies.csv   ← Primary 10,000 movie dataset
│   ├── movie_meta_cache.json    ← Precomputed TMDB posters & trailer cache
│   └── movies.py                ← Legacy 25-movie mock data (archived)
├── templates/
│   ├── index.html          ← Main discovery homepage
│   └── movie.html          ← Movie detail and similar titles page
└── static/
    ├── css/style.css       ← Dark cinematic theme styling
    └── js/app.js           ← Interactive frontend controller
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `GET /` | `GET` | Main discovery homepage |
| `GET /movie/<id>` | `GET` | Movie details & similar recommendations |
| `GET /api/surprise` | `GET` | Returns a single smart recommendation |
| `GET /api/trailer/<id>` | `GET` | Retrieves trailer details for a movie |
| `GET /api/recommend` | `GET` | Returns hybrid / content / collab recommendations |
| `GET /api/search?q=` | `GET` | Live search across 10,000 titles |
| `GET /api/genre/<genre>`| `GET` | Filter movies by genre |
| `GET /api/stats` | `GET` | ML model statistics |
| `GET /api/why/<id>` | `GET` | Explains ML reasoning for recommendation |
| `GET /api/compare/<id>` | `GET` | Multi-algorithm side-by-side comparison |
| `POST /api/quiz` | `POST` | Scores movies based on user preferences |
| `POST /api/rate/<id>` | `POST` | Submits 1-5 star rating and updates model |
| `POST /api/like/<id>` | `POST` | Toggles movie in user watchlist |
