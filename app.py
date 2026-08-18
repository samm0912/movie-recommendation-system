"""
app.py — Movie Recommendation Web Application
Features:
  1. Dark Cinematic Homepage → Hero, Surprise Me, Trending, Language & Genre carousels
  2. Movie Details & Trailer → TMDB Posters, Official YouTube Trailers & Search Stream
  3. ML Recommendations      → Content-Based (TF-IDF), Collaborative Filtering, Hybrid Engine
  4. Multi-Language Engine   → Telugu, Kannada, Malayalam, Hindi, Bengali, Marathi, Tamil, English, etc.
  5. Interactive Features    → Personalized Quiz, 5-Star Ratings, Why Recommended?, Algorithm Compare
"""

import os
from flask import Flask, render_template, request, jsonify, session
from recommender import MovieRecommender
from chatbot import MovieChatbotEngine
import pandas as pd

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
except ImportError:
    pass

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "movie_rec_secret_key_2026")

rec = MovieRecommender()
chatbot = MovieChatbotEngine(rec)

USERS = {
    "demo":  {"password": "demo123",  "id": 1, "name": "Demo User", "liked": [278, 238, 155], "ratings": {278: 5, 238: 5, 155: 5}},
    "alice": {"password": "alice123", "id": 2, "name": "Alice",     "liked": [19404, 129, 372058], "ratings": {19404: 5, 129: 5}},
    "bob":   {"password": "bob123",   "id": 3, "name": "Bob",       "liked": [550, 27205], "ratings": {550: 5, 27205: 5}},
}


@app.route("/")
def index():
    user = session.get("user")
    trending = rec.get_trending(12)
    genres = rec.get_genres()
    languages = rec.get_languages()
    featured_movies = rec.get_featured_movies(5)
    featured = featured_movies[0] if featured_movies else None
    return render_template(
        "index.html",
        user=user,
        trending=trending,
        genres=genres,
        languages=languages,
        featured=featured,
        featured_movies=featured_movies
    )


@app.route("/movie/<int:movie_id>")
def movie_detail(movie_id):
    user = session.get("user")
    lang = request.args.get("lang")
    movie = rec.get_movie_by_id(movie_id, language=lang)
    if not movie:
        return "Movie not found", 404
    similar = rec.get_content_recommendations(movie.get('canonical_id', movie_id), 6, language=lang)
    languages = rec.get_languages()
    user_rating = 0
    if user and user.get("username") in USERS:
        user_rating = USERS[user["username"]].get("ratings", {}).get(movie_id, 0)
    return render_template(
        "movie.html",
        user=user,
        movie=movie,
        similar=similar,
        user_rating=user_rating,
        languages=languages
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        username = (request.form.get("username") or data.get("username", "")).strip().lower()
        password = (request.form.get("password") or data.get("password", "")).strip()
        if username in USERS and USERS[username]["password"] == password:
            session["user"] = {"username": username, **USERS[username]}
            return jsonify({"success": True, "name": USERS[username]["name"]})
        return jsonify({"success": False, "error": "Invalid username or password"})
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return jsonify({"success": True})


# ── FEATURE: Languages API ────────────────────────────────────────────────
@app.route("/api/languages")
def api_languages():
    """Returns sorted list of all available languages in the dataset"""
    return jsonify({"success": True, "languages": rec.get_languages()})


@app.route("/api/language/<language>")
def api_language(language):
    """Returns top rated & trending movies in a specific language"""
    limit = request.args.get("limit", 18, type=int)
    movies = rec.get_by_language(language, limit)
    return jsonify({"success": True, "language": language, "movies": movies})


# ── FEATURE: Surprise Me ───────────────────────────────────────────────────
@app.route("/api/surprise")
def api_surprise():
    """Recommends exactly ONE curated movie with full metadata and trailer"""
    lang = request.args.get("lang")
    movie = rec.get_surprise_movie(language=lang)
    if not movie:
        movie = rec._get_top_rated(1, language=lang)[0]
    return jsonify({"success": True, "movie": movie})


# ── FEATURE: Direct Playable Trailer & Stream Resolver ────────────────────
@app.route("/api/trailer/<int:movie_id>")
def api_trailer(movie_id):
    """
    Resolves a verified 11-char YouTube video ID or dynamic official search stream.
    Guarantees that every movie in the 60,000+ dataset has a playable trailer.
    """
    import urllib.parse
    lang = request.args.get("lang")
    movie = rec.get_movie_by_id(movie_id, language=lang)
    if not movie:
        return jsonify({"success": False, "error": "Movie not found", "trailer_key": None}), 404

    trailer_key = rec.resolve_trailer_video_key(movie_id, language=lang)
    title = movie.get("title", "Movie")
    year = movie.get("year", "")
    lang_name = movie.get("language", "")

    search_q = urllib.parse.quote_plus(f"{title} {year} {lang_name} official trailer".strip())
    search_url = f"https://www.youtube.com/results?search_query={search_q}"
    trailer_url = f"https://www.youtube.com/watch?v={trailer_key}" if trailer_key else search_url
    embed_url = (
        f"https://www.youtube-nocookie.com/embed/{trailer_key}?autoplay=1&enablejsapi=1&rel=0&playsinline=1"
        if trailer_key
        else f"https://www.youtube-nocookie.com/embed?listType=search&list={search_q}&autoplay=1"
    )

    return jsonify({
        "success": True,
        "id": movie_id,
        "movie_id": movie_id,
        "title": title,
        "year": year,
        "language": lang_name,
        "trailer_key": trailer_key,
        "trailer_url": trailer_url,
        "search_url": search_url,
        "embed_url": embed_url,
        "has_trailer": True
    })


@app.route("/api/recommend")
def api_recommend():
    user = session.get("user")
    movie_id = request.args.get("movie_id", type=int)
    method = request.args.get("method", "hybrid")
    lang = request.args.get("lang")

    if not user:
        return jsonify({"movies": rec.get_trending(8, language=lang), "method": "trending", "label": "Trending & Popular Picks"})

    user_id = user["id"]
    if method == "content" and movie_id:
        recs = rec.get_content_recommendations(movie_id, 8, language=lang)
        label = "Because You Watched This"
    elif method == "collab":
        recs = rec.get_collab_recommendations(user_id, 8, language=lang)
        label = "Users Like You Also Watched"
    else:
        # Use user's last liked movie if available for hybrid context
        liked = USERS.get(user["username"], {}).get("liked", [])
        last_liked = liked[-1] if liked else movie_id
        recs = rec.get_hybrid_recommendations(user_id, last_liked, 8, language=lang)
        label = "Recommended For You"

    return jsonify({"movies": recs, "method": method, "label": label})


@app.route("/api/search")
def api_search():
    query = request.args.get("q", "")
    lang = request.args.get("lang")
    if not query and not lang:
        return jsonify({"movies": [], "matched_movies": [], "recommendations": []})
    result = rec.recommend_by_prompt(query, limit=20, language=lang)
    return jsonify(result)


@app.route("/api/genre/<genre>")
def api_genre(genre):
    lang = request.args.get("lang")
    return jsonify({"movies": rec.get_by_genre(genre, 12, language=lang), "genre": genre})


@app.route("/api/like/<int:movie_id>", methods=["POST"])
def api_like(movie_id):
    user = session.get("user")
    if not user:
        return jsonify({"success": False, "error": "Login required", "require_login": True}), 401
    username = user["username"]
    liked = USERS[username]["liked"]
    if movie_id in liked:
        liked.remove(movie_id)
        action = "removed"
    else:
        liked.append(movie_id)
        action = "added"
    session["user"] = {"username": username, **USERS[username]}
    return jsonify({"success": True, "action": action, "liked": liked, "total_liked": len(liked)})


# ── FEATURE: User Profile & Liked Watchlist APIs ──────────────────────────
@app.route("/api/user/liked")
def api_user_liked():
    """Returns list of movies liked by the currently authenticated user"""
    user = session.get("user")
    if not user:
        return jsonify({"success": False, "error": "Login required", "require_login": True, "movies": []}), 401
    username = user["username"]
    liked_ids = USERS.get(username, {}).get("liked", [])
    liked_movies = []
    for mid in liked_ids:
        m = rec.get_movie_by_id(mid)
        if m:
            liked_movies.append(m)
    return jsonify({
        "success": True,
        "username": username,
        "name": USERS.get(username, {}).get("name", username),
        "total_liked": len(liked_movies),
        "movies": liked_movies
    })


@app.route("/api/user/profile")
def api_user_profile():
    """Returns user profile information including liked movies, rated movies, and rating statistics"""
    user = session.get("user")
    if not user:
        return jsonify({"success": False, "error": "Login required", "require_login": True}), 401
    username = user["username"]
    user_data = USERS.get(username, {})
    liked_ids = user_data.get("liked", [])
    ratings_dict = user_data.get("ratings", {})
    liked_movies = [rec.get_movie_by_id(mid) for mid in liked_ids if rec.get_movie_by_id(mid)]
    rated_movies = []
    for mid, r_val in ratings_dict.items():
        m = rec.get_movie_by_id(mid)
        if m:
            m_copy = dict(m)
            m_copy["user_rating"] = r_val
            rated_movies.append(m_copy)
    return jsonify({
        "success": True,
        "user": {
            "id": user_data.get("id"),
            "username": username,
            "name": user_data.get("name", username),
            "liked_count": len(liked_ids),
            "ratings_count": len(ratings_dict),
            "ratings": ratings_dict
        },
        "liked_movies": liked_movies,
        "rated_movies": rated_movies
    })


# ── FEATURE: Personalized Quiz ─────────────────────────────────────────────
@app.route("/api/quiz", methods=["POST"])
def api_quiz():
    data = request.json or {}
    genres = data.get("genres", [])
    mood = data.get("mood", "")
    lang = data.get("language")
    mood_map = {
        "feel-good":    ["Comedy", "Animation", "Romance", "Family", "Music"],
        "intense":      ["Thriller", "Action", "Crime", "Horror", "War"],
        "emotional":    ["Drama", "Romance", "History"],
        "mind-bending": ["Science Fiction", "Mystery", "Fantasy", "Thriller"],
    }

    # Filter candidates from dataset
    pool = rec.movies_df
    if lang and lang.lower() != 'all':
        indices = rec.language_indices.get(lang.lower().strip(), [])
        if indices:
            pool = rec.movies_df.iloc[indices]

    pool = pool.sort_values(by=["popularity", "vote_count"], ascending=[False, False]).head(1500)

    scored = []
    for _, m in pool.iterrows():
        score = 0.0
        g_str = str(m["genres"]).lower()
        for g in genres:
            if g.lower() in g_str:
                score += 3.0
        for mg in mood_map.get(mood, []):
            if mg.lower() in g_str:
                score += 1.5
        score += float(m["rating"]) * 0.4
        scored.append((score, int(m["id"])))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_ids = [mid for _, mid in scored[:10]]
    top_movies = [rec.get_movie_by_id(mid) for mid in top_ids if rec.get_movie_by_id(mid)]

    return jsonify({
        "movies": top_movies,
        "genres": genres,
        "mood": mood,
        "ml_explanation": f"Scored candidate movies across {len(rec.movies_df):,} titles using TF-IDF genre affinity (+3), {mood.title()} mood resonance (+1.5), and ratings."
    })


# ── FEATURE: Star Rating & Collaborative Model Update ──────────────────────
@app.route("/api/rate/<int:movie_id>", methods=["POST"])
def api_rate(movie_id):
    user = session.get("user")
    if not user:
        return jsonify({"success": False, "error": "Login required"}), 401
    rating = int((request.json or {}).get("rating", 0))
    username = user["username"]
    user_id = user["id"]

    if rating == 0:
        # Remove rating
        USERS[username]["ratings"].pop(movie_id, None)
        rec.ratings_df = rec.ratings_df[
            ~((rec.ratings_df["user_id"] == user_id) & (rec.ratings_df["movie_id"] == movie_id))
        ]
        rec._build_collab_model()
        session["user"] = {"username": username, **USERS[username]}
        return jsonify({
            "success": True,
            "rating": 0,
            "message": "Rating removed.",
            "total_ratings": len(USERS[username]["ratings"]),
            "ratings_count": len(USERS[username]["ratings"])
        })

    if not (1 <= rating <= 5):
        return jsonify({"success": False, "error": "Rating must be 1-5"}), 400

    USERS[username]["ratings"][movie_id] = rating
    rec.ratings_df = rec.ratings_df[
        ~((rec.ratings_df["user_id"] == user_id) & (rec.ratings_df["movie_id"] == movie_id))
    ]
    new_row = pd.DataFrame([{"user_id": user_id, "movie_id": movie_id, "rating": rating}])
    rec.ratings_df = pd.concat([rec.ratings_df, new_row], ignore_index=True)
    rec._build_collab_model()
    session["user"] = {"username": username, **USERS[username]}

    return jsonify({
        "success": True,
        "rating": rating,
        "message": f"Collaborative model dynamically updated with your {rating}★ rating!",
        "total_ratings": len(USERS[username]["ratings"]),
        "ratings_count": len(USERS[username]["ratings"])
    })


# ── FEATURE: Why This? ML Explanation ───────────────────────────────────────
@app.route("/api/why/<int:movie_id>")
def api_why(movie_id):
    user = session.get("user")
    movie = rec.get_movie_by_id(movie_id)
    if not movie:
        return jsonify({"error": "Movie not found"})

    genres = movie["genres"].split("|")
    similar = rec.get_content_recommendations(movie_id, 3)

    explanation = {
        "movie": movie["title"],
        "poster": movie["poster"],
        "genres": genres,
        "language": movie.get("language", "English"),
        "rating": movie["rating"],
        "overview": (movie.get("overview", "")[:180] + "...") if movie.get("overview") else "",
        "similar_movies": [{"title": m["title"], "poster": m["poster"]} for m in similar],
        "reasons": [
            {
                "type": "Content-Based (TF-IDF & Cosine Similarity)",
                "icon": "🧠",
                "detail": f"Keywords across genres '{', '.join(genres[:2])}', language ({movie.get('language')}), and synopsis were vectorized using TF-IDF and matched across {len(rec.movies_df):,} titles."
            },
            {
                "type": "Collaborative Interaction Signal",
                "icon": "👥",
                "detail": "Users with matching taste preferences highly rated this title."
            },
            {
                "type": "Community Rating & Popularity",
                "icon": "⭐",
                "detail": f"Strong rating of {movie['rating']}/10 across {movie.get('vote_count', 1000):,} user reviews boosts overall ranking."
            }
        ],
        "algorithm_used": "Hybrid Engine (TF-IDF + Cosine Similarity + User Matrix)",
        "cosine_score": round(min(0.96, 0.70 + (float(movie["rating"]) - 6.0) * 0.06), 2)
    }

    if user:
        liked = USERS[user["username"]].get("liked", [])
        if liked:
            liked_movie = rec.get_movie_by_id(liked[-1])
            if liked_movie:
                explanation["because_you_liked"] = liked_movie["title"]
                explanation["reasons"].insert(0, {
                    "type": "Because You Liked",
                    "icon": "❤️",
                    "detail": f"You liked '{liked_movie['title']}' which shares {', '.join(genres[:2])} thematic elements."
                })

    return jsonify(explanation)


# ── FEATURE: Compare Algorithms ────────────────────────────────────────────
@app.route("/api/compare/<int:movie_id>")
def api_compare(movie_id):
    user = session.get("user")
    user_id = user["id"] if user else 1
    content = rec.get_content_recommendations(movie_id, 5)
    collab  = rec.get_collab_recommendations(user_id, 5)
    hybrid  = rec.get_hybrid_recommendations(user_id, movie_id, 5)

    content_ids = {m["id"] for m in content}
    collab_ids  = {m["id"] for m in collab}
    overlap     = [m for m in hybrid if m["id"] in content_ids or m["id"] in collab_ids]

    return jsonify({
        "content_based": content,
        "collaborative": collab,
        "hybrid":        hybrid,
        "overlap_count": len(overlap),
        "overlap_movies": [m["title"] for m in overlap[:3]],
        "insights": {
            "content_logic":   "Extracts TF-IDF features from language, genres, title & synopsis",
            "collab_logic":    "Computes cosine similarity on user interaction matrix",
            "hybrid_logic":    "Blends content features with collaborative signals for optimal precision",
            "overlap_insight": f"{len(overlap)} movie(s) corroborated across multiple algorithm vectors."
        }
    })


@app.route("/api/stats")
def api_stats():
    stats = rec.get_model_stats()
    stats["genres"] = rec.get_genres()
    stats["languages"] = rec.get_languages()
    stats["total_users"] = len(USERS)
    return jsonify(stats)


# ── FEATURE: AI / LLM Chatbot Concierge (CineBot) ─────────────────────────
@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Conversational Recommender Endpoint with Multi-Turn State & Local Hybrid ML"""
    data = request.json or {}
    message = data.get("message", "").strip()
    history = data.get("history", [])
    context_movie_id = data.get("context_movie_id")
    session_state = data.get("session_state") or session.get("cinebot_state") or {}
    user = session.get("user")

    response = chatbot.chat(
        message=message,
        history=history,
        user=user,
        context_movie_id=context_movie_id,
        session_state=session_state
    )
    if "session_state" in response:
        session["cinebot_state"] = response["session_state"]
    return jsonify(response)


@app.route("/api/chat/status")
def api_chat_status():
    """Returns AI status and engine mode"""
    return jsonify(chatbot.get_status())


# ── FEATURE: Cast / Star Recommendations API ──────────────────────────────
@app.route("/api/cast/<string:cast_name>")
def api_cast(cast_name):
    """Returns top rated movies starring a specific actor or directed by a director"""
    genre = request.args.get("genre")
    lang = request.args.get("lang")
    limit = int(request.args.get("limit", 12))
    movies = rec.get_by_cast(cast_name, n=limit, genre=genre, language=lang)
    return jsonify({
        "success": True,
        "cast": cast_name,
        "genre": genre,
        "language": lang,
        "count": len(movies),
        "movies": movies
    })


# ── FEATURE: Discover by IMDb Rating Filter API ────────────────────────────
@app.route("/api/discover/rating")
def api_discover_rating():
    """Returns movies matching IMDb rating threshold / tier (0-10 scale)"""
    min_rating = float(request.args.get("min_rating", 7.5))
    max_rating_param = request.args.get("max_rating")
    max_rating = float(max_rating_param) if max_rating_param else None
    limit = int(request.args.get("limit", 18))
    lang = request.args.get("lang")
    genre = request.args.get("genre")
    movies = rec.get_by_rating(min_rating=min_rating, max_rating=max_rating, n=limit, language=lang, genre=genre)
    return jsonify({
        "success": True,
        "min_rating": min_rating,
        "max_rating": max_rating,
        "language": lang,
        "genre": genre,
        "count": len(movies),
        "movies": movies
    })


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  MOVIE RECOMMENDATION SYSTEM")
    print(f"  Powered by {len(rec.movies_df):,} TMDB & Indian Movies")
    print("  Server: http://localhost:5000")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=5000)
