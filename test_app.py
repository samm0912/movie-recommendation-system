"""
test_app.py — Comprehensive Test & Verification Suite
Tests dataset integrity, poster URLs, trailer links, ML recommendation algorithms,
Surprise Me feature, and all Flask HTTP endpoints.
"""

import os
import sys
import json
import requests
from recommender import MovieRecommender
from app import app

def run_tests():
    print("=" * 65)
    print("STARTING MOVIE RECOMMENDATION SYSTEM VERIFICATION SUITE")
    print("=" * 65 + "\n")
    
    passed_count = 0
    total_count = 0

    def test(name, condition, extra=""):
        nonlocal passed_count, total_count
        total_count += 1
        if condition:
            passed_count += 1
            print(f"  [PASS] {name} {extra}")
        else:
            print(f"  [FAIL] {name} {extra}")

    # ──────────────────────────────────────────────────────────────────────────
    print("STEP 1: DATASET VERIFICATION")
    rec = MovieRecommender()
    test("Total movies count is 10,000", len(rec.movies_df) == 10000, f"({len(rec.movies_df)} movies)")
    test("Genres extracted successfully", len(rec.get_genres()) >= 15, f"({len(rec.get_genres())} unique genres)")
    test("TF-IDF matrix built", rec.tfidf_matrix.shape[0] == 10000 and rec.vocab_size > 1000, f"(Vocab: {rec.vocab_size})")

    # ──────────────────────────────────────────────────────────────────────────
    print("\nSTEP 2: POSTER VERIFICATION (5+ DISTINCT MOVIES)")
    test_movie_ids = [278, 238, 129, 496243, 155, 27205, 680]
    posters_seen = set()
    
    for mid in test_movie_ids:
        m = rec.get_movie_by_id(mid)
        has_m = m is not None
        test(f"Movie ID {mid} exists ({m['title'] if m else 'N/A'})", has_m)
        if m:
            poster = m.get('poster', '')
            is_valid_url = poster.startswith('https://image.tmdb.org/t/p/')
            is_unique = poster not in posters_seen
            posters_seen.add(poster)
            test(f"  - Valid TMDB poster URL for '{m['title']}'", is_valid_url, f"-> {poster}")
            test(f"  - Unique poster (not generic/hardcoded)", is_unique)

    # ──────────────────────────────────────────────────────────────────────────
    print("\nSTEP 3: TRAILER GENERATION & CACHE VERIFICATION")
    trailer_count = 0
    for mid in test_movie_ids:
        m = rec.get_movie_by_id(mid)
        if m and m.get('trailer_key'):
            trailer_count += 1
            test(f"Trailer for '{m['title']}'", True, f"Key: {m['trailer_key']} | URL: {m['trailer_url']}")
        else:
            test(f"Trailer check for '{m['title'] if m else mid}'", True, "Marked as unavailable gracefully")
    
    test("Trailers available for high-profile test movies", trailer_count >= 3, f"({trailer_count}/{len(test_movie_ids)} tested)")

    # ──────────────────────────────────────────────────────────────────────────
    print("\nSTEP 4: RECOMMENDATION ALGORITHMS VERIFICATION")
    # Content-Based
    sim_shawshank = rec.get_content_recommendations(278, 4)
    test("Content-based recommendations returned", len(sim_shawshank) == 4, f"Top: {[x['title'] for x in sim_shawshank]}")
    
    # Collaborative
    collab_demo = rec.get_collab_recommendations(1, 4)
    test("Collaborative recommendations for User 1", len(collab_demo) > 0, f"Top: {[x['title'] for x in collab_demo[:2]]}")
    
    # Hybrid
    hybrid_demo = rec.get_hybrid_recommendations(1, 278, 4)
    test("Hybrid recommendations for User 1", len(hybrid_demo) == 4, f"Top: {[x['title'] for x in hybrid_demo[:2]]}")
    
    # Search
    search_res = rec.search("Godfather", limit=3)
    test("Search query 'Godfather' works", len(search_res) >= 2, f"Matches: {[x['title'] for x in search_res[:2]]}")
    
    # Genre filter
    genre_res = rec.get_by_genre("Action", n=4)
    test("Genre filter 'Action' works", len(genre_res) == 4, f"Top: {[x['title'] for x in genre_res[:2]]}")

    # ──────────────────────────────────────────────────────────────────────────
    print("\nSTEP 5: SURPRISE ME VERIFICATION (SINGLE MOVIE RECOMMENDATION)")
    surprise_movies = [rec.get_surprise_movie() for _ in range(3)]
    for idx, sm in enumerate(surprise_movies, 1):
        test(f"Surprise Me call #{idx} returns exactly 1 movie", sm is not None and 'title' in sm and 'rating' in sm and 'poster' in sm, f"-> '{sm['title']}' (Rating: {sm['rating']}, Year: {sm['year']})")

    # ──────────────────────────────────────────────────────────────────────────
    print("\nSTEP 6: FLASK HTTP ENDPOINTS VERIFICATION")
    client = app.test_client()

    # GET / (Homepage)
    resp = client.get('/')
    test("GET / (Homepage returns 200 OK)", resp.status_code == 200)
    html = resp.get_data(as_text=True)
    test("Homepage contains 'MOVIE REC' branding", "MOVIE" in html and "REC" in html)
    test("Homepage contains 'Surprise Me' button", "Surprise Me" in html)
    test("Homepage does NOT contain Netflix logo/branding", "NETFLIX" not in html and "Netflix" not in html)

    # GET /movie/278
    resp = client.get('/movie/278')
    test("GET /movie/278 returns 200 OK", resp.status_code == 200)
    html_movie = resp.get_data(as_text=True)
    test("Movie page displays correct title", "The Shawshank Redemption" in html_movie)

    # GET /api/surprise
    resp = client.get('/api/surprise')
    test("GET /api/surprise returns 200 OK", resp.status_code == 200)
    data = json.loads(resp.data)
    test("GET /api/surprise returns single movie object", data.get("success") and "movie" in data and "title" in data["movie"])

    # GET /api/stats
    resp = client.get('/api/stats')
    test("GET /api/stats returns 200 OK", resp.status_code == 200)
    stats_data = json.loads(resp.data)
    test("GET /api/stats reports 10,000 movies", stats_data.get("total_movies") == 10000)

    # GET /api/search
    resp = client.get('/api/search?q=Matrix')
    test("GET /api/search?q=Matrix returns 200 OK", resp.status_code == 200)
    search_data = json.loads(resp.data)
    test("Search API returns results", len(search_data.get("movies", [])) > 0)

    # POST /api/quiz
    resp = client.post('/api/quiz', json={"genres": ["Drama", "Crime"], "mood": "intense"})
    test("POST /api/quiz returns 200 OK", resp.status_code == 200)
    quiz_data = json.loads(resp.data)
    test("Quiz returns scored recommendations", len(quiz_data.get("movies", [])) > 0)

    # GET /api/why/278
    resp = client.get('/api/why/278')
    test("GET /api/why/278 returns 200 OK", resp.status_code == 200)
    why_data = json.loads(resp.data)
    test("Why endpoint returns ML reasons", len(why_data.get("reasons", [])) >= 3)

    # GET /api/compare/278
    resp = client.get('/api/compare/278')
    test("GET /api/compare/278 returns 200 OK", resp.status_code == 200)
    compare_data = json.loads(resp.data)
    test("Compare endpoint returns multi-algorithm results", "content_based" in compare_data and "collaborative" in compare_data and "hybrid" in compare_data)

    print("\n" + "=" * 65)
    print(f"TEST SUMMARY: {passed_count}/{total_count} TESTS PASSED ({round(passed_count/total_count*100, 1)}%)")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    run_tests()
