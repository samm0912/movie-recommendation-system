"""
chatbot.py — AI & Conversational Recommender Engine (CineBot)
Supports:
  1. Google Gemini API integration (via REST for zero-dependency reliability)
  2. Smart Local NLP Fallback Engine (Intent matching, Entity extraction, TF-IDF + Hybrid queries)
  3. Structured response format with rich movie cards, trailers, ratings, and follow-up prompts
"""

import os
import re
import json
import random
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class MovieChatbotEngine:
    def __init__(self, recommender):
        self.rec = recommender
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.gemini_model = "gemini-1.5-flash"

    def get_status(self):
        """Returns the status of the chatbot backend"""
        return {
            "available": True,
            "has_gemini_key": bool(self.api_key),
            "mode": "gemini_ai" if self.api_key else "local_nlp_engine",
            "model_name": "Gemini 1.5 Flash + Local Hybrid ML" if self.api_key else "Local Hybrid NLP Engine"
        }

    def chat(self, message, history=None, user=None, context_movie_id=None):
        """
        Main chat handler. Attempts Gemini API first if configured,
        otherwise seamlessly uses the high-precision Local NLP Recommender Engine.
        """
        msg = str(message or "").strip()
        if not msg:
            return {
                "success": True,
                "reply": "👋 Hi there! I'm **CineBot**, your AI movie concierge. Ask me for recommendations, search by mood or title, or say *'Surprise Me'*!",
                "movies": [],
                "suggested_prompts": ["🍿 Surprise Me", "🌌 Sci-Fi like Interstellar", "😂 Feel-Good Comedy", "🧠 How does ML work?"],
                "mode": "greeting"
            }

        # If Gemini API key is available, try LLM response
        if self.api_key:
            try:
                llm_res = self._call_gemini(msg, history, user)
                if llm_res and llm_res.get("success"):
                    return llm_res
            except Exception as e:
                print(f"[CineBot] Gemini API call fallback triggered: {e}")

        # High-Precision Local NLP Fallback Engine
        return self._local_fallback_chat(msg, history, user, context_movie_id)

    # ── 1. Google Gemini LLM Integration ──────────────────────────────────────
    def _call_gemini(self, message, history=None, user=None):
        """Calls Gemini API via HTTPS REST with dataset grounding context"""
        # Pre-fetch candidate movies from prompt for grounding
        local_search = self.rec.recommend_by_prompt(message, limit=6)
        candidates = local_search.get("movies", [])
        candidate_summary = []
        for m in candidates[:6]:
            candidate_summary.append({
                "id": m.get("id"),
                "title": m.get("title"),
                "rating": m.get("rating"),
                "year": m.get("year"),
                "genres": m.get("genres"),
                "overview": (m.get("overview") or "")[:150]
            })

        system_instruction = (
            "You are CineBot, an intelligent, enthusiastic movie assistant integrated into a Movie Recommender System. "
            "You have access to a database of 10,000 TMDB movies with ratings, genres, and trailers. "
            "Always be helpful, cinematic, concise, and recommend relevant movies. "
            "Ground your recommendations in the provided movie database candidates when relevant. "
            "Format movie titles in **bold**."
        )

        user_context = f"User profile: {user.get('name') if user else 'Guest'}. "
        dataset_context = f"Top matching movies in local database: {json.dumps(candidate_summary)}. "
        prompt_text = f"{system_instruction}\n\n{user_context}\n{dataset_context}\nUser Question: {message}\n\nPlease respond helpfully to the user."

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt_text}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 600,
            }
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            reply_text = data["candidates"][0]["content"]["parts"][0]["text"]
            
            # Enrich movie cards from dataset
            matched_movies = candidates[:5] if candidates else []
            return {
                "success": True,
                "reply": reply_text,
                "movies": matched_movies,
                "suggested_prompts": self._generate_followup_prompts(message, matched_movies),
                "mode": "gemini_ai"
            }
        return None

    # ── 2. Smart Local NLP Fallback Engine ─────────────────────────────────────
    def _local_fallback_chat(self, message, history=None, user=None, context_movie_id=None):
        """
        High-precision local conversational NLP engine.
        Parses intents, extracts movie titles & moods, and generates rich recommendations.
        """
        q = message.strip()
        q_lower = q.lower()
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', q_lower)

        # ── INTENT A: Greetings ──────────────────────────────────────────────
        if re.match(r'^(hi|hello|hey|greetings|hola|sup|good\s+(morning|evening|afternoon))\b', q_lower):
            name = user.get("name") if user else "movie fan"
            return {
                "success": True,
                "reply": f"👋 Hello **{name}**! I'm **CineBot**, your AI movie assistant.\n\n"
                         f"You can ask me things like:\n"
                         f"• *'Recommend movies like Interstellar'*\n"
                         f"• *'I want a mind-bending sci-fi thriller'*\n"
                         f"• *'Surprise me with a top-rated movie'*\n"
                         f"• *'Show me trailers for The Dark Knight'*",
                "movies": self.rec.get_trending(4),
                "suggested_prompts": ["🍿 Surprise Me", "🌌 Movies like Interstellar", "🎭 Best Thrillers", "😂 Comedy Classics"],
                "mode": "local_fallback"
            }

        # ── INTENT B: Surprise Me / Random Discovery ─────────────────────────
        if re.search(r'\b(surprise|random|pick one|choose for me|give me a movie|something good)\b', q_lower):
            movie = self.rec.get_surprise_movie()
            if not movie:
                movie = self.rec._get_top_rated(1)[0]
            trailer_note = "🎬 Has official trailer available!" if movie.get("has_trailer") else ""
            return {
                "success": True,
                "reply": f"🎉 **Surprise Pick for You:** **{movie['title']}** ({movie['year']})\n\n"
                         f"⭐ **Rating:** {movie['rating']}/10 · 🎭 **Genres:** {movie['genres'].replace('|', ', ')}\n\n"
                         f"📝 *{movie.get('overview', '')[:200]}...*\n\n{trailer_note}",
                "movies": [movie],
                "suggested_prompts": ["🍿 Another Surprise", f"🔍 Similar to {movie['title']}", "▶ Watch Trailer"],
                "mode": "local_fallback"
            }

        # ── INTENT C: Explanation / How ML Works ─────────────────────────────
        if re.search(r'\b(how.*(work|algorithm|ml|model|recommend)|explain.*model|what algorithm)\b', q_lower):
            stats = self.rec.get_model_stats()
            return {
                "success": True,
                "reply": f"🧠 **How the Recommendation Engine Works:**\n\n"
                         f"1. **TF-IDF Content Filtering**: Vectorizes movie titles, genres, and overviews across **{stats['total_movies']:,} movies** using **{stats['vocab_size']:,} text features**.\n"
                         f"2. **Cosine Similarity**: Measures mathematical distance between movie vectors to find the closest thematic matches.\n"
                         f"3. **Collaborative Filtering**: Builds a dynamic User-Movie interaction matrix from ratings to find viewers with similar tastes.\n"
                         f"4. **Hybrid Blending**: Harmonizes content affinity and collaborative signals for optimal accuracy ({stats['accuracy_score']}).",
                "movies": self.rec.get_trending(3),
                "suggested_prompts": ["🍿 Test with Interstellar", "🎭 Compare Algorithms", "⭐ Take Taste Quiz"],
                "mode": "local_fallback"
            }

        # ── INTENT D: Trailer Inquiry ────────────────────────────────────────
        if re.search(r'\b(trailer|watch trailer|play trailer|video)\b', q_lower):
            # Extract title candidate
            for _, row in self.rec.movies_df.iterrows():
                t = str(row['title']).lower()
                if len(t) > 3 and t in q_lower:
                    m = self.rec.get_movie_by_id(int(row['id']))
                    if m:
                        return {
                            "success": True,
                            "reply": f"🎬 Here is the official trailer for **{m['title']}** ({m['year']}). Click **'▶ Trailer'** on the card below to launch the video player!",
                            "movies": [m],
                            "suggested_prompts": [f"🔍 Similar to {m['title']}", "🍿 Surprise Me"],
                            "mode": "local_fallback"
                        }

        # ── INTENT E: Specific Title Match & Similar Recommendations ────────
        # (Handles queries like "Interstellar", "I liked Interstellar", "movies like Fight Club")
        prompt_res = self.rec.recommend_by_prompt(q, limit=8)
        matched = prompt_res.get("matched_movies", [])
        recs = prompt_res.get("recommendations", [])

        if matched:
            target_movie = matched[0]
            other_matched = matched[1:] if len(matched) > 1 else []
            # Combine target movie first, then similar recommendations
            all_cards = [target_movie] + other_matched + recs[:5]
            
            # Format high-accuracy AI response
            reply_lines = [
                f"🚀 Found **{target_movie['title']}** ({target_movie['year']}) ⭐ {target_movie['rating']}/10.",
                f"\nBased on its theme (*{target_movie['genres'].replace('|', ', ')}*) and synopsis, here are the **top similar recommendations** calculated via TF-IDF cosine similarity & hybrid ranking:"
            ]
            
            return {
                "success": True,
                "reply": "\n".join(reply_lines),
                "movies": all_cards,
                "suggested_prompts": [
                    f"🎬 Trailer for {target_movie['title']}",
                    f"✨ Why {target_movie['title']}?",
                    "🌌 More Sci-Fi",
                    "🍿 Surprise Me"
                ],
                "mode": "local_fallback"
            }

        # ── INTENT F: Mood / Vibe Recognition ────────────────────────────────
        mood_map = {
            "feel-good": ("feel-good & uplifting", ["Comedy", "Animation", "Family", "Romance"]),
            "happy": ("fun & lighthearted", ["Comedy", "Family", "Music"]),
            "mind-bending": ("mind-bending & psychological", ["Science Fiction", "Mystery", "Thriller"]),
            "scary": ("spine-chilling horror", ["Horror", "Thriller", "Mystery"]),
            "horror": ("spine-chilling horror", ["Horror", "Thriller"]),
            "intense": ("high-octane action & suspense", ["Action", "Thriller", "Crime"]),
            "action": ("thrilling action", ["Action", "Adventure"]),
            "emotional": ("deeply emotional & moving", ["Drama", "Romance", "History"]),
            "sad": ("tearjerker drama", ["Drama", "Romance"]),
            "romantic": ("romantic & heartfelt", ["Romance", "Drama", "Comedy"]),
            "cozy": ("cozy & heartwarming", ["Animation", "Family", "Comedy"]),
        }

        matched_mood_key = None
        for key in mood_map:
            if key in q_lower or (key == "cozy" and "rainy" in q_lower):
                matched_mood_key = key
                break

        if matched_mood_key:
            label, genres = mood_map[matched_mood_key]
            # Pool top rated movies in those genres
            candidates = []
            for g in genres:
                candidates.extend(self.rec.get_by_genre(g, 4))
            
            # Deduplicate
            seen = set()
            unique_candidates = []
            for m in candidates:
                if m["id"] not in seen:
                    seen.add(m["id"])
                    unique_candidates.append(m)
            
            unique_candidates.sort(key=lambda x: (float(x.get("rating", 0)), float(x.get("popularity", 0))), reverse=True)
            top_mood_movies = unique_candidates[:6]
            
            return {
                "success": True,
                "reply": f"✨ Here are the best **{label}** movies from our 10,000 TMDB collection tailored for your vibe:",
                "movies": top_mood_movies,
                "suggested_prompts": ["🍿 Surprise Me", "🎭 More in this genre", "🧠 Explain recommendations"],
                "mode": "local_fallback"
            }

        # ── INTENT G: General Natural Language / Keyword Search ──────────────
        search_movies = prompt_res.get("movies", [])
        if search_movies:
            return {
                "success": True,
                "reply": f"🎬 Here are the top matching titles from our 10,000 movies database for *'{q}'*:",
                "movies": search_movies[:6],
                "suggested_prompts": ["🍿 Surprise Me", "🔥 What's Trending?", "⭐ Best Rated"],
                "mode": "local_fallback"
            }

        # Fallback if nothing matched
        return {
            "success": True,
            "reply": f"🤔 I couldn't find an exact match for *'{q}'* in the dataset. Here are some of the most critically acclaimed titles you might enjoy:",
            "movies": self.rec._get_top_rated(5),
            "suggested_prompts": ["🍿 Surprise Me", "🌌 Sci-Fi Thrillers", "🎭 Crime Dramas", "😂 Feel-Good Comedy"],
            "mode": "local_fallback"
        }

    def _generate_followup_prompts(self, query, movies):
        """Generates dynamic, contextual follow-up chip prompts"""
        chips = []
        if movies:
            first_title = movies[0].get("title", "")
            if len(first_title) < 20:
                chips.append(f"🎬 Trailer for {first_title}")
                chips.append(f"🔍 Similar to {first_title}")
        chips.append("🍿 Surprise Me")
        chips.append("🔥 What's Trending?")
        return chips[:4]
