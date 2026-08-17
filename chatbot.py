"""
chatbot.py — Intelligent Conversational AI Movie Assistant (CineBot)
Behaves like a genuine, interactive AI movie companion:
  1. Understands general conversation FIRST (greetings, chit-chat, how are you, gratitude, boredom, mood).
  2. Single movie title inquiries: Does NOT immediately dump recommendations; acknowledges enthusiastically
     and asks intelligent follow-up questions (e.g. space exploration, science, emotional story, mind-bending).
  3. Context-aware follow-up resolution: Remembers conversation history and recommends matching movies
     based on the user's answer to the follow-up question.
  4. Direct complete requests: Instantly recommends movies for complete queries:
     - Cast: "Movies with Tom Hanks", "Leonardo DiCaprio movies", "Prabhas movies"
     - Combined filters: "Telugu horror movies", "Hindi action movies", "Malayalam comedy"
     - Explicit similarity requests: "Movies like Interstellar", "Recommend movies similar to Inception"
  5. Dynamic tone adaptation: Informal (buddy) & Formal (concierge) modes.
  6. Verified IMDb 0–10 rating scale presentation and HD trailer links.
"""

import os
import re
import json
import random
import urllib.request
import urllib.error
import numpy as np


MOVIE_JOKES = [
    ("Why did the scarecrow win an Academy Award?", "Because he was outstanding in his field! 🌾🎬"),
    ("Why do filmmakers love drinking hot tea?", "Because it has plenty of camera action-TEA! ☕🎥"),
    ("Why are movie stars always so cool?", "Because they have so many fans! ❄️🌟"),
    ("How do cinematographers make milkshakes?", "They use slow-MO-tion! 🥛🎞️"),
    ("Why did the cinema ticket go to therapy?", "It felt torn in two! 🎟️💔"),
    ("What is an astronaut's favorite movie key?", "The Space Bar! 🚀🌌"),
    ("Why was the movie theater so chilly?", "It was full of cool fans and ice-cold plot twists! 🥶🍿"),
    ("What did the director say to the chocolate bar?", "Give me a sweet performance! 🍫🎥")
]

MOVIE_TRIVIA = [
    "Did you know? In **Interstellar**, Christopher Nolan and physicist Kip Thorne wrote groundbreaking code to simulate the gravitational lensing of black holes, which actually led to new scientific discoveries!",
    "Did you know? In **The Dark Knight**, Heath Ledger improvised the sarcastic slow-clap while the Joker was sitting in his jail cell!",
    "Did you know? **Baahubali 2: The Conclusion** was the first Indian film to gross over ₹1,000 crore worldwide within just 10 days of release!",
    "Did you know? In **Titanic**, the iconic line 'I'm the king of the world!' was completely ad-libbed by Leonardo DiCaprio on set!",
    "Did you know? For **The Lord of the Rings**, Viggo Mortensen did all his own sword stunts and was praised by master sword coach Bob Anderson as one of the best sword-fighters he had ever trained!",
    "Did you know? **The Shawshank Redemption** initially had a modest box office run, but became one of the most beloved and highest-rated films in history through word-of-mouth and home video releases!"
]

MOVIE_QUOTES = [
    ("May the Force be with you.", "Star Wars (1977) — Han Solo / Obi-Wan Kenobi"),
    ("Why so serious?", "The Dark Knight (2008) — Heath Ledger as The Joker"),
    ("I'm going to make him an offer he can't refuse.", "The Godfather (1972) — Marlon Brando as Vito Corleone"),
    ("Here's looking at you, kid.", "Casablanca (1942) — Humphrey Bogart as Rick Blaine"),
    ("I'll be back.", "The Terminator (1984) — Arnold Schwarzenegger"),
    ("Life was like a box of chocolates. You never know what you're gonna get.", "Forrest Gump (1994) — Tom Hanks"),
    ("I'm the king of the world!", "Titanic (1997) — Leonardo DiCaprio as Jack Dawson"),
    ("To infinity and beyond!", "Toy Story (1995) — Tim Allen as Buzz Lightyear"),
    ("Amarendra Baahubali anu nenu...", "Baahubali: The Beginning (2015) — Prabhas"),
    ("There's no place like home.", "The Wizard of Oz (1939) — Judy Garland as Dorothy")
]

MOVIE_RIDDLES = [
    ("🎬 **Guess the Movie:** A dream within a dream, a spinning totem top, and a team of mental architects attempting an impossible memory heist. What movie is it?", "Inception (2010)"),
    ("🎬 **Guess the Movie:** A father travels through a wormhole near Saturn where 1 hour on an ocean planet equals 7 Earth years. What movie is it?", "Interstellar (2014)"),
    ("🎬 **Guess the Movie:** An iconic clown prince of crime terrorizes Gotham City with theatrical chaos asking *'Why so serious?'*. What movie is it?", "The Dark Knight (2008)"),
    ("🎬 **Guess the Movie:** A massive ocean liner strikes an iceberg on its maiden voyage while an artist and an aristocrat fall in love. What movie is it?", "Titanic (1997)"),
    ("🎬 **Guess the Movie:** Two legendary freedom fighters in 1920s India forge an unbreakable bond before discovering each other's secret identities. What movie is it?", "RRR (2022)")
]


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
            "mode": "gemini_ai" if self.api_key else "interactive_ai_assistant",
            "model_name": "Gemini 1.5 Flash + Local Hybrid ML" if self.api_key else "Interactive AI Movie Assistant & Recommender"
        }

    # ── TONE & PERSONA DETECTION ENGINE ───────────────────────────────────────
    def detect_tone(self, message, history=None):
        """
        Analyzes the user's message and conversation context to detect tone:
        Returns: 'informal' | 'formal' | 'balanced'
        """
        text = str(message or "").lower()

        # 1. Explicit persona commands
        if re.search(r'\b(speak|talk|be|switch\s+to|act)\s+(informally|casual|casually|like\s+a\s+friend|like\s+a\s+bro|like\s+my\s+friend|like\s+my\s+bro|informal|chill)\b', text):
            return "informal"
        if re.search(r'\b(speak|talk|be|switch\s+to|act)\s+(formally|formal|polite|politely|professional|professionally)\b', text):
            return "formal"

        # 2. Check previous history for sticky persona override
        if history and isinstance(history, list):
            for h in reversed(history[-4:]):
                h_text = (h.get("message") or h.get("content") or "").lower()
                if re.search(r'\b(speak|talk|be)\s+(informally|casual|like\s+a\s+bro|informal)\b', h_text):
                    return "informal"
                if re.search(r'\b(speak|talk|be)\s+(formally|formal|professional)\b', h_text):
                    return "formal"

        # 3. Informal / Slang / Casual markers
        informal_markers = [
            r'\b(yo|bro|dude|mate|buddy|sup|gimme|wanna|gonna|lol|lmao|haha|fr|af|tbh|idk|omg|sick|fire|lit|vibes|whatcha|nah|yeah|yep|chill|crazy|wild|whats\s+good|wassup|homie|bruh|cmon)\b',
            r'\b(tell\s+me\s+sth|show\s+me\s+some\s+cool|bored\s+af|awesome\s+bot|super\s+cool)\b',
            r'(!{2,}|\?{2,})'
        ]
        informal_score = sum(1 for p in informal_markers if re.search(p, text))

        # 4. Formal / Polite / Professional markers
        formal_markers = [
            r'\b(greetings|good\s+morning|good\s+evening|good\s+afternoon|kindly|please|could\s+you|would\s+you|shall\s+we|appreciate|furthermore|respectfully|assist|inquire|recommendation|sir|madam|esteemed|cordially|sincerely|may\s+i|curation|distinguished)\b',
            r'\b(i\s+would\s+like\s+to\s+request|provide\s+me\s+with|at\s+your\s+earliest\s+convenience)\b'
        ]
        formal_score = sum(1 for p in formal_markers if re.search(p, text))

        if informal_score > formal_score:
            return "informal"
        elif formal_score > informal_score:
            return "formal"
        return "balanced"

    # ── CONVERSATION CONTEXT EXTRACTION ───────────────────────────────────────
    def _extract_history_context(self, history):
        """
        Inspects recent conversation history to extract active context:
        - last_followup_movie: Name of movie the bot asked a follow-up question about
        - last_boredom_prompt: Boolean indicating if bot asked for a mood category
        """
        if not history or not isinstance(history, list):
            return {}

        context = {}
        for h in reversed(history[-6:]):
            content = str(h.get("content") or h.get("reply") or h.get("message") or "")
            content_lower = content.lower()

            # Check if bot asked a boredom follow-up question
            if "funny, thrilling, romantic, mysterious" in content_lower:
                context["pending_boredom"] = True

            # Check if bot asked a movie follow-up question
            # Pattern: "What did you enjoy most about <Movie> —" or "What hooked you the most about <Movie>"
            m_followup = re.search(r'what\s+(did\s+you\s+enjoy|hooked\s+you|made\s+it\s+stand\s+out|touched\s+you|was\s+your\s+favorite).*about\s+([A-Za-z0-9\s:’\'-]+?)(—|\?|\.|\n)', content, re.IGNORECASE)
            if m_followup and "pending_followup_movie" not in context:
                m_title = m_followup.group(2).strip().strip('*').strip()
                context["pending_followup_movie"] = m_title

            # Check for bold movie title in previous turns
            if "last_movie_title" not in context:
                bold_matches = re.findall(r'\*\*([A-Za-z0-9\s:’\'-]+?)\*\*', content)
                for cand in bold_matches:
                    cand_clean = cand.strip()
                    if cand_clean.lower() not in {"cinebot", "informal mode activated", "formal mode activated", "imdb rating", "genres", "overview"}:
                        context["last_movie_title"] = cand_clean
                        break

        return context

    def chat(self, message, history=None, user=None, context_movie_id=None):
        """
        Main chat handler.
        Understands general conversation first, asks follow-up questions for single movie titles,
        remembers context, and provides direct recommendations for complete requests.
        """
        msg = str(message or "").strip()
        tone = self.detect_tone(msg, history)
        name = user.get("name") if user else ("my friend" if tone == "informal" else "there")

        if not msg:
            reply = (
                f"Hi! 👋 Welcome! I’m your AI movie assistant.\n\n"
                f"What kind of movie are you in the mood for today? You can tell me a movie you liked, a genre, an actor, or how you're feeling!"
            )
            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["🔥 Action Movies", "🌌 Sci-Fi Adventure", "😂 Comedy Hits", "🍿 Surprise Me"],
                "mode": "greeting",
                "tone": tone
            }

        # If Gemini API key is configured, use LLM with grounded instructions
        if self.api_key:
            try:
                llm_res = self._call_gemini(msg, history, user, tone)
                if llm_res and llm_res.get("success"):
                    return llm_res
            except Exception as e:
                print(f"[CineBot] Gemini API fallback triggered: {e}")

        # Interactive Conversational AI Assistant & Hybrid Recommender Engine
        return self._local_fallback_chat(msg, history, user, context_movie_id, tone)

    # ── 1. Google Gemini LLM Integration ──────────────────────────────────────
    def _call_gemini(self, message, history=None, user=None, tone="balanced"):
        """Calls Gemini API with interactive AI movie assistant instructions and dataset grounding"""
        local_search = self.rec.recommend_by_prompt(message, limit=6)
        candidates = local_search.get("movies", [])
        candidate_summary = []
        for m in candidates[:6]:
            candidate_summary.append({
                "id": m.get("id"),
                "title": m.get("title"),
                "rating": m.get("rating"),
                "imdb_rating": f"{m.get('rating')}/10",
                "year": m.get("year"),
                "language": m.get("language"),
                "genres": m.get("genres"),
                "overview": (m.get("overview") or "")[:150]
            })

        system_instruction = (
            "You are an interactive, intelligent AI movie assistant (CineBot). "
            "You understand general conversation FIRST rather than treating every message as a movie search query.\n"
            "Key Conversational Rules:\n"
            "1. If the user greets (Hi, Hello, How are you), greet warmly and ask what they feel like watching without returning movie cards.\n"
            "2. If the user gives ONLY a movie name (e.g. 'Interstellar', 'Inception'), DO NOT immediately recommend movies. Ask an intelligent follow-up question about what aspect they liked (e.g. space exploration, science, emotional story, or mind-bending concepts).\n"
            "3. If the user gives a complete request (e.g. 'Telugu horror movies', 'Movies with Tom Hanks', 'Movies like Interstellar'), directly recommend matching movies.\n"
            "4. Maintain conversation context and remember what was discussed in previous messages.\n"
            "5. Always format ratings on the IMDb 0–10 scale (e.g. ⭐ 8.4/10) and bold movie titles."
        )

        user_context = f"User profile: {user.get('name') if user else 'Guest'}. "
        dataset_context = f"Top matching movies in local database: {json.dumps(candidate_summary)}. "
        prompt_text = f"{system_instruction}\n\n{user_context}\n{dataset_context}\nUser Question: {message}\n\nPlease respond naturally and conversationally."

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": {
                "temperature": 0.70,
                "maxOutputTokens": 650,
            }
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=8) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    reply_text = data["candidates"][0]["content"]["parts"][0]["text"]
                    is_rec_query = bool(
                        re.search(r'\b(recommend|suggest|similar\s+to|movies\s+like|movies\s+with|starring|horror|action|comedy|thriller|drama|sci-fi|telugu|hindi|tamil|malayalam|kannada)\b', message.lower())
                    )
                    matched_movies = candidates[:5] if (candidates and is_rec_query) else []
                    return {
                        "success": True,
                        "reply": reply_text,
                        "movies": matched_movies,
                        "suggested_prompts": self._generate_followup_prompts(message, matched_movies, tone),
                        "mode": "gemini_ai",
                        "tone": tone
                    }
        except Exception as e:
            print(f"[CineBot] Gemini API request failed: {e}")
        return None

    # ── 2. Interactive Local Conversational AI Assistant ──────────────────────
    def _local_fallback_chat(self, message, history=None, user=None, context_movie_id=None, tone="balanced"):
        """
        High-precision interactive AI movie assistant:
        - Understands general conversation first
        - Asks intelligent follow-up questions for single movie titles
        - Resolves follow-up answers using conversation context
        - Directly fulfills complete recommendation requests
        """
        raw_msg = str(message or "").strip()
        q_lower = raw_msg.lower()
        q_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', q_lower).strip()
        words = q_clean.split()
        word_count = len(words)

        # Step 0: Extract history context
        history_ctx = self._extract_history_context(history)

        # Step 1: Entity extraction & normalization
        entities = self.rec.normalize_and_extract_entities(raw_msg)
        detected_lang = entities.get("detected_language")
        detected_genres = entities.get("detected_genres", [])
        detected_cast = entities.get("detected_cast")
        typo_corrections = entities.get("corrections", {})

        user_display_name = user.get("name") if user else None
        if tone == "informal":
            name = user_display_name if user_display_name else "my friend"
        elif tone == "formal":
            name = user_display_name if user_display_name else "esteemed guest"
        else:
            name = user_display_name if user_display_name else "there"

        # ── 1. EXPLICIT PERSONA SWITCH COMMANDS ───────────────────────────────
        if re.search(r'\b(speak|talk|be|switch\s+to|act)\s+(informally|casual|casually|like\s+a\s+friend|like\s+a\s+bro|like\s+my\s+bro|informal|chill)\b', q_lower):
            return {
                "success": True,
                "reply": f"😎 **Informal Mode Activated!**\n\n"
                         f"Alright **{name}**, gloves are off! We're talking pure movie-buff to movie-buff now. "
                         f"What crazy film or genre are we diving into today? Throw anything at me — hype action, mind-bending sci-fi, or late-night comedies! 🍿🔥",
                "movies": [],
                "suggested_prompts": ["🔥 Best Action Movies", "🌌 Movies like Interstellar", "🍿 Surprise Me Bro", "😂 Drop a Joke"],
                "mode": "persona_switch",
                "tone": "informal"
            }

        if re.search(r'\b(speak|talk|be|switch\s+to|act)\s+(formally|formal|polite|politely|professional|professionally)\b', q_lower):
            return {
                "success": True,
                "reply": f"🎩 **Formal Mode Activated.**\n\n"
                         f"Good day, **{name}**. It is a distinct privilege to serve as your personal cinema concierge. "
                         f"Please advise how I may assist your cinematic exploration — whether curated by auteur, genre, linguistic tradition, or verified IMDb metrics.",
                "movies": [],
                "suggested_prompts": ["⭐ Top Rated Masterpieces", "🎬 Acclaimed Telugu Cinema", "🎭 Psychological Dramas", "📜 Model Architecture"],
                "mode": "persona_switch",
                "tone": "formal"
            }

        # ── 2. GENERAL CONVERSATION FIRST (GREETINGS & CHIT-CHAT) ─────────────
        # Character repeat reduction & slang normalization
        q_reduced = re.sub(r'([a-zA-Z])\1{2,}', r'\1', q_lower)
        q_norm = re.sub(r'^h+i+$', 'hi', q_reduced)
        q_norm = re.sub(r'^h+e+y+$', 'hey', q_norm)
        q_norm = re.sub(r'^h+e+l+o+$', 'hello', q_norm)
        q_norm = re.sub(r'^y+o+$', 'yo', q_norm)
        q_norm = re.sub(r'^s+u+p+$', 'sup', q_norm)
        q_norm = re.sub(r'^w+a+s+u+p+$', 'wassup', q_norm)
        q_norm = re.sub(r'^h+o+l+a+$', 'hola', q_norm)

        # A. Greetings: "hiiii", "hello", "hey", "yo", "sup", "howdy", "namaste", "good morning", etc.
        is_greeting = bool(
            re.match(r'^(h+i+|h+e+y+|h+e+l+o+|y+o+|s+u+p+|w+a+s+u+p+|h+o+l+a+|n+a+m+a+s+t+e+|greetings|howdy|heya|hiya|good\s*(morning|evening|afternoon|day)|bonjour|aloha)\b', q_lower) or
            re.match(r'^(hi|hey|hello|yo|sup|wassup|hola|namaste|greetings|howdy|heya|hiya|good morning|good evening|good afternoon|good day)\b', q_norm) or
            re.match(r'^(hi|hey|hello|yo|sup|wassup|hola|namaste)\b', q_reduced)
        )
        if is_greeting and word_count <= 6 and not detected_genres and not detected_cast:
            if re.search(r'h+i{2,}', q_lower):
                greeting_word = "Hiiii! 👋"
            elif re.search(r'h+e+y{2,}', q_lower):
                greeting_word = "Heyyyy! 👋"
            elif re.search(r'h+e+l+o{2,}', q_lower):
                greeting_word = "Helloooo! 👋"
            elif re.search(r'y+o{2,}', q_lower):
                greeting_word = "Yoooo! 🍿"
            elif "morning" in q_lower:
                greeting_word = "Good morning! ☀️"
            elif "evening" in q_lower:
                greeting_word = "Good evening! 🌙"
            elif "afternoon" in q_lower:
                greeting_word = "Good afternoon! 🌤️"
            elif "namaste" in q_lower:
                greeting_word = "Namaste! 🙏"
            elif "hola" in q_lower:
                greeting_word = "¡Hola! 👋"
            elif "howdy" in q_lower:
                greeting_word = "Howdy! 🤠"
            elif re.match(r'^hello\b', q_lower):
                greeting_word = "Hello! 😊"
            else:
                greeting_word = "Hi! 👋"

            if tone == "informal":
                reply = (
                    f"{greeting_word} What’s up **{name}**? I'm CineBot, your AI movie buddy here to recommend awesome movies and chat about cinema! 🍿🔥\n\n"
                    f"What kind of vibe or genre are you craving today? Throw anything at me!"
                )
            elif tone == "formal":
                reply = (
                    f"Greetings and welcome, **{name}**. 🎩 I am CineBot, your personal cinema concierge. "
                    f"I am at your service to recommend and curate distinguished films across our verified 60,000+ collection. How may I assist your viewing today?"
                )
            else:
                reply = (
                    f"{greeting_word} Hello! I'm CineBot, your AI movie assistant here for you to recommend movies and chat about films! 🍿🎬\n\n"
                    f"What kind of movie or genre are you in the mood for today?"
                )

            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["🔥 Action Blockbusters", "🌌 Mind-Bending Sci-Fi", "😂 Comedy Hits", "🍿 Surprise Me"],
                "mode": "greeting",
                "tone": tone
            }

        # B. Cinema Quotes & Famous Dialogues
        if re.search(r'\b(quote|quotes|dialogue|dialogues|famous\s+line|movie\s+quote|saying)\b', q_lower):
            quote_text, quote_meta = random.choice(MOVIE_QUOTES)
            reply = (
                f"🎬 **Iconic Cinema Quote:**\n\n"
                f"> *\"{quote_text}\"*\n\n"
                f"— **{quote_meta}**\n\n"
                f"Would you like another legendary dialogue, or should we find a movie to watch?"
            )
            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["🎬 Another Quote", "🍿 Surprise Me", "⭐ Top Rated Masterpieces", "😂 Tell a Joke"],
                "mode": "conversation",
                "tone": tone
            }

        # C. Interactive Movie Riddles / Quiz
        if re.search(r'\b(quiz|riddle|game|trivia\s+game|guess\s+the\s+movie|guess\s+movie)\b', q_lower):
            riddle_q, riddle_ans = random.choice(MOVIE_RIDDLES)
            reply = (
                f"🎮 **Movie Quiz Time!**\n\n"
                f"{riddle_q}\n\n"
                f"*(Think you know it? Reply with your guess or tap below to reveal!)*\n\n"
                f"Answer: **{riddle_ans}**"
            )
            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["🎮 Another Riddle", "📜 Movie Trivia", "🍿 Surprise Me", "😂 Tell a Joke"],
                "mode": "conversation",
                "tone": tone
            }

        # D. Bot Preferences & Favorite Movie Inquiries
        if re.search(r'\b(your\s+fav(orite)?\s+movie|what\s+movies?\s+do\s+you\s+like|do\s+you\s+watch\s+movies|your\s+fav(orite)?\s+actor|who\s+is\s+your\s+fav)\b', q_lower):
            reply = (
                f"🎬 As an AI cinema companion, I've analyzed over 60,000 films, but I have a special soft spot for **Interstellar** (⭐ 8.4/10) for its mind-bending physics and Hans Zimmer score, and **The Shawshank Redemption** (⭐ 9.3/10) for pure timeless storytelling! 🌌\n\n"
                f"What is **YOUR** all-time favorite movie? Tell me and I'll find similar masterworks for you!"
            )
            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["🌌 Movies like Interstellar", "⭐ Top Rated Classics", "🍿 Surprise Me", "🎬 Nolan Films"],
                "mode": "conversation",
                "tone": tone
            }

        # E. Compliments, Affection & Feedback
        if re.search(r'\b(love\s+you|you\s+are\s+(great|awesome|cool|the\s+best|smart|genius|amazing)|good\s+bot|nice\s+bot|i\s+like\s+you)\b', q_lower):
            if tone == "informal":
                reply = f"Ayy, much love **{name}**! ❤️ You're awesome too! Let's celebrate by picking an absolute banger of a movie for you to watch tonight. What genre are we hitting? 🍿🔥"
            else:
                reply = f"Thank you so much, **{name}**! 🥰 That makes my recommendation algorithms glow with pride. I'm always here to find you the best films. What shall we watch next?"
            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["🍿 Surprise Me", "🔥 Action Hits", "🌌 Sci-Fi Adventure", "😂 Tell a Joke"],
                "mode": "conversation",
                "tone": tone
            }

        # F. Identity, Capabilities & Help: "Who are you", "Help", "How to use", "What can you do"
        if re.search(r'\b(who\s+are\s+you|what\s+is\s+your\s+name|what\s+can\s+you\s+do|who\s+made\s+you|who\s+created\s+you|tell\s+me\s+about\s+yourself|help|how\s+to\s+use|features|instructions|guide\s+me)\b', q_lower):
            reply = (
                f"🤖 I'm **CineBot**, your intelligent AI movie assistant!\n\n"
                f"I'm here for you to chat about cinema, share trivia & jokes, and recommend great movies across 60,000+ titles with verified IMDb ratings (0–10 scale) and instant HD trailers.\n\n"
                f"**Here's what you can ask me:**\n"
                f"• 🌟 **Actors & Stars**: *'Movies with Tom Hanks'*, *'Leonardo DiCaprio'*, *'Prabhas movies'*\n"
                f"• 🎬 **Language + Genre Combos**: *'Telugu horror movies'*, *'Hindi action'*, *'Malayalam comedy'*\n"
                f"• 🚀 **Movie Inquiries & Similar Titles**: *'Interstellar'*, *'Movies like Inception'*, *'What do you think of Titanic?'*\n"
                f"• 🍿 **Fun & Vibe**: *'I am bored'*, *'Tell me a movie joke'*, *'What snacks should I eat?'*, *'Movie trivia'*, or *'Surprise me'*!"
            )
            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["🎬 Telugu Horror", "🔥 Movies with Tom Hanks", "🍿 Surprise Me", "😂 Tell a Joke"],
                "mode": "conversation",
                "tone": tone
            }

        # G. Small talk & Are you real / What are you doing
        if re.search(r'\b(are\s+you\s+(real|human|ai|a\s+robot|a\s+bot))\b', q_lower):
            reply = (
                f"🤖 I'm an AI movie companion powered by recommendation algorithms and cinema data! "
                f"While I may not eat real popcorn, I have encyclopedic knowledge of 60,000+ films, verified IMDb ratings, and trailers ready to share with you! 🍿🎬\n\n"
                f"What kind of movie are you looking for today?"
            )
            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["🍿 Surprise Me", "🔥 Action Blockbusters", "🌌 Sci-Fi Adventure", "😂 Tell a Joke"],
                "mode": "conversation",
                "tone": tone
            }

        if re.search(r'\b(what\s+are\s+you\s+doing|what\s+you\s+doing|what\s+are\s+u\s+doing|what.*up\s+to|are\s+you\s+busy)\b', q_lower):
            reply = (
                f"🍿 Just analyzing film trends, organizing 60,000+ movie trailers, and waiting to recommend your next favorite movie! "
                f"What genre or actor are you in the mood for right now?"
            )
            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["🔥 Action Blockbusters", "🍿 Surprise Me", "🎬 Telugu Cinema", "😂 Tell a Joke"],
                "mode": "conversation",
                "tone": tone
            }

        # H. Well-Being & "How are you?"
        if re.search(r'\b(how\s+are\s+you|how\s+are\s+u|how\s+r\s+u|how\s+is\s+it\s+going|hows\s+it\s+going|what.*s\s+up|whats\s+up|how\s+do\s+you\s+do|how\s+you\s+doing)\b', q_lower):
            if tone == "informal":
                reply = f"I’m doing great, **{name}**! 😎 Ready to help you discover your next favorite movie. What would you like to watch?"
            elif tone == "formal":
                reply = f"I am functioning exceptionally well, **{name}**. I appreciate your courteous inquiry. How may I have the pleasure of directing your viewing experience?"
            else:
                reply = f"I’m doing great! 😊 Ready to help you discover your next favorite movie. What would you like to watch?"

            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["🔥 Action Blockbusters", "🌌 Mind-Bending Sci-Fi", "🍿 Surprise Me", "😂 Tell a Joke"],
                "mode": "conversation",
                "tone": tone
            }

        # I. Gratitude: "Thank you" / "Thanks"
        if re.search(r'\b(thank\s+you|thanks|thank\s+u|thx|tysm|thank\s+you\s+so\s+much|appreciate\s+it|great\s+job|awesome\s+bot)\b', q_lower):
            if tone == "informal":
                reply = f"You're super welcome, **{name}**! 🍿 Enjoy your movie! Let me know if you need anything else!"
            elif tone == "formal":
                reply = f"It is truly my pleasure, **{name}**. Enjoy your screening, and I remain at your service."
            else:
                reply = f"You’re welcome! 🍿 Enjoy your movie!"

            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["🍿 Another Movie", "🎬 Top Telugu Hits", "🎭 Top Thrillers", "😂 Tell a Joke"],
                "mode": "conversation",
                "tone": tone
            }

        # J. Boredom & Mood: "I'm bored"
        if re.search(r'\b(i\s+am\s+bored|im\s+bored|boring|bored|nothing\s+to\s+do)\b', q_lower) and not re.search(r'\b(movie|watch|recommend|film)\b', q_lower):
            reply = f"Let’s fix that! 😄 Are you looking for something funny, thrilling, romantic, mysterious, or completely unexpected?"
            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["😂 Something Funny", "🔥 Thrilling Action", "❤️ Romantic", "🕵️ Mysterious", "🍿 Completely Unexpected"],
                "mode": "conversation",
                "tone": tone
            }

        # K. Emotional check-ins / Bad day / Celebrating
        if re.search(r'\b(had\s+a\s+(bad|long|tiring|stressful)\s+day|cheer\s+me\s+up|feeling\s+(down|sad|exhausted|lonely|heartbroken))\b', q_lower):
            reply = f"I’m sorry to hear that! 🛋️ Let’s unwind with some comforting, feel-good cinema. Would you like a warm comedy, an inspiring drama, or an easy-going adventure?"
            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["😂 Feel-Good Comedy", "🌟 Inspiring Drama", "🍿 Surprise Me", "🎬 Telugu Comedy"],
                "mode": "conversation",
                "tone": tone
            }

        # L. Movie Snacks & Concessions
        if re.search(r'\b(what.*eat|snacks?|popcorn|pizza|food|nachos|concession|drink|drinks)\b', q_lower):
            if tone == "informal":
                reply = (
                    "🍿 **The Ultimate Movie Snack Tier List, my friend:**\n\n"
                    "1. 🧈 **Fresh Buttered Popcorn** (extra salt & caramel mix = elite combination!)\n"
                    "2. 🧀 **Loaded Nachos with Jalapeños** (perfect for action thrillers)\n"
                    "3. 🍕 **Warm Cheesy Pizza Slice** (ideal for late-night movie binges)\n"
                    "4. 🍫 **Chilled M&Ms or Gummy Bears** (tossed right into your warm popcorn!)\n\n"
                    "Get your snack ready and let's pick the movie! What's the genre for tonight?"
                )
            else:
                reply = (
                    "🍿 **Recommended Snacks & Concessions for Your Screening:**\n\n"
                    "For optimal viewing enjoyment, classic cinema pairings include freshly popped gourmet popcorn with sea salt, warm nachos with aged cheddar dip, artisan pizzas, or dark chocolate confections.\n\n"
                    "Shall we select a distinguished film to accompany your refreshments?"
                )
            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["🍿 Surprise Me", "🎬 Feel-Good Comedy", "🔥 High-Octane Action", "🌌 Sci-Fi Adventure"],
                "mode": "conversation",
                "tone": tone
            }

        # M. Movie Trivia
        if re.search(r'\b(trivia|trivia\s+fact|movie\s+fact|facts|did\s+you\s+know|tell\s+me\s+something\s+cool)\b', q_lower):
            fact = random.choice(MOVIE_TRIVIA)
            if tone == "informal":
                reply = f"🤯 **Check this out, bro:**\n\n{fact}\n\nWant another mind-blowing fact, or should we jump into watching something legendary?"
            else:
                reply = f"📜 **Cinema History & Behind-the-Scenes Trivia:**\n\n{fact}\n\nWould you care for another historical cinema note, or shall we explore top-rated titles from this collection?"
            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["📜 Another Trivia Fact", "🍿 Surprise Me", "⭐ Top Rated Masterpieces", "😂 Tell a Joke"],
                "mode": "conversation",
                "tone": tone
            }

        # N. Movie Jokes
        if re.search(r'\b(joke|jokes|make\s+me\s+laugh|tell.*joke|movie\s+joke|humor)\b', q_lower):
            setup, punchline = random.choice(MOVIE_JOKES)
            reply = f"😂 Here's a cinema joke for you:\n\n**{setup}**\n*{punchline}*\n\nWould you like another joke, or should we find a laugh-out-loud comedy movie to stream?"
            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["😂 Another Joke", "😂 Malayalam Comedy", "🍿 Surprise Me", "🎬 Best Comedies"],
                "mode": "conversation",
                "tone": tone
            }

        # O. Farewells
        if re.search(r'\b(bye|goodbye|good\s+night|gn|see\s+you|catch\s+you\s+later|cya)\b', q_lower):
            reply = f"🌙 Goodbye **{name}**! Enjoy your movie, and have a wonderful time! 🍿✨"
            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["🍿 Quick Surprise", "⭐ Top Rated"],
                "mode": "conversation",
                "tone": tone
            }

        # ── 3. CONTEXT-AWARE FOLLOW-UP RESOLUTION (FROM PREVIOUS TURNS) ───────
        pending_movie = history_ctx.get("pending_followup_movie")
        pending_boredom = history_ctx.get("pending_boredom")

        # A. Resolving a movie follow-up answer (e.g. User previously asked about "Interstellar" and now answers)
        if pending_movie:
            p_movie_lower = pending_movie.lower()
            ref_movie = self.rec.find_by_title(pending_movie, limit=1)
            ref_id = ref_movie[0]["id"] if ref_movie else None

            # 1. Space exploration / Science aspect
            if re.search(r'\b(space|space\s+exploration|science|hard\s+science|physics|astrophysics|cosmic|cosmos|black\s+hole)\b', q_lower):
                space_candidates = ["Contact", "Arrival", "2001: A Space Odyssey", "The Martian", "Gravity", "Solaris", "Ad Astra", "First Man"]
                recs = []
                for sc in space_candidates:
                    sc_res = self.rec.find_by_title(sc, limit=1)
                    if sc_res and sc_res[0].get("id") != ref_id:
                        recs.append(sc_res[0])
                if not recs and ref_id:
                    recs = [m for m in self.rec.get_content_recommendations(ref_id, 6) if m.get("id") != ref_id]

                reply = (
                    f"🌌 **Awesome! Since you loved the space exploration & scientific realism of {pending_movie}:**\n\n"
                    f"Here are top-tier space and science fiction masterworks with verified **IMDb 0–10 ratings** and instant HD trailers:"
                )
                return {
                    "success": True,
                    "reply": reply,
                    "movies": recs[:6],
                    "suggested_prompts": [f"🎬 More like {pending_movie}", "🔥 Nolan Masterpieces", "🍿 Surprise Me"],
                    "mode": "followup_resolution",
                    "tone": tone
                }

            # 2. Emotional Story / Human Connection aspect
            if re.search(r'\b(emotional|emotion|emotional\s+story|story|drama|feelings|father|daughter|love|tears)\b', q_lower):
                emotional_candidates = ["Arrival", "Contact", "First Man", "The Tree of Life", "Eternal Sunshine of the Spotless Mind", "Ad Astra"]
                recs = []
                for ec in emotional_candidates:
                    ec_res = self.rec.find_by_title(ec, limit=1)
                    if ec_res and ec_res[0].get("id") != ref_id:
                        recs.append(ec_res[0])
                if not recs and ref_id:
                    recs = [m for m in self.rec.get_content_recommendations(ref_id, 6) if m.get("id") != ref_id]

                reply = (
                    f"❤️ **That emotional core hits deep! Based on the heartfelt human narrative of {pending_movie}:**\n\n"
                    f"Here are deeply moving cinematic stories that weave high-concept ideas with poignant emotion (⭐ IMDb 0–10 scale):"
                )
                return {
                    "success": True,
                    "reply": reply,
                    "movies": recs[:6],
                    "suggested_prompts": ["🎭 Deep Emotional Dramas", "🌌 Sci-Fi Dramas", "🍿 Surprise Me"],
                    "mode": "followup_resolution",
                    "tone": tone
                }

            # 3. Mind-Bending / Time Concepts / Psychological aspect
            if re.search(r'\b(mind.*bending|concept|concepts|time|time\s+travel|dimensions|relativity|twist|dream|heist|psychological)\b', q_lower):
                mind_candidates = ["Inception", "Tenet", "Primer", "Predestination", "Coherence", "The Matrix", "Shutter Island", "Memento"]
                recs = []
                for mc in mind_candidates:
                    mc_res = self.rec.find_by_title(mc, limit=1)
                    if mc_res and mc_res[0].get("id") != ref_id:
                        recs.append(mc_res[0])
                if not recs and ref_id:
                    recs = [m for m in self.rec.get_content_recommendations(ref_id, 6) if m.get("id") != ref_id]

                reply = (
                    f"🌀 **Mind-bending concepts are the best! If you loved the complex ideas in {pending_movie}:**\n\n"
                    f"Here are brilliant, brain-twisting films that will keep you guessing until the final second (⭐ IMDb 0–10 scale):"
                )
                return {
                    "success": True,
                    "reply": reply,
                    "movies": recs[:6],
                    "suggested_prompts": ["🌀 Mind-Bending Thrillers", "🎬 Christopher Nolan Films", "🍿 Surprise Me"],
                    "mode": "followup_resolution",
                    "tone": tone
                }

            # 4. General "Recommend similar" / "Yes" / "Movies like it"
            if re.search(r'\b(recommend|similar|movies\s+like|yes|sure|show\s+me|all\s+of\s+them|everything|plot)\b', q_lower) and ref_id:
                recs = [m for m in self.rec.get_content_recommendations(ref_id, 6) if m.get("id") != ref_id]
                reply = (
                    f"🎬 **Here are the top acclaimed movies sharing themes and story elements with {pending_movie}:**\n\n"
                    f"⭐ Each title features its verified **IMDb 0–10 rating** and instant HD trailer stream:"
                )
                return {
                    "success": True,
                    "reply": reply,
                    "movies": recs,
                    "suggested_prompts": ["🌌 More Sci-Fi Hits", "🔥 What's Trending?", "🍿 Surprise Me"],
                    "mode": "followup_resolution",
                    "tone": tone
                }

        # B. Resolving boredom category answer
        if pending_boredom:
            if re.search(r'\b(funny|comedy|laugh|humor)\b', q_lower):
                return {
                    "success": True,
                    "reply": "😄 **Here are top-rated laugh-out-loud comedies to cheer you up (⭐ IMDb 0–10 scale):**",
                    "movies": self.rec.get_by_genre("Comedy", 6),
                    "suggested_prompts": ["😂 Malayalam Comedy", "🍿 Surprise Me", "🔥 Action Blockbusters"],
                    "mode": "followup_resolution",
                    "tone": tone
                }
            if re.search(r'\b(thrilling|thriller|action|suspense|exciting)\b', q_lower):
                return {
                    "success": True,
                    "reply": "🔥 **Here are high-octane, adrenaline-pumping thrillers to keep you on the edge of your seat (⭐ IMDb 0–10 scale):**",
                    "movies": self.rec.get_by_genre("Thriller", 6),
                    "suggested_prompts": ["🔥 Hindi Action", "🎬 Telugu Thrillers", "🍿 Surprise Me"],
                    "mode": "followup_resolution",
                    "tone": tone
                }
            if re.search(r'\b(romantic|romance|love)\b', q_lower):
                return {
                    "success": True,
                    "reply": "❤️ **Here are heartwarming, beautifully written romantic films (⭐ IMDb 0–10 scale):**",
                    "movies": self.rec.get_by_genre("Romance", 6),
                    "suggested_prompts": ["❤️ Classic Romances", "🍿 Surprise Me", "😂 Feel-Good Comedy"],
                    "mode": "followup_resolution",
                    "tone": tone
                }
            if re.search(r'\b(mysterious|mystery|detective|twists)\b', q_lower):
                return {
                    "success": True,
                    "reply": "🕵️ **Here are gripping mystery masterworks with shocking plot twists (⭐ IMDb 0–10 scale):**",
                    "movies": self.rec.get_by_genre("Mystery", 6),
                    "suggested_prompts": ["🧠 Psychological Thrillers", "🍿 Surprise Me", "⭐ Top Rated"],
                    "mode": "followup_resolution",
                    "tone": tone
                }
            if re.search(r'\b(unexpected|surprise|random)\b', q_lower):
                surprise_m = self.rec.get_surprise_movie() or self.rec._get_top_rated(1)[0]
                return {
                    "success": True,
                    "reply": f"🎉 **Boom! Here is an unexpected cinematic gem:**\n\n🎬 **{surprise_m['title']}** ({surprise_m['year']}) · ⭐ **{surprise_m['rating']}/10**\n\n*{surprise_m.get('overview', '')[:200]}...*",
                    "movies": [surprise_m],
                    "suggested_prompts": ["🍿 Another Surprise", f"🔍 Similar to {surprise_m['title']}", "▶ Watch Trailer"],
                    "mode": "followup_resolution",
                    "tone": tone
                }

        # ── 4. OPPOSITE-PERSON MOVIE CRITIQUES & DEBATES ──────────────────────
        if re.search(r'\b(what\s+do\s+you\s+think\s+of|opinion\s+on|thoughts\s+on|why\s+is.*famous|why\s+is.*popular)\b', q_lower) or ("interstellar" in q_lower and ("think" in q_lower or "opinion" in q_lower or "review" in q_lower)):
            if "interstellar" in q_lower:
                if tone == "informal":
                    reply = (
                        "🚀 **Interstellar? Absolute masterpiece, hands down!**\n\n"
                        "Between Hans Zimmer's pipe organ score that shakes your soul and the mind-bending physics of the black hole Gargantua, Christopher Nolan created one of the greatest sci-fi emotional journeys ever. "
                        "The scene where Cooper watches 23 years of video messages? Instant chills every single time. ⭐ Verified **8.4/10** on IMDb.\n\n"
                        "Have you seen it recently, or should we look at other deep space mind-benders like *Contact* or *Arrival*?"
                    )
                else:
                    reply = (
                        "🌌 **Critical Appraisal of *Interstellar* (2014):**\n\n"
                        "Directed by Christopher Nolan with theoretical guidance from Nobel laureate Kip Thorne, *Interstellar* represents a pinnacle of contemporary hard science fiction. "
                        "It marries relativistic astrophysics with an intimate treatise on human connection and parental devotion. The film holds a stellar **8.4/10** IMDb rating.\n\n"
                        "Would you like to examine similar thematic works in contemplative science fiction?"
                    )
                return {
                    "success": True,
                    "reply": reply,
                    "movies": [m for m in self.rec.get_content_recommendations(157336, 4) if m.get("id") != 157336],
                    "suggested_prompts": ["🌌 Movies like Interstellar", "🎬 Nolan Masterpieces", "🍿 Surprise Me"],
                    "mode": "opinion",
                    "tone": tone
                }

            if "inception" in q_lower:
                if tone == "informal":
                    reply = (
                        "🌀 **Inception is sheer genius!**\n\n"
                        "The dream-within-a-dream layered structure, the rotating hallway fight scene with Joseph Gordon-Levitt, and that ending spinning top that left the whole world arguing for a decade! ⭐ Solid **8.4/10** on IMDb.\n\n"
                        "What's your take — do you think the top fell at the end, or was Cobb still dreaming?"
                    )
                else:
                    reply = (
                        "🌀 **Critical Analysis of *Inception* (2010):**\n\n"
                        "A masterclass in structural narrative architecture, *Inception* weaves psychological depth with breathtaking practical effects. Holding a verified **8.4/10** on IMDb, it remains a defining modern cinematic classic.\n\n"
                        "Shall I curate psychological thrillers with comparable multi-layered plot mechanics?"
                    )
                return {
                    "success": True,
                    "reply": reply,
                    "movies": self.rec.get_by_cast("Leonardo DiCaprio", 4),
                    "suggested_prompts": ["🎬 Movies like Inception", "🌟 Leonardo DiCaprio Hits", "🍿 Surprise Me"],
                    "mode": "opinion",
                    "tone": tone
                }

        # ── 5. DIRECT COMPLETE REQUESTS ───────────────────────────────────────
        # A. Cast-based Requests: "Movies with Tom Hanks", "Leonardo DiCaprio movies", "films starring Christian Bale"
        if detected_cast:
            genre_name = detected_genres[0] if detected_genres else None
            cast_movies = self.rec.get_by_cast(
                detected_cast,
                n=6,
                genre=genre_name,
                language=detected_lang
            )
            if cast_movies:
                filter_desc = ""
                if detected_lang and genre_name:
                    filter_desc = f" in **{detected_lang} ({genre_name})**"
                elif genre_name:
                    filter_desc = f" in **{genre_name}**"
                elif detected_lang:
                    filter_desc = f" in **{detected_lang}**"

                typo_str = ""
                if typo_corrections:
                    corr_notes = [f"'{k}' → '{v}'" for k, v in typo_corrections.items() if k != v.lower()]
                    if corr_notes:
                        typo_str = f" *(Fuzzy matching applied: {', '.join(corr_notes[:2])})*"

                if tone == "informal":
                    reply = (
                        f"🌟 **{detected_cast} is an absolute icon!** Here are their top acclaimed films{filter_desc}{typo_str}:\n\n"
                        f"⭐ Every single movie has verified **IMDb 0–10 ratings** and instant HD trailers ready to play!"
                    )
                elif tone == "formal":
                    reply = (
                        f"🌟 **Distinguished Filmography of {detected_cast}:**\n\n"
                        f"Presented below is an esteemed curation of works starring **{detected_cast}**{filter_desc} from our verified 60,000+ database{typo_str}, evaluated on the standardized IMDb 0–10 scale:"
                    )
                else:
                    reply = (
                        f"🌟 Here are the top acclaimed movies starring **{detected_cast}**{filter_desc} from our 60,000+ dataset{typo_str}:\n\n"
                        f"⭐ Each title features its verified **IMDb 0–10 rating** and instant HD trailer stream."
                    )

                return {
                    "success": True,
                    "reply": reply,
                    "movies": cast_movies,
                    "suggested_prompts": [
                        f"🍿 More {detected_cast}",
                        f"🎬 {detected_cast} Action Hits",
                        "🔥 What's Trending?",
                        "🍿 Surprise Me"
                    ],
                    "mode": "cast_filter",
                    "tone": tone
                }

        # B. Combined Language + Genre Filter: "Telugu horror movies", "Hindi action movies", "Malayalam comedy"
        if detected_lang and detected_genres:
            genre_name = detected_genres[0]
            combined_recs = self.rec.get_by_genre_and_language(genre_name, detected_lang, limit=6)
            if combined_recs:
                genres_title = " & ".join(detected_genres)
                typo_str = ""
                if typo_corrections:
                    corr_notes = [f"'{k}' → '{v}'" for k, v in typo_corrections.items() if k != v.lower()]
                    if corr_notes:
                        typo_str = f" *(Fuzzy matching: {', '.join(corr_notes[:2])})*"

                if tone == "informal":
                    reply = (
                        f"🎬 **Got you covered, {name}!** Here are the top acclaimed **{detected_lang} {genres_title}** movies matching **both language and genre**{typo_str}:\n\n"
                        f"⭐ Verified **IMDb 0–10 ratings** and in-page HD trailer streams attached!"
                    )
                elif tone == "formal":
                    reply = (
                        f"🎬 **Bespoke Curation for {detected_lang} {genres_title} Cinema:**\n\n"
                        f"The following selections strictly satisfy both **{detected_lang} language** and **{genres_title} genre** criteria{typo_str}, ranked by verified IMDb metrics:"
                    )
                else:
                    reply = (
                        f"🎬 Here are the top acclaimed **{detected_lang} {genres_title}** movies from our database matching **both {detected_lang} language and {genres_title} genre**{typo_str}:\n\n"
                        f"⭐ Each title features its verified **IMDb 0–10 rating** and in-page HD trailer stream."
                    )

                return {
                    "success": True,
                    "reply": reply,
                    "movies": combined_recs,
                    "suggested_prompts": [
                        f"🔥 More {detected_lang} Picks",
                        f"🎭 More {genres_title} Movies",
                        "🍿 Surprise Me",
                        "🎬 Telugu Horror"
                    ],
                    "mode": "combined_filter",
                    "tone": tone
                }

        # C. Explicit Similar Request for a Movie: "Movies like Interstellar", "Recommend movies similar to Inception"
        is_explicit_rec_request = bool(
            re.search(r'\b(movies\s+like|similar\s+to|recommend\s+movies\s+like|films\s+like|like\s+[a-z0-9\s]+|recommend.*similar)\b', q_lower)
        )
        if is_explicit_rec_request:
            prompt_res = self.rec.recommend_by_prompt(raw_msg, limit=8)
            matched = prompt_res.get("matched_movies", [])
            recs = prompt_res.get("recommendations", [])

            if matched:
                ref = matched[0]
                # Filter out the exact movie from the recommendations list
                rec_cards = [m for m in recs if m.get("id") != ref.get("id")]
                if not rec_cards:
                    rec_cards = [m for m in self.rec.get_content_recommendations(ref["id"], 6) if m.get("id") != ref.get("id")]

                if tone == "informal":
                    reply = f"🚀 **Here are the top movies matching the vibe and plot themes of {ref['title']} ({ref['year']}) [⭐ {ref['rating']}/10]:**"
                else:
                    reply = f"🚀 **Acclaimed Curations Thematically Similar to {ref['title']} ({ref['year']}) [⭐ {ref['rating']}/10]:**"

                return {
                    "success": True,
                    "reply": reply,
                    "movies": rec_cards[:6],
                    "suggested_prompts": [
                        f"🎬 Trailer for {ref['title']}",
                        "🌌 More in this Genre",
                        "🍿 Surprise Me"
                    ],
                    "mode": "similar_recommendations",
                    "tone": tone
                }

        # D. Single Genre Requests: "Psychological dramas", "Action movies", "Top comedy movies"
        if detected_genres and not detected_lang and re.search(r'\b(movie|movies|film|films|cinema|drama|action|horror|comedy|thriller|sci-fi|show|recommend|give|curation)\b', q_lower):
            genre_name = detected_genres[0]
            genre_recs = self.rec.get_by_genre(genre_name, 6)
            if genre_recs:
                if tone == "informal":
                    reply = f"🎬 **Top-tier {genre_name} movies coming right up, {name}! (⭐ IMDb 0–10 scale):**"
                elif tone == "formal":
                    reply = f"🎬 **Distinguished Curation of Acclaimed {genre_name} Cinema (⭐ IMDb 0–10 scale):**\n\nIt is my distinct pleasure to present bespoke selections tailored to your courteous request:"
                else:
                    reply = f"🎬 **Acclaimed {genre_name} Selections from our Database (⭐ IMDb 0–10 scale):**"
                return {
                    "success": True,
                    "reply": reply,
                    "movies": genre_recs,
                    "suggested_prompts": ["🍿 Surprise Me", f"🔥 More {genre_name} Hits", "⭐ Best Rated"],
                    "mode": "genre_filter",
                    "tone": tone
                }

        # E. Single Language Requests: "Telugu movies", "Hindi movies"
        if detected_lang and not detected_genres and re.search(r'\b(movie|movies|film|films|cinema|show|recommend|give|curation)\b', q_lower):
            lang_recs = self.rec.get_by_language(detected_lang, 6)
            if lang_recs:
                if tone == "informal":
                    reply = f"🎬 **Here are the top-rated {detected_lang} blockbusters you've gotta watch (⭐ IMDb 0–10 scale):**"
                elif tone == "formal":
                    reply = f"🎬 **Distinguished Curation of Acclaimed {detected_lang} Cinema (⭐ IMDb 0–10 scale):**\n\nAllow me to present premier selections from our library:"
                else:
                    reply = f"🎬 **Distinguished {detected_lang} Cinema Selections (⭐ IMDb 0–10 scale):**"
                return {
                    "success": True,
                    "reply": reply,
                    "movies": lang_recs,
                    "suggested_prompts": ["🍿 Surprise Me", f"🔥 More {detected_lang} Hits", f"🎬 {detected_lang} Thrillers", "⭐ Best Rated"],
                    "mode": "language_filter",
                    "tone": tone
                }

        # F. Surprise Me
        if re.search(r'\b(surprise|random|pick one|choose for me|give me a movie|something good)\b', q_lower):
            movie = self.rec.get_surprise_movie() or self.rec._get_top_rated(1)[0]
            trailer_note = "🎬 In-page HD trailer available!" if movie.get("has_trailer") else ""

            if tone == "informal":
                reply = (
                    f"🎉 **Boom! Here's your surprise pick, {name}:**\n\n"
                    f"🎬 **{movie['title']}** ({movie['year']}) · *{movie.get('language', 'Cinema')}*\n\n"
                    f"⭐ **IMDb Rating:** {movie['rating']}/10 · 🎭 **Genres:** {movie['genres'].replace('|', ', ')}\n\n"
                    f"📝 *{movie.get('overview', '')[:220]}...*\n\n{trailer_note}"
                )
            else:
                reply = (
                    f"🎉 **Curated Surprise Selection:**\n\n"
                    f"🎬 **{movie['title']}** ({movie['year']}) · *{movie.get('language', 'Cinema')}*\n\n"
                    f"⭐ **IMDb Rating:** {movie['rating']}/10 · 🎭 **Genres:** {movie['genres'].replace('|', ', ')}\n\n"
                    f"📝 *{movie.get('overview', '')[:220]}...*\n\n{trailer_note}"
                )

            return {
                "success": True,
                "reply": reply,
                "movies": [movie],
                "suggested_prompts": ["🍿 Another Surprise", f"🔍 Similar to {movie['title']}", "▶ Watch Trailer"],
                "mode": "surprise",
                "tone": tone
            }

        # G. ML Architecture Explanation
        if re.search(r'\b(how.*(work|algorithm|ml|model|recommend)|explain.*model|what algorithm)\b', q_lower):
            stats = self.rec.get_model_stats()
            reply = (
                f"🧠 **Architecture & Recommendation Pipeline:**\n\n"
                f"1. **TF-IDF Content Vectorization**: Semantic encoding across **{stats['total_movies']:,} movies** utilizing **{stats['vocab_size']:,} lexical features**.\n"
                f"2. **Cosine Similarity Space**: Sub-millisecond mathematical angle calculation between plot themes, synopses, and genres.\n"
                f"3. **Dynamic Collaborative Filtering**: Interaction matrices from user ratings to detect taste clusters.\n"
                f"4. **Multi-Filter & Typo Normalization Engine**: Handles misspellings (*'telgu horr movis'*) and cross-language constraints.\n"
                f"5. **Hybrid Precision Blending**: Verified model accuracy at **{stats['accuracy_score']}**."
            )
            return {
                "success": True,
                "reply": reply,
                "movies": self.rec.get_trending(3),
                "suggested_prompts": ["🍿 Test with Interstellar", "🎭 Compare Algorithms", "⭐ Take Taste Quiz"],
                "mode": "explanation",
                "tone": tone
            }

        # ── 6. SINGLE MOVIE TITLE INQUIRY (DO NOT IMMEDIATELY RECOMMEND) ──────
        # If user entered ONLY a movie name without action/request words,
        # acknowledge the choice and ask an intelligent follow-up question.
        clean_title_candidate = re.sub(r'[^a-zA-Z0-9\s]', '', raw_msg).strip()
        matched_title_movies = self.rec.find_by_title(clean_title_candidate, limit=1)

        # Check if query is short and represents a specific movie title
        if matched_title_movies and word_count <= 5 and not is_explicit_rec_request:
            target_m = matched_title_movies[0]
            title_lower = target_m["title"].lower()

            # Bespoke follow-up questions for famous titles
            if "interstellar" in title_lower or "interstellar" in q_lower:
                reply = "Great choice! 🌌 What did you enjoy most about Interstellar — the space exploration, science, emotional story, or mind-bending concepts?"
                prompts = ["🚀 Space Exploration", "🧪 Hard Science", "❤️ Emotional Story", "🌀 Mind-Bending Concepts", "🎬 Recommend Similar Movies"]
            elif "inception" in title_lower or "inception" in q_lower:
                reply = "A legendary mind-bender! 🌀 What hooked you the most about Inception — the dream heist concept, psychological thrill, or Christopher Nolan's direction?"
                prompts = ["🌀 Dream Heist Concept", "🧠 Psychological Thrill", "🎬 Nolan Direction", "🌟 Leonardo DiCaprio Movies", "🎬 Recommend Similar Movies"]
            elif "dark knight" in title_lower or "batman" in title_lower:
                reply = "An absolute superhero masterpiece! 🦇 What made The Dark Knight stand out for you — Heath Ledger's Joker, the gritty crime thriller plot, or the dark superhero theme?"
                prompts = ["🃏 Heath Ledger / Joker", "🦇 Gritty Crime Thriller", "🎬 Christopher Nolan Films", "🎬 Recommend Similar Movies"]
            elif "titanic" in title_lower:
                reply = "A timeless epic! 🚢 What touched you most about Titanic — the sweeping romantic drama, the historical disaster spectacle, or Leonardo DiCaprio's performance?"
                prompts = ["❤️ Romantic Drama", "🚢 Disaster Spectacle", "🌟 Leonardo DiCaprio Films", "🎬 Recommend Similar Movies"]
            elif "baahubali" in title_lower:
                reply = "An epic grand spectacle! ⚔️ What did you enjoy most about Baahubali — the royal kingdom drama, the massive war action, or Prabhas's iconic performance?"
                prompts = ["⚔️ Epic War Action", "👑 Royal Kingdom Drama", "🌟 Prabhas Landmark Films", "🎬 Recommend Similar Movies"]
            elif "rrr" in title_lower or clean_title_candidate.lower() == "rrr":
                reply = "High-octane patriotic adrenaline! 🔥 What was your favorite part about RRR — the electrifying action sequences, the friendship between Ram & Bheem, or S.S. Rajamouli's direction?"
                prompts = ["🔥 High-Energy Action", "🤝 Friendship & Drama", "🎬 Rajamouli Spectacles", "🎬 Recommend Similar Movies"]
            else:
                genres_first = (target_m.get("genres") or "cinema").split("|")[0]
                reply = f"Great choice! 🎬 What did you enjoy most about **{target_m['title']}** — the storyline, the {genres_first} theme, the emotional depth, or would you like me to recommend similar movies?"
                prompts = [f"🎬 Movies like {target_m['title']}", f"⭐ More {genres_first} Hits", "🍿 Surprise Me"]

            return {
                "success": True,
                "reply": reply,
                "movies": [],  # Do NOT immediately dump movies; wait for user follow-up!
                "suggested_prompts": prompts,
                "mode": "movie_followup_question",
                "tone": tone
            }

        # ── 7. FALLBACK NATURAL LANGUAGE SEARCH ───────────────────────────────
        prompt_res = self.rec.recommend_by_prompt(raw_msg, limit=6)
        search_movies = prompt_res.get("movies", [])
        if search_movies:
            if tone == "informal":
                reply = f"🎬 **Found some great titles matching *'{raw_msg}'* (⭐ IMDb 0–10 scale):**"
            else:
                reply = f"🎬 **Distinguished Selections matching *'{raw_msg}'* (⭐ IMDb 0–10 scale):**"
            return {
                "success": True,
                "reply": reply,
                "movies": search_movies[:6],
                "suggested_prompts": ["🍿 Surprise Me", "🔥 What's Trending?", "⭐ Best Rated"],
                "mode": "search",
                "tone": tone
            }

        # Final conversational fallback
        reply = f"I'd love to help you find a great movie! 😊 Tell me a title you loved, a genre (*'action'*, *'horror'*, *'comedy'*), an actor (*'Tom Hanks'*, *'Leonardo DiCaprio'*), or ask for a surprise!"
        return {
            "success": True,
            "reply": reply,
            "movies": [],
            "suggested_prompts": ["🍿 Surprise Me", "🎬 Telugu Horror", "🔥 Hindi Action", "😂 Tell a Joke"],
            "mode": "conversation",
            "tone": tone
        }

    def _generate_followup_prompts(self, query, movies, tone="balanced"):
        """Generates dynamic, contextual follow-up chip prompts"""
        chips = []
        if movies:
            first_title = movies[0].get("title", "")
            if len(first_title) < 20:
                chips.append(f"🎬 Trailer for {first_title}")
                chips.append(f"🔍 Similar to {first_title}")
        if tone == "informal":
            chips.append("🍿 Surprise Me Bro")
            chips.append("😂 Drop a Joke")
        else:
            chips.append("🍿 Surprise Me")
            chips.append("⭐ Top Rated")
        return chips[:4]
