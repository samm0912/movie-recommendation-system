"""
chatbot.py — Intelligent Multi-Turn Conversational AI Movie Assistant (CineBot)
Behaves like a genuine, interactive AI movie companion:
  1. Real multi-turn conversation: Maintains session state (current movie, genre, mood, shown IDs, latest recommendations).
  2. Non-repeating movie pagination: Understands "Give me more" / "Show More Movies" and pulls next unseen set.
  3. Contextual mood & genre follow-ups: Understands "Make them more emotional", "Make them funnier", "More intense", etc.
  4. Ordinal movie details: Understands "Tell me about the first one", "Tell me about the 2nd one", etc., using real dataset data.
  5. Initial movie recommendations: "I liked Interstellar" immediately recommends 5 real movies from the ML recommendation engine.
  6. Grounded in real TMDB dataset with verified IMDb 0–10 ratings and instant trailer links.
  7. Natural chit-chat, jokes, trivia, opinions, and error handling without leaking keys or errors.
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

MOOD_MAP = {
    "emotional": {
        "label": "Deep & Emotional",
        "genres": ["Drama", "Romance", "History"],
        "keywords": ["emotional", "heartbreaking", "poignant", "father", "daughter", "tears", "loss", "devotion", "grief", "love", "touching"],
        "reply_intro": "Sure! Here are some more emotional options:"
    },
    "intense": {
        "label": "Intense & Thrilling",
        "genres": ["Thriller", "Action", "Crime", "Horror", "War"],
        "keywords": ["thriller", "suspense", "danger", "heist", "adrenaline", "chase", "explosive", "gritty", "dark"],
        "reply_intro": "Got it! Here are more high-intensity, thrilling options:"
    },
    "mind-bending": {
        "label": "Mind-Bending & Sci-Fi",
        "genres": ["Science Fiction", "Mystery", "Fantasy", "Thriller"],
        "keywords": ["time", "space", "dimension", "reality", "quantum", "puzzle", "twist", "psychological", "dream", "memory"],
        "reply_intro": "Brain-twisters incoming! Here are mind-bending and cerebral picks:"
    },
    "feel-good": {
        "label": "Feel-Good & Comedy",
        "genres": ["Comedy", "Animation", "Family", "Music", "Romance"],
        "keywords": ["funny", "hilarious", "laugh", "humor", "warm", "cheerful", "uplifting", "lighthearted"],
        "reply_intro": "Let's lighten the mood! Here are top feel-good and comedy options:"
    },
    "scary": {
        "label": "Scary & Horror",
        "genres": ["Horror", "Thriller", "Mystery"],
        "keywords": ["scary", "horror", "creepy", "ghost", "haunted", "supernatural", "fear", "dark"],
        "reply_intro": "Prepare for chills! Here are spine-tingling and dark options:"
    },
    "romantic": {
        "label": "Romantic & Love",
        "genres": ["Romance", "Drama"],
        "keywords": ["romance", "love", "couple", "passion", "sweetheart", "relationship"],
        "reply_intro": "Heartwarming love stories coming up! Here are romantic options:"
    },
    "action": {
        "label": "Action-Packed",
        "genres": ["Action", "Adventure"],
        "keywords": ["action", "explosive", "superhero", "combat", "fight", "stunts"],
        "reply_intro": "High-octane blockbusters ready! Here are action-packed options:"
    }
}


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

    # ── SESSION STATE INITIALIZATION & NORMALIZATION ─────────────────────────
    def _normalize_session_state(self, state, context_movie_id=None):
        """Ensures session_state has all necessary fields with correct types"""
        if not isinstance(state, dict):
            state = {}

        normalized = {
            "current_movie_id": state.get("current_movie_id"),
            "current_movie_title": state.get("current_movie_title"),
            "current_genres": list(state.get("current_genres", []) or []),
            "current_mood": state.get("current_mood"),
            "current_cast": state.get("current_cast"),
            "current_language": state.get("current_language"),
            "shown_movie_ids": list(state.get("shown_movie_ids", []) or []),
            "latest_recommended_movies": list(state.get("latest_recommended_movies", []) or []),
            "last_intent": state.get("last_intent")
        }

        # Override with context_movie_id if supplied from detail page
        if context_movie_id and not normalized["current_movie_id"]:
            m = self.rec.get_movie_by_id(context_movie_id)
            if m:
                normalized["current_movie_id"] = m["id"]
                normalized["current_movie_title"] = m["title"]
                normalized["current_genres"] = [g.strip() for g in str(m.get("genres", "")).split("|") if g.strip()]

        return normalized

    # ── CONTEXT & INTENT RESOLVER ────────────────────────────────────────────
    def _detect_ordinal_index(self, text):
        """
        Parses ordinal references to target items from latest results:
        e.g. "first one", "1st movie", "second", "third", "4th", "last one"
        Returns integer index (0, 1, 2, 3, 4, -1) or None.
        """
        t = text.lower()
        if re.search(r'\b(first(\s+one|\s+movie)?|1st(\s+one|\s+movie)?|number\s+1|#1|the\s+1st)\b', t):
            return 0
        if re.search(r'\b(second(\s+one|\s+movie)?|2nd(\s+one|\s+movie)?|number\s+2|#2|the\s+2nd)\b', t):
            return 1
        if re.search(r'\b(third(\s+one|\s+movie)?|3rd(\s+one|\s+movie)?|number\s+3|#3|the\s+3rd)\b', t):
            return 2
        if re.search(r'\b(fourth(\s+one|\s+movie)?|4th(\s+one|\s+movie)?|number\s+4|#4|the\s+4th)\b', t):
            return 3
        if re.search(r'\b(fifth(\s+one|\s+movie)?|5th(\s+one|\s+movie)?|number\s+5|#5|the\s+5th)\b', t):
            return 4
        if re.search(r'\b(last(\s+one|\s+movie)?|final(\s+one|\s+movie)?)\b', t):
            return -1
        return None

    def _detect_mood_shift(self, text):
        """
        Detects mood shift intents:
        e.g. "make them more emotional", "more funny", "more intense", "make them scarier"
        Returns mood key ('emotional', 'intense', 'mind-bending', 'feel-good', 'scary', 'romantic', 'action') or None.
        """
        t = text.lower()
        if re.search(r'\b(emotional|sad|tearjerker|drama|heartwarming|touching|cry|tear|feelings)\b', t):
            return "emotional"
        if re.search(r'\b(intense|thrilling|thriller|suspense|dark|gritty|edge\s+of\s+seat)\b', t):
            return "intense"
        if re.search(r'\b(mind.*bending|mindbending|sci-fi|scifi|cerebral|twist|time\s+travel|dimensions|brain.*twister|complex)\b', t):
            return "mind-bending"
        if re.search(r'\b(funnier|funny|comedy|laugh|humor|feel.*good|lighthearted|chill)\b', t):
            return "feel-good"
        if re.search(r'\b(scarier|scary|horror|creepy|ghost|spooky|haunted|frightening)\b', t):
            return "scary"
        if re.search(r'\b(romantic|romance|love|love\s+story|romcom)\b', t):
            return "romantic"
        if re.search(r'\b(action|action.*packed|explosive|superhero|fight|adrenaline)\b', t):
            return "action"
        return None

    # ── MOVIE EXTRACTION FROM NATURAL TEXT ───────────────────────────────────
    def _extract_movie_title_from_text(self, text):
        """
        Extracts candidate movie title from user phrasing like:
        - "I liked Interstellar"
        - "I love Inception"
        - "Movies like The Dark Knight"
        - "Recommend movies similar to Titanic"
        - "What about RRR?"
        - "Interstellar"
        """
        raw = text.strip()
        t_clean = re.sub(r'[^a-zA-Z0-9\s:’\'-]', '', raw).strip()

        # Remove leading phrasing patterns
        pref_patterns = [
            r'^(i\s+(really\s+)?(liked|like|loved|love|watched|enjoyed|am\s+a\s+fan\s+of|prefer)\s+)',
            r'^(recommend(\s+me)?\s+(some\s+)?movies?\s+(similar\s+to|like)\s+)',
            r'^(suggest(\s+me)?\s+(some\s+)?movies?\s+(similar\s+to|like)\s+)',
            r'^(movies?\s+(similar\s+to|like)\s+)',
            r'^(films?\s+(similar\s+to|like)\s+)',
            r'^(tell\s+me\s+about\s+)',
            r'^(what\s+do\s+you\s+think\s+of\s+)',
            r'^(what\s+about\s+)'
        ]

        candidate = t_clean
        for pat in pref_patterns:
            candidate = re.sub(pat, '', candidate, flags=re.IGNORECASE).strip()

        # Remove trailing punctuation / noise
        candidate = re.sub(r'\s+(movie|film|films|movies)$', '', candidate, flags=re.IGNORECASE).strip()
        return candidate if candidate else t_clean

    # ── MAIN CHAT HANDLER ───────────────────────────────────────────────────
    def chat(self, message, history=None, user=None, context_movie_id=None, session_state=None):
        """
        Main chat handler.
        Understands general conversation first, asks/answers follow-ups,
        remembers context across multiple turns, paginates without duplicates,
        and provides direct recommendations grounded in the 60,000+ dataset.
        """
        msg = str(message or "").strip()
        state = self._normalize_session_state(session_state, context_movie_id)
        tone = self.detect_tone(msg, history)

        if not msg:
            reply = (
                "Hi! 👋 Welcome! I’m your AI movie assistant.\n\n"
                "What kind of movie are you in the mood for today? You can tell me a movie you liked, a genre, an actor, or how you're feeling!"
            )
            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["🔥 Action Movies", "🌌 Sci-Fi Adventure", "😂 Comedy Hits", "🍿 Surprise Me"],
                "session_state": state,
                "mode": "greeting",
                "tone": tone
            }

        # Multi-turn local engine
        return self._conversational_pipeline(msg, history, user, tone, state)

    # ── CORE MULTI-TURN CONVERSATIONAL PIPELINE ──────────────────────────────
    def _conversational_pipeline(self, raw_msg, history, user, tone, state):
        q_lower = raw_msg.lower().strip()
        q_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', q_lower).strip()
        words = q_clean.split()
        word_count = len(words)

        user_display_name = user.get("name") if user else None
        if tone == "informal":
            name = user_display_name if user_display_name else "my friend"
        elif tone == "formal":
            name = user_display_name if user_display_name else "esteemed guest"
        else:
            name = user_display_name if user_display_name else "there"

        # ── 1. PERSONA SWITCH COMMANDS ────────────────────────────────────────
        if re.search(r'\b(speak|talk|be|switch\s+to|act)\s+(informally|casual|casually|like\s+a\s+friend|like\s+a\s+bro|like\s+my\s+bro|informal|chill)\b', q_lower):
            return {
                "success": True,
                "reply": f"😎 **Informal Mode Activated!**\n\n"
                         f"Alright **{name}**, gloves are off! We're talking pure movie-buff to movie-buff now. "
                         f"What crazy film or genre are we diving into today? Throw anything at me — hype action, mind-bending sci-fi, or late-night comedies! 🍿🔥",
                "movies": [],
                "suggested_prompts": ["🔥 Best Action Movies", "🌌 Movies like Interstellar", "🍿 Surprise Me Bro", "😂 Drop a Joke"],
                "session_state": state,
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
                "session_state": state,
                "mode": "persona_switch",
                "tone": "formal"
            }

        # ── 2. GENERAL CONVERSATION (GREETINGS, CHIT-CHAT, JOKES, TRIVIA) ──────
        q_reduced = re.sub(r'([a-zA-Z])\1{2,}', r'\1', q_lower)
        is_greeting = bool(
            re.match(r'^(hi|hey|hello|yo|sup|wassup|hola|namaste|greetings|howdy|heya|hiya|good morning|good evening|good afternoon|good day)\b', q_reduced) or
            re.match(r'^(hi|hey|hello|yo|sup|wassup|hola|namaste)\b', q_lower)
        )

        # Check if greeting has no movie search intent attached
        if is_greeting and word_count <= 6 and not any(k in q_lower for k in ["liked", "like", "movie", "watch", "recommend", "interstellar", "inception", "action", "horror", "comedy"]):
            if "morning" in q_lower:
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
            elif re.match(r'^yo\b', q_lower):
                greeting_word = "Yo! 🍿"
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
                "session_state": state,
                "mode": "greeting",
                "tone": tone
            }

        # How are you?
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
                "session_state": state,
                "mode": "conversation",
                "tone": tone
            }

        # Gratitude: "Thank you"
        if re.search(r'\b(thank\s+you|thanks|thank\s+u|thx|tysm|thank\s+you\s+so\s+much|appreciate\s+it|great\s+job|awesome\s+bot)\b', q_lower) and word_count <= 6:
            if tone == "informal":
                reply = f"You're super welcome, **{name}**! 🍿 Enjoy your movie! Let me know if you need anything else!"
            elif tone == "formal":
                reply = f"It is truly my pleasure, **{name}**. Enjoy your screening, and I remain at your service."
            else:
                reply = "You’re welcome! 🍿 Enjoy your movie!"

            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["➕ Show More Movies", "🍿 Surprise Me", "😂 Tell a Joke"],
                "session_state": state,
                "mode": "conversation",
                "tone": tone
            }

        # Boredom: "I'm bored"
        if re.search(r'\b(i\s+am\s+bored|im\s+bored|boring|bored|nothing\s+to\s+do)\b', q_lower) and not re.search(r'\b(movie|watch|recommend|film)\b', q_lower):
            reply = "Let’s fix that! 😄 Are you looking for something funny, thrilling, romantic, mysterious, or completely unexpected?"
            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["😂 Something Funny", "🔥 Thrilling Action", "❤️ Romantic", "🕵️ Mysterious", "🍿 Completely Unexpected"],
                "session_state": state,
                "mode": "conversation",
                "tone": tone
            }

        # Movie Jokes
        if re.search(r'\b(joke|jokes|make\s+me\s+laugh|tell.*joke|movie\s+joke|humor)\b', q_lower):
            setup, punchline = random.choice(MOVIE_JOKES)
            reply = f"😂 Here's a cinema joke for you:\n\n**{setup}**\n*{punchline}*\n\nWould you like another joke, or should we find a laugh-out-loud comedy movie to stream?"
            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["😂 Another Joke", "😂 Comedy Hits", "🍿 Surprise Me"],
                "session_state": state,
                "mode": "conversation",
                "tone": tone
            }

        # Movie Trivia
        if re.search(r'\b(trivia|trivia\s+fact|movie\s+fact|facts|did\s+you\s+know|tell\s+me\s+something\s+cool)\b', q_lower):
            fact = random.choice(MOVIE_TRIVIA)
            reply = f"🤯 **Cinema Trivia:**\n\n{fact}\n\nWant another cool fact, or should we find a movie to watch?"
            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["📜 Another Trivia Fact", "🍿 Surprise Me", "⭐ Top Rated"],
                "session_state": state,
                "mode": "conversation",
                "tone": tone
            }

        # Quotes
        if re.search(r'\b(quote|quotes|dialogue|dialogues|famous\s+line|movie\s+quote)\b', q_lower):
            q_text, q_meta = random.choice(MOVIE_QUOTES)
            reply = f"🎬 **Iconic Cinema Quote:**\n\n> *\"{q_text}\"*\n\n— **{q_meta}**"
            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["🎬 Another Quote", "🍿 Surprise Me", "⭐ Top Rated"],
                "session_state": state,
                "mode": "conversation",
                "tone": tone
            }

        # Who are you / Identity
        if re.search(r'\b(who\s+are\s+you|what\s+is\s+your\s+name|what\s+can\s+you\s+do|who\s+made\s+you|help|guide\s+me)\b', q_lower):
            reply = (
                f"🤖 I'm **CineBot**, your intelligent conversational movie assistant!\n\n"
                f"I remember our conversation context and recommend real movies from our verified 60,000+ dataset with IMDb ratings and instant trailers.\n\n"
                f"**Things you can say:**\n"
                f"• *\"I liked Interstellar\"* → Get tailored recommendations\n"
                f"• *\"Give me more\"* → Fetch the next set of unseen movies\n"
                f"• *\"Make them more emotional\"* or *\"Make them funnier\"* → Pivot the mood\n"
                f"• *\"Tell me about the first one\"* → Learn full details about any movie in the list"
            )
            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["🌌 I liked Interstellar", "🔥 Action Movies", "🍿 Surprise Me", "😂 Tell a Joke"],
                "session_state": state,
                "mode": "conversation",
                "tone": tone
            }

        # Snacks & Concessions
        if re.search(r'\b(what.*(eat|snack)|snacks?|popcorn|pizza|food|nachos)\b', q_lower) and word_count <= 15:
            reply = (
                "🍿 **Movie Snack Tier List:**\n\n"
                "1. 🧈 **Fresh Buttered Popcorn** with extra salt & caramel\n"
                "2. 🧀 **Loaded Nachos with Warm Cheese**\n"
                "3. 🍕 **Fresh Pizza Slice**\n"
                "4. 🍫 **Chilled M&Ms**\n\n"
                "Grab your snack and let me know what we're watching!"
            )
            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["🍿 Surprise Me", "🔥 Action Blockbusters", "🌌 Sci-Fi Adventure"],
                "session_state": state,
                "mode": "conversation",
                "tone": tone
            }

        # Farewells
        if re.search(r'\b(bye|goodbye|good\s+night|gn|see\s+you|catch\s+you\s+later|cya)\b', q_lower) and word_count <= 5:
            reply = f"🌙 Goodbye **{name}**! Enjoy your movie and have a wonderful time! 🍿✨"
            return {
                "success": True,
                "reply": reply,
                "movies": [],
                "suggested_prompts": ["🍿 Quick Surprise", "⭐ Top Rated"],
                "session_state": state,
                "mode": "conversation",
                "tone": tone
            }

        # ── 3. MOVIE OPINIONS & CRITIQUES ("What do you think of Interstellar?") ──
        if re.search(r'\b(what\s+do\s+you\s+think\s+of|opinion\s+on|thoughts\s+on|review\s+of|why\s+is.*famous)\b', q_lower) or ("interstellar" in q_lower and ("think" in q_lower or "opinion" in q_lower)):
            if "interstellar" in q_lower:
                reply = (
                    "🚀 **Interstellar? Absolute masterpiece, hands down!**\n\n"
                    "Between Hans Zimmer's pipe organ score and the physics of the black hole Gargantua, Christopher Nolan created one of the greatest sci-fi emotional journeys ever. "
                    "The scene where Cooper watches 23 years of video messages gives instant chills every single time. ⭐ Verified **8.4/10** on IMDb.\n\n"
                    "Here are some top works with similar depth and scale:"
                )
                recs = [m for m in self.rec.get_content_recommendations(157336, 5) if m.get("id") != 157336]
                state["current_movie_id"] = 157336
                state["current_movie_title"] = "Interstellar"
                state["latest_recommended_movies"] = recs
                for m in recs:
                    state["shown_movie_ids"].append(m["id"])
                return {
                    "success": True,
                    "reply": reply,
                    "movies": recs,
                    "suggested_prompts": ["➕ Show More Movies", "🎭 Make them more emotional", "🔍 Tell me about the first one"],
                    "session_state": state,
                    "mode": "opinion",
                    "tone": tone
                }

        # ── 4. ORDINAL MOVIE DETAILS INQUIRY ("Tell me about the first one") ──
        ord_idx = self._detect_ordinal_index(q_lower)
        is_detail_query = bool(
            ord_idx is not None or
            re.search(r'\b(tell\s+me\s+about|what\s+is\s+(the\s+)?(plot|story|synopsis|overview|details?|rating)\s+of|explain\s+the|info\s+on)\b', q_lower)
        )

        if is_detail_query and state["latest_recommended_movies"] and ord_idx is not None:
            latest = state["latest_recommended_movies"]
            if ord_idx == -1:
                target_m = latest[-1]
            elif 0 <= ord_idx < len(latest):
                target_m = latest[ord_idx]
            else:
                target_m = latest[0]

            # Fetch fresh metadata from recommender engine
            full_movie = self.rec.get_movie_by_id(target_m["id"]) or target_m
            genres_fmt = str(full_movie.get("genres", "")).replace("|", ", ")
            overview = full_movie.get("overview") or "No overview available for this title."
            lang_str = full_movie.get("language", "English")

            reply = (
                f"🎬 **{full_movie['title']}** ({full_movie.get('year', '')})\n\n"
                f"⭐ **IMDb Rating:** {full_movie.get('rating', '7.0')}/10 · 🌐 **Language:** {lang_str}\n"
                f"🎭 **Genres:** {genres_fmt}\n\n"
                f"📖 **Synopsis:**\n{overview}"
            )

            state["last_intent"] = "movie_detail"
            return {
                "success": True,
                "reply": reply,
                "movies": [full_movie],
                "suggested_prompts": [
                    f"🎬 More like {full_movie['title']}",
                    "➕ Show More Movies",
                    "🎭 Make them more emotional",
                    "🍿 Surprise Me"
                ],
                "session_state": state,
                "mode": "movie_details",
                "tone": tone
            }

        # ── 5. "MORE LIKE THIS" / "MORE LIKE THE FIRST ONE" ──────────────────
        if re.search(r'\b(more\s+like\s+this|more\s+like\s+the\s+first|more\s+like\s+the\s+second|similar\s+to\s+this)\b', q_lower):
            ref_movie_id = state.get("current_movie_id")
            if ord_idx is not None and state["latest_recommended_movies"]:
                if 0 <= ord_idx < len(state["latest_recommended_movies"]):
                    ref_movie_id = state["latest_recommended_movies"][ord_idx]["id"]

            if not ref_movie_id and state["latest_recommended_movies"]:
                ref_movie_id = state["latest_recommended_movies"][0]["id"]

            if ref_movie_id:
                ref_m = self.rec.get_movie_by_id(ref_movie_id)
                if ref_m:
                    state["current_movie_id"] = ref_m["id"]
                    state["current_movie_title"] = ref_m["title"]
                    state["current_genres"] = [g.strip() for g in str(ref_m.get("genres", "")).split("|") if g.strip()]

                    # Exclude already shown movies
                    shown_set = set(state["shown_movie_ids"])
                    shown_set.add(ref_m["id"])

                    raw_recs = self.rec.get_content_recommendations(ref_m["id"], n=25, language=state.get("current_language"))
                    unseen_recs = [m for m in raw_recs if m["id"] not in shown_set][:5]

                    if len(unseen_recs) < 5 and state["current_genres"]:
                        genre_fill = self.rec.get_by_genre(state["current_genres"][0], n=20, exclude_ids=list(shown_set))
                        for gm in genre_fill:
                            if gm["id"] not in shown_set and gm["id"] not in [x["id"] for x in unseen_recs]:
                                unseen_recs.append(gm)
                                if len(unseen_recs) >= 5:
                                    break

                    for m in unseen_recs:
                        state["shown_movie_ids"].append(m["id"])
                    state["latest_recommended_movies"] = unseen_recs
                    state["last_intent"] = "recommend_similar"

                    reply = f"Here are more movies with a similar vibe and story to **{ref_m['title']}**:"
                    return {
                        "success": True,
                        "reply": reply,
                        "movies": unseen_recs,
                        "suggested_prompts": ["➕ Show More Movies", "🎭 Make them more emotional", "🔍 Tell me about the first one", "🍿 Surprise Me"],
                        "session_state": state,
                        "mode": "similar_recommendations",
                        "tone": tone
                    }

        # ── 6. PAGINATION / "GIVE ME MORE" / "SHOW MORE MOVIES" ───────────────
        is_give_more = bool(
            re.search(r'\b(give\s+me\s+more|show\s+more(\s+movies)?|more\s+movies|more\s+picks|more\s+options|give\s+more|fetch\s+more|more\s+like\s+these|next(\s+set|\s+movies|\s+page)?)\b', q_lower) or
            q_clean in ["more", "give me more", "show more", "next", "more movies", "show more movies"]
        )

        if is_give_more:
            shown_set = set(state.get("shown_movie_ids", []))
            unseen = []

            # Strategy 1: Active movie content recommendations
            if state.get("current_movie_id"):
                mid = state["current_movie_id"]
                raw_pool = self.rec.get_content_recommendations(mid, n=40, language=state.get("current_language"))
                # If mood active, prioritize mood genres
                if state.get("current_mood") and state["current_mood"] in MOOD_MAP:
                    target_genres = [g.lower() for g in MOOD_MAP[state["current_mood"]]["genres"]]
                    mood_matches = [m for m in raw_pool if m["id"] not in shown_set and any(g.lower() in m.get("genres", "").lower() for g in target_genres)]
                    other_matches = [m for m in raw_pool if m["id"] not in shown_set and m not in mood_matches]
                    candidate_pool = mood_matches + other_matches
                else:
                    candidate_pool = [m for m in raw_pool if m["id"] not in shown_set]

                unseen.extend(candidate_pool[:5])

            # Strategy 2: Genre or Language pool
            if len(unseen) < 5 and state.get("current_genres"):
                for g in state["current_genres"][:2]:
                    genre_recs = self.rec.get_by_genre(g, n=25, language=state.get("current_language"), exclude_ids=list(shown_set))
                    for m in genre_recs:
                        if m["id"] not in shown_set and m["id"] not in [x["id"] for x in unseen]:
                            unseen.append(m)
                            if len(unseen) >= 5:
                                break
                    if len(unseen) >= 5:
                        break

            # Strategy 3: Cast pool
            if len(unseen) < 5 and state.get("current_cast"):
                cast_recs = self.rec.get_by_cast(state["current_cast"], n=25, language=state.get("current_language"))
                for m in cast_recs:
                    if m["id"] not in shown_set and m["id"] not in [x["id"] for x in unseen]:
                        unseen.append(m)
                        if len(unseen) >= 5:
                            break

            # Strategy 4: Trending fallback
            if len(unseen) < 5:
                trending_pool = self.rec.get_trending(40, language=state.get("current_language"))
                for m in trending_pool:
                    if m["id"] not in shown_set and m["id"] not in [x["id"] for x in unseen]:
                        unseen.append(m)
                        if len(unseen) >= 5:
                            break

            # Record newly shown IDs
            for m in unseen:
                state["shown_movie_ids"].append(m["id"])
            state["latest_recommended_movies"] = unseen
            state["last_intent"] = "give_more"

            if tone == "informal":
                reply = "Boom! Here are 5 more fresh picks from the catalog! 🍿🔥"
            elif tone == "formal":
                reply = "Certainly. Here is an additional curation of distinguished selections:"
            else:
                reply = "Absolutely! Here are some more movies you might enjoy:"

            return {
                "success": True,
                "reply": reply,
                "movies": unseen,
                "suggested_prompts": ["➕ Show More Movies", "🎭 Make them more emotional", "🔥 Make them more intense", "🔍 Tell me about the first one"],
                "session_state": state,
                "mode": "pagination",
                "tone": tone
            }

        # ── 7. ENTITY EXTRACTION & COMBINED FILTERS (LANGUAGE / GENRE / CAST) ──
        entities = self.rec.normalize_and_extract_entities(raw_msg)
        detected_lang = entities.get("detected_language")
        detected_genres = entities.get("detected_genres", [])
        detected_cast = entities.get("detected_cast")

        # A. Cast-based Requests (e.g. "Movies with Tom Hanks", "Prabhas movies")
        if detected_cast:
            state["current_cast"] = detected_cast
            if detected_lang:
                state["current_language"] = detected_lang
            if detected_genres:
                state["current_genres"] = detected_genres

            shown_set = set(state.get("shown_movie_ids", []))
            genre_name = detected_genres[0] if detected_genres else None
            cast_movies = self.rec.get_by_cast(
                detected_cast,
                n=15,
                genre=genre_name,
                language=detected_lang
            )
            unseen = [m for m in cast_movies if m["id"] not in shown_set][:5]
            if not unseen:
                unseen = cast_movies[:5]

            for m in unseen:
                state["shown_movie_ids"].append(m["id"])
            state["latest_recommended_movies"] = unseen
            state["last_intent"] = "cast_filter"

            if tone == "informal":
                reply = f"🌟 **{detected_cast} is an absolute icon!** Here are top-tier bangers starring {detected_cast} for you, my friend! 🍿🔥"
            elif tone == "formal":
                reply = f"🌟 **Distinguished Filmography of {detected_cast}:** Presented below is an esteemed curation of works from our collection:"
            else:
                reply = f"🌟 Here are top acclaimed movies starring **{detected_cast}** from our 60,000+ dataset:"

            return {
                "success": True,
                "reply": reply,
                "movies": unseen,
                "suggested_prompts": ["➕ Show More Movies", "🔍 Tell me about the first one", "🍿 Surprise Me"],
                "session_state": state,
                "mode": "cast_filter",
                "tone": tone
            }

        # B. Combined Language + Genre Filter (e.g. "Telugu horror movies", "Hindi action")
        if detected_lang and detected_genres:
            state["current_language"] = detected_lang
            state["current_genres"] = detected_genres
            genre_name = detected_genres[0]
            shown_set = set(state.get("shown_movie_ids", []))

            combined_recs = self.rec.get_by_genre_and_language(genre_name, detected_lang, limit=15)
            unseen = [m for m in combined_recs if m["id"] not in shown_set][:5]
            if not unseen:
                unseen = combined_recs[:5]

            for m in unseen:
                state["shown_movie_ids"].append(m["id"])
            state["latest_recommended_movies"] = unseen
            state["last_intent"] = "combined_filter"

            genres_title = " & ".join(detected_genres)
            if tone == "informal":
                reply = f"🎬 **Got you covered, {name}!** Here are top-tier **{detected_lang} {genres_title}** bangers matching both language and genre! 🍿🔥"
            elif tone == "formal":
                reply = f"🎬 **Bespoke Curation for {detected_lang} {genres_title} Cinema:** The following distinguished selections satisfy both criteria:"
            else:
                reply = f"🎬 Here are top acclaimed **{detected_lang} {genres_title}** movies from our dataset matching both criteria:"

            return {
                "success": True,
                "reply": reply,
                "movies": unseen,
                "suggested_prompts": ["➕ Show More Movies", "🔍 Tell me about the first one", f"🔥 More {detected_lang} Hits", "🍿 Surprise Me"],
                "session_state": state,
                "mode": "combined_filter",
                "tone": tone
            }

        # ── 8. MOOD REFINEMENT ("Make them more emotional", "Make them funnier") ──
        detected_mood = self._detect_mood_shift(q_lower)
        is_mood_refinement = bool(
            detected_mood and not detected_lang and (
                re.search(r'\b(make\s+them|more|something|give\s+me|turn|pivot|switch|want)\b', q_lower) or
                word_count <= 3
            )
        )

        if is_mood_refinement:
            state["current_mood"] = detected_mood
            mood_info = MOOD_MAP[detected_mood]
            target_genres = [g.lower() for g in mood_info["genres"]]
            shown_set = set(state.get("shown_movie_ids", []))
            unseen = []

            # 1. If we have an anchor movie, score candidates by similarity + mood genres
            if state.get("current_movie_id"):
                raw_similar = self.rec.get_content_recommendations(state["current_movie_id"], n=50, language=state.get("current_language"))
                # Filter for mood genres
                mood_matches = [m for m in raw_similar if m["id"] not in shown_set and any(g.lower() in str(m.get("genres", "")).lower() for g in target_genres)]
                unseen.extend(mood_matches[:5])

            # 2. If fewer than 5, pull from mood genres in dataset
            if len(unseen) < 5:
                for g in mood_info["genres"]:
                    g_recs = self.rec.get_by_genre(g, n=25, language=state.get("current_language"), exclude_ids=list(shown_set))
                    for m in g_recs:
                        if m["id"] not in shown_set and m["id"] not in [x["id"] for x in unseen]:
                            unseen.append(m)
                            if len(unseen) >= 5:
                                break
                    if len(unseen) >= 5:
                        break

            for m in unseen:
                state["shown_movie_ids"].append(m["id"])
            state["latest_recommended_movies"] = unseen
            state["last_intent"] = "mood_refinement"

            reply = mood_info.get("reply_intro", f"Sure! Here are some more {detected_mood} options:")
            return {
                "success": True,
                "reply": reply,
                "movies": unseen,
                "suggested_prompts": ["➕ Show More Movies", "🔍 Tell me about the first one", "🌀 Mind-Bending Options", "🍿 Surprise Me"],
                "session_state": state,
                "mode": "mood_refinement",
                "tone": tone
            }

        # ── 9. INITIAL MOVIE REQUEST ("I liked Interstellar", "Interstellar", "I liked Interstellar and Inception") ──
        prompt_res = self.rec.recommend_by_prompt(raw_msg, limit=6, language=state.get("current_language"))
        matched_movies = prompt_res.get("matched_movies", [])
        has_movie_intent = bool(
            re.search(r'\b(liked|like|loved|love|watched|enjoyed|favorite|similar|movies\s+like|films\s+like|recommend|fan\s+of)\b', q_lower) or
            (matched_movies and word_count <= 6)
        )

        if matched_movies and has_movie_intent:
            shown_set = set(state.get("shown_movie_ids", []))
            
            if len(matched_movies) == 1:
                target_m = matched_movies[0]
                state["current_movie_id"] = target_m["id"]
                state["current_movie_title"] = target_m["title"]
                state["current_genres"] = [g.strip() for g in str(target_m.get("genres", "")).split("|") if g.strip()]

                raw_recs = prompt_res.get("recommendations", [])
                unseen_recs = [m for m in raw_recs if m["id"] not in shown_set and m["id"] != target_m["id"]][:4]

                # Fallback if fewer than 4 recs
                if len(unseen_recs) < 4:
                    extra_recs = self.rec.get_content_recommendations(target_m["id"], n=10, language=state.get("current_language"))
                    for em in extra_recs:
                        if em["id"] not in shown_set and em["id"] != target_m["id"] and em["id"] not in [x["id"] for x in unseen_recs]:
                            unseen_recs.append(em)
                            if len(unseen_recs) >= 4:
                                break

                # Display input movie itself as #1, followed by top 4 related movies
                result_movies = [target_m] + unseen_recs

                for m in result_movies:
                    state["shown_movie_ids"].append(m["id"])
                state["latest_recommended_movies"] = result_movies
                state["last_intent"] = "movie_recommendation"

                if tone == "informal":
                    reply = f"Great choice, **{name}**! Here is **{target_m['title']}** (⭐ {target_m['rating']}/10) followed by top similar movies based on theme, genre, and storytelling:"
                elif tone == "formal":
                    reply = f"An exquisite selection. Presented below is **{target_m['title']}** (⭐ {target_m['rating']}/10) followed by a bespoke curation of distinguished akin works:"
                else:
                    reply = f"Great choice! Here is **{target_m['title']}** (⭐ {target_m['rating']}/10) followed by top movies you might enjoy based on theme and storytelling:"

            else:
                # Multi-movie preference (e.g. "I liked Interstellar and Inception")
                state["current_movie_id"] = matched_movies[0]["id"]
                state["current_movie_title"] = ' & '.join([m['title'] for m in matched_movies])
                
                raw_recs = prompt_res.get("recommendations", [])
                matched_ids = {m["id"] for m in matched_movies}
                needed_recs = max(1, 5 - len(matched_movies))
                unseen_recs = [m for m in raw_recs if m["id"] not in shown_set and m["id"] not in matched_ids][:needed_recs]

                result_movies = matched_movies + unseen_recs

                for m in result_movies:
                    state["shown_movie_ids"].append(m["id"])
                state["latest_recommended_movies"] = result_movies
                state["last_intent"] = "multi_movie_recommendation"

                titles_str = ' and '.join([f"**{m['title']}**" for m in matched_movies])

                if tone == "informal":
                    reply = f"Awesome taste, **{name}**! You like both {titles_str}! Here are your matched picks followed by combined recommendations:"
                elif tone == "formal":
                    reply = f"Distinguished preferences. Acknowledging your affinity for {titles_str}, presented below are the matched titles accompanied by synergistic recommendations:"
                else:
                    reply = f"Awesome taste! Recognizing your preference for {titles_str}, here are both movies followed by recommendations matching their combined themes:"

            return {
                "success": True,
                "reply": reply,
                "movies": result_movies,
                "suggested_prompts": [
                    "➕ Show More Movies",
                    "🎭 Make them more emotional",
                    "🌀 Mind-Bending Concepts",
                    "🔍 Tell me about the first one"
                ],
                "session_state": state,
                "mode": "movie_recommendation",
                "tone": tone
            }

        # ── 10. SINGLE GENRE OR LANGUAGE REQUESTS ─────────────────────────────
        if detected_genres and not detected_lang:
            genre_name = detected_genres[0]
            state["current_genres"] = detected_genres
            shown_set = set(state.get("shown_movie_ids", []))
            genre_recs = self.rec.get_by_genre(genre_name, n=15, exclude_ids=list(shown_set))
            unseen = [m for m in genre_recs if m["id"] not in shown_set][:5]
            if not unseen:
                unseen = genre_recs[:5]

            for m in unseen:
                state["shown_movie_ids"].append(m["id"])
            state["latest_recommended_movies"] = unseen
            state["last_intent"] = "genre_filter"

            if tone == "informal":
                reply = f"🎬 **Top-tier {genre_name} bangers coming right up, {name}! (⭐ IMDb 0–10 scale):**"
            elif tone == "formal":
                reply = f"🎬 **Distinguished Curation of Acclaimed {genre_name} Cinema (⭐ IMDb 0–10 scale):**\n\nIt is my distinct pleasure to present bespoke selections tailored to your courteous request:"
            else:
                reply = f"🎬 **Top-rated {genre_name} movies coming right up:**"

            return {
                "success": True,
                "reply": reply,
                "movies": unseen,
                "suggested_prompts": ["➕ Show More Movies", "🔍 Tell me about the first one", "🍿 Surprise Me"],
                "session_state": state,
                "mode": "genre_filter",
                "tone": tone
            }

        if detected_lang and not detected_genres:
            state["current_language"] = detected_lang
            shown_set = set(state.get("shown_movie_ids", []))
            lang_recs = self.rec.get_by_language(detected_lang, n=15)
            unseen = [m for m in lang_recs if m["id"] not in shown_set][:5]
            if not unseen:
                unseen = lang_recs[:5]

            for m in unseen:
                state["shown_movie_ids"].append(m["id"])
            state["latest_recommended_movies"] = unseen
            state["last_intent"] = "language_filter"

            if tone == "informal":
                reply = f"🎬 **Here are the top-rated {detected_lang} blockbusters you've gotta watch (⭐ IMDb 0–10 scale):**"
            elif tone == "formal":
                reply = f"🎬 **Distinguished Curation of Acclaimed {detected_lang} Cinema (⭐ IMDb 0–10 scale):**\n\nAllow me to present premier selections from our library:"
            else:
                reply = f"🎬 **Acclaimed {detected_lang} Cinema selections:**"

            return {
                "success": True,
                "reply": reply,
                "movies": unseen,
                "suggested_prompts": ["➕ Show More Movies", "🔍 Tell me about the first one", f"🔥 More {detected_lang} Hits"],
                "session_state": state,
                "mode": "language_filter",
                "tone": tone
            }

        # ── 11. SURPRISE ME ───────────────────────────────────────────────────
        if re.search(r'\b(surprise|random|pick one|choose for me|give me a movie)\b', q_lower):
            movie = self.rec.get_surprise_movie() or self.rec._get_top_rated(1)[0]
            state["shown_movie_ids"].append(movie["id"])
            state["latest_recommended_movies"] = [movie]
            state["current_movie_id"] = movie["id"]
            state["current_movie_title"] = movie["title"]

            reply = (
                f"🎉 **Boom! Here's your surprise pick, {name}:**\n\n"
                f"🎬 **{movie['title']}** ({movie['year']}) · *{movie.get('language', 'Cinema')}*\n"
                f"⭐ **IMDb Rating:** {movie['rating']}/10 · 🎭 **Genres:** {movie['genres'].replace('|', ', ')}\n\n"
                f"📝 *{movie.get('overview', '')[:220]}...*"
            )
            return {
                "success": True,
                "reply": reply,
                "movies": [movie],
                "suggested_prompts": ["🍿 Another Surprise", f"🔍 Similar to {movie['title']}", "▶ Watch Trailer"],
                "session_state": state,
                "mode": "surprise",
                "tone": tone
            }

        # ── 12. NATURAL SEARCH FALLBACK ───────────────────────────────────────
        prompt_res = self.rec.recommend_by_prompt(raw_msg, limit=8)
        search_movies = prompt_res.get("movies", [])
        if search_movies:
            shown_set = set(state.get("shown_movie_ids", []))
            unseen = [m for m in search_movies if m["id"] not in shown_set][:5]
            if not unseen:
                unseen = search_movies[:5]

            for m in unseen:
                state["shown_movie_ids"].append(m["id"])
            state["latest_recommended_movies"] = unseen
            state["last_intent"] = "search"

            reply = f"🎬 **Found some great titles matching *'{raw_msg}'*:**"
            return {
                "success": True,
                "reply": reply,
                "movies": unseen,
                "suggested_prompts": ["➕ Show More Movies", "🔍 Tell me about the first one", "🍿 Surprise Me"],
                "session_state": state,
                "mode": "search",
                "tone": tone
            }

        # ── 13. FINAL FRIENDLY FALLBACK ───────────────────────────────────────
        reply = "I'd love to help you find a great movie! 😊 Tell me a title you loved (e.g. *'I liked Interstellar'*), a genre (*'action'*, *'horror'*, *'comedy'*), an actor (*'Tom Hanks'*, *'Prabhas'*), or ask for a surprise!"
        return {
            "success": True,
            "reply": reply,
            "movies": [],
            "suggested_prompts": ["🌌 I liked Interstellar", "🎬 Telugu Horror", "🔥 Hindi Action", "🍿 Surprise Me"],
            "session_state": state,
            "mode": "conversation",
            "tone": tone
        }
