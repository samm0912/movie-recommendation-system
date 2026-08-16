"""
build_dataset.py — Merge, clean, normalize, deduplicate, and cluster multi-language movies.
Generates:
  - movie-recommendation/data/movies_merged.csv (60,000+ movies with canonical_id & available_languages)
  - movie-recommendation/data/movie_meta_cache.json (Preprocessed metadata cache with language_variants)
"""

import os
import re
import json
import urllib.parse
from collections import defaultdict
import numpy as np
import pandas as pd

LANG_CODE_MAP = {
    'en': 'English', 'hi': 'Hindi', 'te': 'Telugu', 'ta': 'Tamil',
    'kn': 'Kannada', 'ml': 'Malayalam', 'bn': 'Bengali', 'mr': 'Marathi',
    'pa': 'Punjabi', 'gu': 'Gujarati', 'ur': 'Urdu', 'ja': 'Japanese',
    'ko': 'Korean', 'es': 'Spanish', 'fr': 'French', 'it': 'Italian',
    'de': 'German', 'zh': 'Chinese', 'cn': 'Chinese', 'ru': 'Russian',
    'pt': 'Portuguese', 'sv': 'Swedish', 'da': 'Danish', 'no': 'Norwegian',
    'pl': 'Polish', 'nl': 'Dutch', 'tr': 'Turkish', 'th': 'Thai',
    'ar': 'Arabic', 'id': 'Indonesian', 'fa': 'Persian'
}

INDIAN_LANG_MAP = {
    'hindi': 'Hindi', 'telugu': 'Telugu', 'tamil': 'Tamil',
    'kannada': 'Kannada', 'malayalam': 'Malayalam', 'bengali': 'Bengali',
    'marathi': 'Marathi', 'punjabi': 'Punjabi', 'gujarati': 'Gujarati',
    'urdu': 'Urdu', 'bhojpuri': 'Bhojpuri', 'oriya': 'Odia',
    'assamese': 'Assamese', 'nepali': 'Nepali', 'sanskrit': 'Sanskrit',
    'rajastani': 'Rajasthani', 'kashmiri': 'Kashmiri', 'konkani': 'Konkani',
    'tulu': 'Tulu', 'english': 'English'
}


def parse_year(val):
    if pd.isna(val):
        return 2000
    m = re.search(r'\b(19\d\d|20\d\d)\b', str(val))
    return int(m.group(1)) if m else 2000


def parse_timing(val):
    if pd.isna(val) or str(val).strip() in ('-', ''):
        return 120
    m = re.search(r'(\d+)', str(val))
    return int(m.group(1)) if m else 120


def parse_rating(val):
    try:
        if pd.isna(val) or str(val).strip() in ('-', ''):
            return 6.5
        r = float(str(val).strip())
        return round(min(10.0, max(1.0, r)), 1)
    except Exception:
        return 6.5


def parse_votes(val):
    try:
        if pd.isna(val) or str(val).strip() in ('-', ''):
            return 50
        v = str(val).replace(',', '').strip()
        return int(float(v))
    except Exception:
        return 50


def clean_genres(g_str):
    if pd.isna(g_str) or not str(g_str).strip() or str(g_str).strip() == '-':
        return 'Drama'
    parts = [p.strip() for p in str(g_str).replace('|', ',').split(',') if p.strip() and p.strip() != '-']
    return '|'.join(parts) if parts else 'Drama'


def build_unified_dataset(
    tmdb_path=None,
    indian_path=None,
    output_path=None,
    cache_output_path=None
):
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    if tmdb_path is None:
        tmdb_path = os.path.join(data_dir, 'top10K-TMDB-movies.csv')
        if not os.path.exists(tmdb_path):
            alt_tmdb = os.path.join(os.path.dirname(base_dir), 'archive (3)', 'top10K-TMDB-movies.csv')
            if os.path.exists(alt_tmdb):
                tmdb_path = alt_tmdb

    if indian_path is None:
        indian_path = os.path.join(os.path.dirname(base_dir), 'archive (2)', 'indian movies.csv')
        if not os.path.exists(indian_path):
            alt_ind = os.path.join(data_dir, 'indian movies.csv')
            if os.path.exists(alt_ind):
                indian_path = alt_ind

    if output_path is None:
        output_path = os.path.join(data_dir, 'movies_merged.csv')

    if cache_output_path is None:
        cache_output_path = os.path.join(data_dir, 'movie_meta_cache.json')

    # Load existing metadata cache for TMDB poster_path and trailer_key
    meta_cache = {}
    if os.path.exists(cache_output_path):
        try:
            with open(cache_output_path, 'r', encoding='utf-8') as f:
                meta_cache = json.load(f)
        except Exception as e:
            print(f"Notice: Initializing fresh cache: {e}")

    print(f"Loading TMDB dataset from: {tmdb_path}")
    df_tmdb = pd.read_csv(tmdb_path)

    print(f"Loading Indian movies dataset from: {indian_path}")
    df_ind = pd.read_csv(indian_path)

    # ── 1. Process TMDB 10,000 Dataset ───────────────────────────────────────
    df_tmdb['title'] = df_tmdb['title'].fillna('Unknown Title').astype(str).str.strip()
    df_tmdb['genre'] = df_tmdb['genre'].fillna('Drama').astype(str)
    df_tmdb['genres'] = df_tmdb['genre'].apply(clean_genres)
    df_tmdb['overview'] = df_tmdb['overview'].fillna('').astype(str)
    df_tmdb['rating'] = pd.to_numeric(df_tmdb['vote_average'], errors='coerce').fillna(6.0).round(1)
    df_tmdb['vote_count'] = pd.to_numeric(df_tmdb['vote_count'], errors='coerce').fillna(100).astype(int)
    df_tmdb['popularity'] = pd.to_numeric(df_tmdb['popularity'], errors='coerce').fillna(10.0).round(2)
    df_tmdb['runtime'] = 120
    df_tmdb['language'] = df_tmdb['original_language'].map(
        lambda x: LANG_CODE_MAP.get(str(x).lower().strip(), str(x).title())
    )
    df_tmdb['id'] = df_tmdb['id'].astype(int)
    df_tmdb['year'] = df_tmdb['release_date'].apply(
        lambda d: int(str(d).split('-')[0]) if pd.notna(d) and re.match(r'^\d{4}', str(d)) else 2000
    )

    # Posters & Trailers for TMDB titles
    def resolve_tmdb_meta(row):
        mid = int(row['id'])
        mid_str = str(mid)
        cached = meta_cache.get(mid_str, {})
        poster_p = cached.get('poster_path')
        if poster_p and str(poster_p).startswith('/'):
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_p}"
        else:
            poster_url = f"https://image.tmdb.org/t/p/w500/{mid}.jpg"

        trailer_k = cached.get('trailer_key')
        if trailer_k:
            trailer_url = f"https://www.youtube.com/watch?v={trailer_k}"
        else:
            q = urllib.parse.quote_plus(f"{row['title']} {row['year']} {row['language']} official trailer")
            trailer_url = f"https://www.youtube.com/results?search_query={q}"

        return pd.Series([poster_url, trailer_url, trailer_k or '', True])

    df_tmdb[['poster', 'trailer_url', 'trailer_key', 'has_trailer']] = df_tmdb.apply(resolve_tmdb_meta, axis=1)
    df_tmdb['source'] = 'tmdb'

    # ── 2. Process Indian 50,000 Dataset ─────────────────────────────────────
    df_ind['title'] = df_ind['Movie Name'].fillna('Unknown Title').astype(str).str.strip()
    df_ind['year'] = df_ind['Year'].apply(parse_year)
    df_ind['runtime'] = df_ind['Timing(min)'].apply(parse_timing)
    df_ind['rating'] = df_ind['Rating(10)'].apply(parse_rating)
    df_ind['vote_count'] = df_ind['Votes'].apply(parse_votes)
    df_ind['genres'] = df_ind['Genre'].apply(clean_genres)
    df_ind['popularity'] = (df_ind['rating'] * 1.5 + np.log1p(df_ind['vote_count']) * 2.0).round(2)
    df_ind['language'] = df_ind['Language'].map(
        lambda x: INDIAN_LANG_MAP.get(str(x).lower().strip(), str(x).strip().title())
    )
    df_ind['overview'] = (
        df_ind['title'] + ' is a ' + df_ind['language'] + ' ' +
        df_ind['genres'].str.replace('|', ', ', regex=False) + ' movie released in ' +
        df_ind['year'].astype(str) + '.'
    )

    # Individual online search poster URL for every Indian movie (Title + Year + Language)
    df_ind['poster'] = df_ind.apply(
        lambda r: f"https://tse2.mm.bing.net/th?q={urllib.parse.quote_plus(str(r['title']) + ' ' + str(r['year']) + ' ' + str(r['language']) + ' movie poster')}&w=500&h=750&c=7&rs=1&p=0",
        axis=1
    )

    # Individual online trailer search URL for every Indian movie
    df_ind['trailer_url'] = df_ind.apply(
        lambda r: f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(str(r['title']) + ' ' + str(r['year']) + ' ' + str(r['language']) + ' official trailer')}",
        axis=1
    )
    df_ind['trailer_key'] = ''
    df_ind['has_trailer'] = True
    df_ind['source'] = 'indian'
    df_ind['id'] = range(1000000, 1000000 + len(df_ind))

    # ── 3. Initial Deduplication Against TMDB Exact Matches ────────────────
    tmdb_key_set = set(df_tmdb['title'].str.lower() + '_' + df_tmdb['year'].astype(str))
    tmdb_title_set = set(df_tmdb['title'].str.lower())

    ind_mask = ~df_ind.apply(
        lambda r: (
            (r['title'].lower() + '_' + str(r['year']) in tmdb_key_set) or
            (r['title'].lower() in tmdb_title_set and r['language'] in ('Hindi', 'English', 'Telugu', 'Tamil'))
        ),
        axis=1
    )
    df_ind_unique = df_ind[ind_mask].copy()
    df_ind_unique['id'] = range(1000000, 1000000 + len(df_ind_unique))

    cols = [
        'id', 'title', 'genres', 'year', 'rating',
        'vote_count', 'popularity', 'runtime', 'language', 'overview',
        'poster', 'trailer_url', 'trailer_key', 'has_trailer', 'source'
    ]

    df_merged = pd.concat([df_tmdb[cols], df_ind_unique[cols]], ignore_index=True)

    # ── 4. Multi-Language Canonical Grouping ────────────────────────────────
    print("Performing multi-language canonical grouping across all titles...")

    def clean_t(t):
        return re.sub(r'[^a-z0-9]', '', str(t).lower().strip())

    title_buckets = defaultdict(list)
    for idx, row in df_merged.iterrows():
        ct = clean_t(row['title'])
        title_buckets[ct].append(row)

    canonical_id_map = {}       # original_id -> canonical_id
    canonical_variants_map = {} # canonical_id -> dict of { lang: variant_dict }
    canonical_langs_map = {}    # canonical_id -> pipe-delimited string of languages

    for ct, rows in title_buckets.items():
        clusters = []
        for r in rows:
            ryear = int(r['year'])
            assigned = False
            for cluster in clusters:
                if any(abs(int(c['year']) - ryear) <= 1 for c in cluster):
                    cluster.append(r)
                    assigned = True
                    break
            if not assigned:
                clusters.append([r])

        for cluster in clusters:
            # Pick canonical primary: prefer TMDB source, then highest vote_count, then rating
            sorted_cluster = sorted(
                cluster,
                key=lambda x: (
                    1 if x['source'] == 'tmdb' else 0,
                    float(x.get('vote_count', 0)),
                    float(x.get('rating', 0))
                ),
                reverse=True
            )
            canonical_row = sorted_cluster[0]
            canonical_id = int(canonical_row['id'])

            variants_dict = {}
            avail_langs = []
            for item in sorted_cluster:
                mid = int(item['id'])
                lang = str(item['language']).strip()
                canonical_id_map[mid] = canonical_id
                if lang not in variants_dict:
                    variants_dict[lang] = {
                        'id': mid,
                        'title': str(item['title']),
                        'language': lang,
                        'year': int(item['year']),
                        'poster': str(item['poster']),
                        'trailer_url': str(item['trailer_url']),
                        'trailer_key': str(item['trailer_key']) if pd.notna(item.get('trailer_key')) and item.get('trailer_key') else None,
                        'has_trailer': bool(item.get('has_trailer', True)),
                        'overview': str(item['overview']),
                        'rating': float(item['rating']),
                        'vote_count': int(item['vote_count']),
                        'runtime': int(item.get('runtime', 120))
                    }
                    avail_langs.append(lang)

            canonical_variants_map[canonical_id] = variants_dict
            canonical_langs_map[canonical_id] = '|'.join(avail_langs)

    df_merged['canonical_id'] = df_merged['id'].map(lambda x: canonical_id_map.get(int(x), int(x)))
    df_merged['available_languages'] = df_merged['canonical_id'].map(lambda x: canonical_langs_map.get(int(x), ''))

    # ── 5. Build Unified Metadata Cache ─────────────────────────────────────
    print("Building unified metadata cache for all 60,000+ movies...")
    full_cache = dict(meta_cache)
    for _, r in df_merged.iterrows():
        mid_str = str(int(r['id']))
        cid = int(r['canonical_id'])
        full_cache[mid_str] = {
            'id': int(r['id']),
            'canonical_id': cid,
            'title': str(r['title']),
            'language': str(r['language']),
            'available_languages': str(r['available_languages']).split('|'),
            'language_variants': canonical_variants_map.get(cid, {}),
            'poster': str(r['poster']),
            'trailer_url': str(r['trailer_url']),
            'trailer_key': str(r['trailer_key']) if pd.notna(r['trailer_key']) and r['trailer_key'] else None,
            'has_trailer': bool(r['has_trailer'])
        }

    with open(cache_output_path, 'w', encoding='utf-8') as f:
        json.dump(full_cache, f)

    total_canonical = df_merged['canonical_id'].nunique()
    multi_lang_count = sum(1 for v in canonical_variants_map.values() if len(v) > 1)

    print(f"Total merged records:       {len(df_merged)}")
    print(f"  - Unique Canonical Movies: {total_canonical}")
    print(f"  - Multi-Language Movies:   {multi_lang_count}")
    print(f"  - TMDB Movies:             {len(df_tmdb)}")
    print(f"  - Indian Movies:           {len(df_ind_unique)}")
    print(f"  - Unique Languages:        {df_merged['language'].nunique()}")
    print(f"  - Metadata Cache Entries:  {len(full_cache):,}")

    final_cols = [
        'id', 'canonical_id', 'title', 'genres', 'year', 'rating',
        'vote_count', 'popularity', 'runtime', 'language', 'available_languages',
        'overview', 'poster', 'trailer_url', 'trailer_key', 'has_trailer', 'source'
    ]
    df_merged[final_cols].to_csv(output_path, index=False)
    print(f"Saved merged dataset to: {output_path}")
    return df_merged


if __name__ == '__main__':
    build_unified_dataset()
