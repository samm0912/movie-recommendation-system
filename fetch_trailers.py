"""
fetch_trailers.py — Metadata & Trailer Fetcher and Caching Engine
Fetches official YouTube trailers, posters, and backdrops from TMDB API for all movies in the dataset.
Caches results to data/movie_meta_cache.json with full statistics.
"""

import os
import sys
import json
import time
import socket
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Resilient DNS fallback for TMDB API
TMDB_KNOWN_IPS = ['3.175.86.103', '3.175.86.37', '3.175.86.50', '3.175.86.67']
try:
    import urllib3.util.connection as urllib_conn
    orig_create_conn = urllib_conn.create_connection

    def custom_create_conn(address, *args, **kwargs):
        host, port = address
        if host == 'api.themoviedb.org':
            host = TMDB_KNOWN_IPS[0]
        return orig_create_conn((host, port), *args, **kwargs)

    urllib_conn.create_connection = custom_create_conn
except Exception as e:
    logger.warning(f"Could not install socket DNS hook: {e}")


def get_api_key():
    key = os.environ.get("TMDB_API_KEY", "").strip()
    if not key:
        # Fallback default key
        key = "8265bd1679663a7ea12ac168da84d2e8"
    return key


def fetch_single_movie_meta(movie_id, session, api_key):
    """
    Fetch poster_path, backdrop_path, and trailer video key for a TMDB movie ID.
    """
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&append_to_response=videos"
    try:
        resp = session.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        if resp.status_code == 200:
            data = resp.json()
            poster_path = data.get("poster_path")
            backdrop_path = data.get("backdrop_path")
            
            videos = data.get("videos", {}).get("results", [])
            # Prioritize official YouTube trailer
            trailer_key = None
            trailer_name = None
            
            # 1st priority: Official Trailer on YouTube
            for v in videos:
                if v.get("site") == "YouTube" and v.get("type") == "Trailer" and v.get("official"):
                    trailer_key = v.get("key")
                    trailer_name = v.get("name")
                    break
            
            # 2nd priority: Any Trailer on YouTube
            if not trailer_key:
                for v in videos:
                    if v.get("site") == "YouTube" and v.get("type") == "Trailer":
                        trailer_key = v.get("key")
                        trailer_name = v.get("name")
                        break
            
            # 3rd priority: Teaser / Clip on YouTube
            if not trailer_key:
                for v in videos:
                    if v.get("site") == "YouTube" and v.get("type") in ("Teaser", "Clip"):
                        trailer_key = v.get("key")
                        trailer_name = v.get("name")
                        break
            
            return {
                "id": movie_id,
                "status": "success",
                "poster_path": poster_path,
                "backdrop_path": backdrop_path,
                "trailer_key": trailer_key,
                "trailer_name": trailer_name,
                "has_trailer": trailer_key is not None
            }
        elif resp.status_code == 404:
            return {"id": movie_id, "status": "not_found", "has_trailer": False}
        else:
            return {"id": movie_id, "status": f"http_{resp.status_code}", "has_trailer": False}
    except Exception as e:
        return {"id": movie_id, "status": f"error: {str(e)[:40]}", "has_trailer": False}


def build_trailer_cache(limit=None, max_workers=25):
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    csv_path = os.path.join(data_dir, 'top10K-TMDB-movies.csv')
    cache_path = os.path.join(data_dir, 'movie_meta_cache.json')
    
    if not os.path.exists(csv_path):
        alt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'archive (3)', 'top10K-TMDB-movies.csv')
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            print(f"Error: Dataset not found at {csv_path}")
            return
    
    df = pd.read_csv(csv_path)
    total_movies = len(df)
    if limit:
        df = df.head(limit)
    
    movie_ids = df['id'].tolist()
    api_key = get_api_key()
    
    # Load existing cache if any
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            logger.info(f"Loaded existing cache with {len(cache)} entries.")
        except Exception:
            cache = {}
    
    to_fetch = [mid for mid in movie_ids if str(mid) not in cache]
    logger.info(f"Total movies in dataset: {total_movies}")
    logger.info(f"Already in cache: {len(cache)}")
    logger.info(f"To fetch: {len(to_fetch)}")
    
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers, max_retries=2)
    session.mount('https://', adapter)
    session.mount('http://', adapter)

    trailers_found = sum(1 for v in cache.values() if v.get("has_trailer"))
    trailers_unavailable = sum(1 for v in cache.values() if not v.get("has_trailer") and v.get("status") == "success")
    failed_matches = sum(1 for v in cache.values() if v.get("status") in ("not_found", "error"))

    if to_fetch:
        logger.info(f"Fetching metadata with {max_workers} worker threads...")
        processed = 0
        save_interval = 250
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_single_movie_meta, mid, session, api_key): mid for mid in to_fetch}
            
            for future in as_completed(futures):
                mid = futures[future]
                res = future.result()
                cache[str(mid)] = res
                
                if res.get("has_trailer"):
                    trailers_found += 1
                elif res.get("status") == "success":
                    trailers_unavailable += 1
                else:
                    failed_matches += 1
                
                processed += 1
                if processed % save_interval == 0 or processed == len(to_fetch):
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(cache, f)
                    logger.info(f"Progress: {processed}/{len(to_fetch)} processed ({len(cache)} total cached).")
    
    # Final save
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)
    
    print("\n" + "="*50)
    print("TRAILER & METADATA FETCH SUMMARY")
    print("="*50)
    print(f"Total movies:         {total_movies}")
    print(f"Trailers found:       {trailers_found}")
    print(f"Trailers unavailable: {trailers_unavailable}")
    print(f"Failed matches:       {failed_matches}")
    print(f"Cache file saved to:  {cache_path}")
    print("="*50 + "\n")


if __name__ == '__main__':
    limit = None
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        limit = int(sys.argv[1])
    build_trailer_cache(limit=limit, max_workers=25)
