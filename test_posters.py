import urllib.request
import urllib.parse
import pandas as pd

df = pd.read_csv("data/movies_merged.csv")
indian_samples = df[df["id"] >= 1000000].sample(10, random_state=42)

for _, row in indian_samples.iterrows():
    title = row['title']
    year = row['year']
    lang = row['language']
    q = f"{title} {year} {lang} movie poster"
    url = f"https://tse2.mm.bing.net/th?q={urllib.parse.quote(q)}&w=500&h=750&c=7&rs=1&p=0"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"[{resp.status}] {title} ({year}, {lang}) -> {url} ({len(resp.read())} bytes)")
    except Exception as e:
        print(f"[ERR] {title}: {e}")
