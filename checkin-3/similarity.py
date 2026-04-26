"""
Similarity between two users across 5 categories.

Reads listening data from both users, joins with enrichment caches for genre
and release year, computes 5 sub-scores + composite:

  hour       cosine on 24-bin hour-of-day distribution (weighted by ms_played)
  month      cosine on 12-bin month-of-year distribution
  artist     weighted Jaccard on per-artist ms_played
  genre      cosine on genre distribution (unknown excluded)
  year       cosine on release-year 5yr-bucket distribution (unknown excluded)
  composite  mean of the five

Runs a validation: same-user split-halves similarity vs cross-user similarity.
Same-user should be noticeably higher if the measure is meaningful.

Usage:
  python similarity.py            # year 2025 only (matches check-in 2 scope)
  python similarity.py --year 0   # all years
"""
import argparse
import csv
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
USERS = {
    "vanessa": HERE / ".." / "vans-bs" / "vans-listening-data_2021-2026.csv",
    "darelle": HERE / ".." / "vans-bs" / "l-listening-data_2017-26.csv",
}
TRACK_CACHE = HERE / "track_metadata.csv"
ARTIST_CACHE = HERE / "artist_metadata.csv"
YEAR_BUCKET = 5  # 5-year buckets for release-year similarity


def load_plays(path, year_filter=None):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            ts = r.get("ts") or ""
            if not ts:
                continue
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if year_filter and dt.year != year_filter:
                continue
            rows.append({
                "ts": dt,
                "uri": r.get("spotify_track_uri") or "",
                "artist": r.get("master_metadata_album_artist_name") or "",
                "ms_played": int(r.get("ms_played") or 0),
            })
    return rows


def load_track_meta(path):
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            year = r.get("release_year") or ""
            out[r["spotify_track_uri"]] = {
                "release_year": int(year) if year.isdigit() else None,
                "artist_mbid": r.get("artist_mbid", ""),
            }
    return out


def load_artist_meta(path):
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            genres = [g for g in (r.get("genres") or "").split("|") if g]
            out[r["artist_mbid"]] = genres
    return out


def hour_dist(rows):
    v = [0.0] * 24
    for r in rows:
        v[r["ts"].hour] += r["ms_played"]
    return v


def month_dist(rows):
    v = [0.0] * 12
    for r in rows:
        v[r["ts"].month - 1] += r["ms_played"]
    return v


def artist_dist(rows):
    d = defaultdict(float)
    for r in rows:
        if r["artist"]:
            d[r["artist"]] += r["ms_played"]
    return dict(d)


def genre_dist(rows, track_meta, artist_meta):
    d = defaultdict(float)
    for r in rows:
        meta = track_meta.get(r["uri"])
        if not meta:
            continue
        genres = artist_meta.get(meta.get("artist_mbid", ""), [])
        if not genres:
            continue
        share = r["ms_played"] / len(genres)
        for g in genres:
            d[g] += share
    return dict(d)


def year_dist(rows, track_meta):
    d = defaultdict(float)
    for r in rows:
        meta = track_meta.get(r["uri"])
        if not meta or meta.get("release_year") is None:
            continue
        bucket = (meta["release_year"] // YEAR_BUCKET) * YEAR_BUCKET
        d[bucket] += r["ms_played"]
    return dict(d)


def cosine(a, b):
    if isinstance(a, list):
        a = {i: v for i, v in enumerate(a)}
    if isinstance(b, list):
        b = {i: v for i, v in enumerate(b)}
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def weighted_jaccard(a, b):
    keys = set(a) | set(b)
    mn = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    mx = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return mn / mx if mx else 0.0


def features(rows, track_meta, artist_meta):
    return {
        "hour": hour_dist(rows),
        "month": month_dist(rows),
        "artist": artist_dist(rows),
        "genre": genre_dist(rows, track_meta, artist_meta),
        "year": year_dist(rows, track_meta),
    }


def similarity(fa, fb):
    out = {
        "hour":   cosine(fa["hour"], fb["hour"]),
        "month":  cosine(fa["month"], fb["month"]),
        "artist": weighted_jaccard(fa["artist"], fb["artist"]),
        "genre":  cosine(fa["genre"], fb["genre"]),
        "year":   cosine(fa["year"], fb["year"]),
    }
    out["composite"] = sum(out.values()) / len(out)
    return out


def split_halves(rows, seed=42):
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    mid = len(shuffled) // 2
    return shuffled[:mid], shuffled[mid:]


def fmt_scores(s):
    order = ["hour", "month", "artist", "genre", "year", "composite"]
    return "  ".join(f"{k}={s[k]:.3f}" for k in order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025, help="filter to year (0 = all)")
    args = ap.parse_args()
    year_filter = args.year if args.year else None

    print("loading enrichment caches...")
    track_meta = load_track_meta(TRACK_CACHE)
    artist_meta = load_artist_meta(ARTIST_CACHE)
    print(f"  {len(track_meta)} tracks, {len(artist_meta)} artists enriched")

    print(f"\nloading user plays (year={year_filter or 'all'})...")
    user_rows = {name: load_plays(path, year_filter) for name, path in USERS.items()}
    for name, rows in user_rows.items():
        print(f"  {name}: {len(rows)} plays")

    feats = {name: features(rows, track_meta, artist_meta) for name, rows in user_rows.items()}
    names = list(user_rows.keys())

    print("\n=== user-vs-user similarity ===")
    s = similarity(feats[names[0]], feats[names[1]])
    print(f"  {names[0]} vs {names[1]}: {fmt_scores(s)}")

    print("\n=== validation: same-user split-halves vs cross-user ===")
    halves = {}
    for name, rows in user_rows.items():
        a, b = split_halves(rows)
        halves[f"{name}_a"] = features(a, track_meta, artist_meta)
        halves[f"{name}_b"] = features(b, track_meta, artist_meta)

    pairs = [
        ("same",  f"{names[0]}_a", f"{names[0]}_b"),
        ("same",  f"{names[1]}_a", f"{names[1]}_b"),
        ("cross", f"{names[0]}_a", f"{names[1]}_a"),
        ("cross", f"{names[0]}_b", f"{names[1]}_b"),
    ]
    for label, la, lb in pairs:
        s = similarity(halves[la], halves[lb])
        print(f"  [{label:5s}] {la} vs {lb}: {fmt_scores(s)}")


if __name__ == "__main__":
    main()
