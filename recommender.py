"""
Personalized Content-based music recommender
Two modes:
  cross-user:  Tracks from the OTHER user's library that the target user
              has never played, ranked by how well they match the target's
              taste profile; what you'd like from your friend's
              library

  discover:   Within a user's own library, tracks played only once,
              ranked by closeness to the user's overall taste profile;
              songs they may have skipped past.

Artist is excluded so recommendations can cross artist lines

Run from the repo root:
    .venv/bin/python recommender.py
    .venv/bin/python recommender.py --year 0      -> all years, not just 2025
    .venv/bin/python recommender.py --top 20
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CHECKIN3 = HERE / "checkin-3"
USERS = {
    "vanessa": HERE / "vans-bs" / "vans-listening-data_2021-2026.csv",
    "darelle": HERE / "vans-bs" / "l-listening-data_2017-26.csv",
}
TRACK_CACHE = CHECKIN3 / "track_metadata.csv"
ARTIST_CACHE = CHECKIN3 / "artist_metadata.csv"
TOP_N_GENRES = 20
YEAR_BUCKET = 5
DISCOVERY_MAX_PLAYS = 1   #tracks played this many times or fewer = "skipped past"
MAX_PER_ARTIST = 2        # cap so one artist doesn't fill the top-N
MIN_TARGET_VECTOR_NORM = 1e-9  # guard against empty profiles


def load_plays(path, year_filter):
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["spotify_track_uri"])
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
    if year_filter:
        df = df[df["ts"].dt.year == year_filter]
    df["ms_played"] = df["ms_played"].fillna(0).astype(int)
    return df.rename(columns={
        "spotify_track_uri": "uri",
        "master_metadata_album_artist_name": "artist",
        "master_metadata_track_name": "track",
    })[["ts", "uri", "artist", "track", "ms_played"]]


def load_enrichment():
    tracks = pd.read_csv(TRACK_CACHE).rename(columns={"spotify_track_uri": "uri"})
    tracks["release_year"] = pd.to_numeric(tracks["release_year"], errors="coerce")
    tracks = tracks.drop_duplicates(subset=["uri"], keep="first")[
        ["uri", "release_year", "artist_mbid"]
    ]

    artists = pd.read_csv(ARTIST_CACHE)
    artists["genres"] = artists["genres"].fillna("").astype(str)
    artists = artists.drop_duplicates(subset=["artist_mbid"], keep="first")[
        ["artist_mbid", "genres"]
    ]
    return tracks, artists


def top_genres(artists, n):
    counter = Counter()
    for g in artists["genres"]:
        for tag in g.split("|"):
            if tag:
                counter[tag] += 1
    return [g for g, _ in counter.most_common(n)]


def build_track_features(tracks, artists, top_genre_list, year_buckets):
    """One row per URI; columns = top-N genres (one-hot, normalized by genre
    count so multi-genre tracks don't dominate) + one-hot year bucket."""
    df = tracks.merge(artists, on="artist_mbid", how="left")
    df["genres"] = df["genres"].fillna("")

    genre_cols = [f"g_{g}" for g in top_genre_list]
    for g, col in zip(top_genre_list, genre_cols):
        df[col] = df["genres"].apply(lambda s, g=g: int(g in s.split("|")))

    #Normalize each row's genre vector so a 5-genre track and a 1-genre track
    #contribute equally per "play" before the ms_played weighting.
    genre_sum = df[genre_cols].sum(axis=1).replace(0, np.nan)
    df[genre_cols] = df[genre_cols].div(genre_sum, axis=0).fillna(0)

    #Release-year bucket one-hot
    year_cols = [f"y_{b}" for b in year_buckets]
    for b, col in zip(year_buckets, year_cols):
        df[col] = 0
    df.loc[df["release_year"].notna(), "_bucket"] = (
        (df.loc[df["release_year"].notna(), "release_year"] // YEAR_BUCKET) * YEAR_BUCKET
    )
    for b, col in zip(year_buckets, year_cols):
        df.loc[df["_bucket"] == b, col] = 1
    df = df.drop(columns=["_bucket"], errors="ignore")

    feat_cols = genre_cols + year_cols
    return df[["uri", "artist_mbid", "genres", "release_year"] + feat_cols], feat_cols


def taste_profile(plays, track_features, feat_cols):
    """ms_played-weighted average of per-track vectors, L2-normalized."""
    weights = plays.groupby("uri")["ms_played"].sum().rename("w").reset_index()
    joined = weights.merge(track_features[["uri"] + feat_cols], on="uri", how="inner")
    if joined.empty:
        return np.zeros(len(feat_cols))
    vecs = joined[feat_cols].values
    w = joined["w"].values.reshape(-1, 1)
    profile = (vecs * w).sum(axis=0) / max(w.sum(), 1)
    norm = np.linalg.norm(profile)
    return profile / norm if norm > MIN_TARGET_VECTOR_NORM else profile


def cosine_scores(profile, vecs):
    norms = np.linalg.norm(vecs, axis=1)
    norms[norms < MIN_TARGET_VECTOR_NORM] = np.nan
    sims = (vecs @ profile) / norms
    return np.nan_to_num(sims, nan=0.0)


def cross_user_recs(source_plays, target_plays, target_profile, track_features,
                    feat_cols, top_n):
    """Tracks the source user played that the target user has NOT played,
    ranked by cosine(target_profile, track_vector)."""
    target_uris = set(target_plays["uri"].unique())
    source_agg = source_plays.groupby("uri").agg(
        source_plays=("ms_played", "size"),
        source_ms=("ms_played", "sum"),
        artist=("artist", "first"),
        track=("track", "first"),
    ).reset_index()
    candidates = source_agg[~source_agg["uri"].isin(target_uris)]
    candidates = candidates.merge(track_features, on="uri", how="left")
    candidates = candidates.dropna(subset=feat_cols, how="all")
    if candidates.empty:
        return candidates

    vecs = candidates[feat_cols].fillna(0).values
    candidates = candidates.assign(score=cosine_scores(target_profile, vecs))
    #Tie-break: stronger source signal wins among equally-matched tracks.
    candidates = candidates.sort_values(
        ["score", "source_ms"], ascending=[False, False]
    )
    candidates = cap_per_artist(candidates, MAX_PER_ARTIST)
    return candidates.head(top_n)


def discovery_recs(plays, profile, track_features, feat_cols, top_n,
                   max_plays=DISCOVERY_MAX_PLAYS):
    """Within-user: tracks played <=max_plays times, ranked by taste-match."""
    agg = plays.groupby("uri").agg(
        play_count=("ms_played", "size"),
        total_ms=("ms_played", "sum"),
        artist=("artist", "first"),
        track=("track", "first"),
    ).reset_index()
    candidates = agg[agg["play_count"] <= max_plays]
    candidates = candidates.merge(track_features, on="uri", how="left")
    candidates = candidates.dropna(subset=feat_cols, how="all")
    if candidates.empty:
        return candidates

    vecs = candidates[feat_cols].fillna(0).values
    candidates = candidates.assign(score=cosine_scores(profile, vecs))
    candidates = candidates.sort_values("score", ascending=False)
    candidates = cap_per_artist(candidates, MAX_PER_ARTIST)
    return candidates.head(top_n)


def cap_per_artist(df, max_per_artist):
    """Keep at most max_per_artist rows per artist (preserves order)."""
    if df.empty or max_per_artist <= 0:
        return df
    return df.groupby("artist", sort=False).head(max_per_artist).reset_index(drop=True)


def fmt_genre_tags(s, max_n=3):
    tags = [t for t in (s or "").split("|") if t][:max_n]
    return ", ".join(tags) if tags else "—"


def print_recs(title, df, source_col=None):
    print(f"\n=== {title} ===")
    if df.empty:
        print("  (no candidates)")
        return
    for i, (_, r) in enumerate(df.iterrows(), 1):
        track = (r.get("track") or "").strip() or "(untitled)"
        artist = (r.get("artist") or "").strip() or "(unknown)"
        year = int(r["release_year"]) if pd.notna(r["release_year"]) else None
        year_str = str(year) if year else "----"
        genres = fmt_genre_tags(r.get("genres", ""))
        line = f"  {i:2d}. {track[:45]:45s}  {artist[:25]:25s}  [{year_str}]  {genres:30s}  score={r['score']:.3f}"
        if source_col and source_col in r:
            line += f"  {source_col}={int(r[source_col])}"
        print(line)


def write_csv(df, path, cols):
    keep = [c for c in cols if c in df.columns]
    df[keep].to_csv(path, index=False)
    print(f"  wrote {path.name} ({len(df)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025, help="filter to year (0 = all)")
    ap.add_argument("--top", type=int, default=10, help="top-N to print per section")
    ap.add_argument("--csv-top", type=int, default=50, help="rows to write to CSV per section")
    args = ap.parse_args()
    year_filter = args.year if args.year else None

    print(f"loading enrichment + plays (year={year_filter or 'all'})...")
    tracks, artists = load_enrichment()
    top_genre_list = top_genres(artists, TOP_N_GENRES)

    # All 5-year buckets present in the catalog.
    years = tracks["release_year"].dropna().astype(int)
    if len(years):
        bmin = int((years.min() // YEAR_BUCKET) * YEAR_BUCKET)
        bmax = int((years.max() // YEAR_BUCKET) * YEAR_BUCKET)
        year_buckets = list(range(bmin, bmax + 1, YEAR_BUCKET))
    else:
        year_buckets = []

    track_features, feat_cols = build_track_features(
        tracks, artists, top_genre_list, year_buckets
    )
    print(f"  {len(track_features)} tracks, {len(feat_cols)} features "
          f"({len(top_genre_list)} genres + {len(year_buckets)} year buckets)")

    user_plays = {u: load_plays(p, year_filter) for u, p in USERS.items()}
    profiles = {u: taste_profile(p, track_features, feat_cols) for u, p in user_plays.items()}
    for u, p in user_plays.items():
        print(f"  {u}: {len(p)} plays, profile norm={np.linalg.norm(profiles[u]):.3f}")

    names = list(USERS.keys())
    a, b = names[0], names[1]

    #Cross-user: each direction
    cross_a_to_b = cross_user_recs(
        user_plays[a], user_plays[b], profiles[b],
        track_features, feat_cols, args.csv_top,
    )
    cross_b_to_a = cross_user_recs(
        user_plays[b], user_plays[a], profiles[a],
        track_features, feat_cols, args.csv_top,
    )
    print_recs(f"cross-user: from {a}'s library → recommend to {b}",
               cross_a_to_b.head(args.top), source_col="source_plays")
    print_recs(f"cross-user: from {b}'s library → recommend to {a}",
               cross_b_to_a.head(args.top), source_col="source_plays")

    #Discovery: each user's own library
    disc = {}
    for u in names:
        disc[u] = discovery_recs(
            user_plays[u], profiles[u], track_features, feat_cols, args.csv_top,
        )
        print_recs(f"discovery: {u}'s library, played ≤{DISCOVERY_MAX_PLAYS}× "
                   f"and matching their taste",
                   disc[u].head(args.top), source_col="play_count")

    # Persist
    print("\nwriting CSVs...")
    out_cols = ["uri", "track", "artist", "genres", "release_year", "score"]
    write_csv(cross_a_to_b.assign(target=b),
              CHECKIN3 / f"recs_cross_{a}_to_{b}.csv",
              out_cols + ["source_plays", "source_ms", "target"])
    write_csv(cross_b_to_a.assign(target=a),
              CHECKIN3 / f"recs_cross_{b}_to_{a}.csv",
              out_cols + ["source_plays", "source_ms", "target"])
    for u in names:
        write_csv(disc[u], CHECKIN3 / f"recs_discovery_{u}.csv",
                  out_cols + ["play_count", "total_ms"])


if __name__ == "__main__":
    main()
