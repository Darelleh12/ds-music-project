"""
Per-user classification: "would user like this track?"

Label: 
    liked = 1 if user played the track >= 2 times, else 0.
    Rationale: returning to a track is a stronger "like" signal than finishing
    it once

Features per (user, track):
  release_year; int, median-imputed when MB has no date
  has_release_year: indicator for the imputation above
  genre_count: genres MB tagged the artist with (0 if unknown)
  has_genre: indicator
  genre_{g}   
  log_artist_other_ms: ln(1 + ms this user played on OTHER tracks by this artist)
                       (leave-one-out — can't see own plays or the label)

train two models per user: DecisionTree and RandomForest, 80/20 stratified
split, and report accuracy / precision / recall / F1 against a majority-class
baseline. Also prints top feature importances.

Run with:
    .venv/bin/python classification.py
    .venv/bin/python classification.py --year 0   --> all years, not just 2025
"""
import argparse
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).parent
USERS = {
    "vanessa": HERE / ".." / "vans-bs" / "vans-listening-data_2021-2026.csv",
    "darelle": HERE / ".." / "vans-bs" / "l-listening-data_2017-26.csv",
}
TRACK_CACHE = HERE / "track_metadata.csv"
ARTIST_CACHE = HERE / "artist_metadata.csv"
TOP_N_GENRES = 15
MIN_PLAYS_PER_USER = 5  # drop users with too little data 
SEED = 42


def load_plays(path, year_filter):
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["spotify_track_uri"])
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
    if year_filter:
        df = df[df["ts"].dt.year == year_filter]
    df["ms_played"] = df["ms_played"].fillna(0).astype(int)
    return df[["ts", "spotify_track_uri", "master_metadata_album_artist_name", "ms_played"]].rename(
        columns={"spotify_track_uri": "uri", "master_metadata_album_artist_name": "artist"}
    )


def load_enrichment():
    tracks = pd.read_csv(TRACK_CACHE).rename(columns={"spotify_track_uri": "uri"})
    tracks["release_year"] = pd.to_numeric(tracks["release_year"], errors="coerce")
    tracks = tracks.drop_duplicates(subset=["uri"], keep="first")[["uri", "release_year", "artist_mbid"]]

    artists = pd.read_csv(ARTIST_CACHE)
    artists["genres"] = artists["genres"].fillna("").astype(str)
    artists = artists.drop_duplicates(subset=["artist_mbid"], keep="first")[["artist_mbid", "genres"]]
    return tracks, artists


def top_genres(artists, n):
    counter = Counter()
    for g in artists["genres"]:
        for tag in g.split("|"):
            if tag:
                counter[tag] += 1
    return [g for g, _ in counter.most_common(n)]


def build_user_dataset(plays, tracks, artists, top_genre_list):
    #Per-track aggregates for this user
    agg = plays.groupby("uri").agg(
        play_count=("ms_played", "size"),
        total_ms=("ms_played", "sum"),
        artist=("artist", "first"),
    ).reset_index()

    # Leave-one-out artist play time (exclude this track's own plays)
    artist_ms = plays.groupby("artist")["ms_played"].sum().rename("artist_total_ms")
    agg = agg.merge(artist_ms, left_on="artist", right_index=True, how="left")
    agg["artist_other_ms"] = agg["artist_total_ms"] - agg["total_ms"]
    agg["log_artist_other_ms"] = np.log1p(agg["artist_other_ms"].clip(lower=0))

    #Enrichment joins
    agg = agg.merge(tracks, on="uri", how="left")
    agg = agg.merge(artists, on="artist_mbid", how="left")
    agg["genres"] = agg["genres"].fillna("")

    # Release year: median-impute + indicator
    agg["has_release_year"] = agg["release_year"].notna().astype(int)
    median_year = agg["release_year"].median()
    agg["release_year"] = agg["release_year"].fillna(median_year)

    # Genre features
    agg["has_genre"] = (agg["genres"] != "").astype(int)
    agg["genre_count"] = agg["genres"].apply(lambda s: len([g for g in s.split("|") if g]))
    for g in top_genre_list:
        col = f"genre_{g.replace(' ', '_').replace('-', '_').replace('/', '_')}"
        agg[col] = agg["genres"].apply(lambda s, g=g: int(g in s.split("|")))

    agg["liked"] = (agg["play_count"] >= 2).astype(int)
    return agg


def feature_columns(df, top_genre_list):
    genre_cols = [f"genre_{g.replace(' ', '_').replace('-', '_').replace('/', '_')}" for g in top_genre_list]
    return ["release_year", "has_release_year", "has_genre", "genre_count",
            "log_artist_other_ms"] + genre_cols


def train_eval(name, X_train, X_test, y_train, y_test, feature_names):
    results = {}
    for label, model in [
        ("DecisionTree", DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=SEED)),
        ("RandomForest", RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                                n_jobs=-1, random_state=SEED)),
    ]:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, pos_label=1,
                                                           average="binary", zero_division=0)
        imps = sorted(zip(feature_names, model.feature_importances_), key=lambda kv: kv[1], reverse=True)
        results[label] = {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "imp": imps[:8]}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025, help="filter to year (0 = all)")
    args = ap.parse_args()
    year_filter = args.year if args.year else None

    print(f"loading enrichment caches + plays (year={year_filter or 'all'})...")
    tracks, artists = load_enrichment()
    top_genre_list = top_genres(artists, TOP_N_GENRES)
    print(f"  {len(tracks)} tracks, {len(artists)} artists")
    print(f"  top-{TOP_N_GENRES} genres: {', '.join(top_genre_list)}")

    for user, path in USERS.items():
        plays = load_plays(path, year_filter)
        ds = build_user_dataset(plays, tracks, artists, top_genre_list)
        if len(ds) < MIN_PLAYS_PER_USER:
            print(f"\n{user}: too little data ({len(ds)} tracks), skipping")
            continue

        feat_cols = feature_columns(ds, top_genre_list)
        X = ds[feat_cols].values
        y = ds["liked"].values
        pos = int(y.sum())
        baseline = max(y.mean(), 1 - y.mean())

        print(f"\n=== {user} ===")
        print(f"  tracks: {len(ds)}  liked (play_count>=2): {pos} ({pos/len(ds)*100:.1f}%)")
        print(f"  majority-class baseline accuracy: {baseline:.3f}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=SEED
        )
        results = train_eval(user, X_train, X_test, y_train, y_test, feat_cols)
        for label, r in results.items():
            print(f"  {label:14s} acc={r['acc']:.3f}  precision={r['precision']:.3f}  "
                  f"recall={r['recall']:.3f}  f1={r['f1']:.3f}")
            lift = r["acc"] - baseline
            print(f"    lift vs baseline: {lift:+.3f}")
            print(f"    top features: " + ", ".join(f"{n}={v:.3f}" for n, v in r["imp"][:5]))


if __name__ == "__main__":
    main()
