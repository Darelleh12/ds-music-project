"""
Presentation visuals for check-in 3 tasks 1 (similarity) and 2 (classification).

Writes PNGs to checkin-3/viz/:
  01_similarity_radar.png         5 sub-scores — vanessa vs darelle
  02_similarity_validation.png    same-user split-halves vs cross-user (validation)
  03_hour_of_day.png              listening by hour, both users overlaid
  04_top_genres.png               top genres by ms_played, both users
  05_release_years.png            release-year distribution, both users (5yr buckets)
  06_classification_accuracy.png  baseline / DT / RF per user
  07_feature_importance.png       top-10 RF feature importances per user

Run:
    .venv/bin/python visuals.py
    .venv/bin/python visuals.py --year 0   # all years, not just 2025
"""
import argparse
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).parent
VIZ_DIR = HERE / "viz"
USERS = {
    "Vanessa": HERE / ".." / "vans-bs" / "vans-listening-data_2021-2026.csv",
    "Darelle": HERE / ".." / "vans-bs" / "l-listening-data_2017-26.csv",
}
TRACK_CACHE = HERE / "track_metadata.csv"
ARTIST_CACHE = HERE / "artist_metadata.csv"
TOP_N_GENRES = 10
YEAR_BUCKET = 5
SEED = 42

COLORS = {"Vanessa": "#e74c3c", "Darelle": "#3498db"}
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
                     "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight"})


# ---------- data loading ----------

def load_plays_raw(path, year_filter):
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["spotify_track_uri"])
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
    if year_filter:
        df = df[df["ts"].dt.year == year_filter]
    df["ms_played"] = df["ms_played"].fillna(0).astype(int)
    df["hour"] = df["ts"].dt.hour
    df["month"] = df["ts"].dt.month
    return df.rename(columns={"spotify_track_uri": "uri",
                              "master_metadata_album_artist_name": "artist"})[
        ["ts", "uri", "artist", "hour", "month", "ms_played"]
    ]


def load_enrichment():
    tracks = pd.read_csv(TRACK_CACHE).rename(columns={"spotify_track_uri": "uri"})
    tracks["release_year"] = pd.to_numeric(tracks["release_year"], errors="coerce")
    tracks = tracks.drop_duplicates(subset=["uri"], keep="first")[["uri", "release_year", "artist_mbid"]]
    artists = pd.read_csv(ARTIST_CACHE)
    artists["genres"] = artists["genres"].fillna("").astype(str)
    artists = artists.drop_duplicates(subset=["artist_mbid"], keep="first")[["artist_mbid", "genres"]]
    return tracks, artists


# ---------- similarity (recomputed here to drive plots) ----------

def hour_vec(df):
    v = np.zeros(24)
    for h, ms in zip(df["hour"], df["ms_played"]):
        v[h] += ms
    return v


def month_vec(df):
    v = np.zeros(12)
    for m, ms in zip(df["month"], df["ms_played"]):
        v[m - 1] += ms
    return v


def artist_dist(df):
    return df.groupby("artist")["ms_played"].sum().to_dict()


def genre_dist(df, track_meta_map, artist_genres_map):
    d = defaultdict(float)
    for uri, ms in zip(df["uri"], df["ms_played"]):
        amid = track_meta_map.get(uri, {}).get("artist_mbid", "")
        genres = artist_genres_map.get(amid, [])
        if not genres:
            continue
        share = ms / len(genres)
        for g in genres:
            d[g] += share
    return dict(d)


def year_dist(df, track_meta_map):
    d = defaultdict(float)
    for uri, ms in zip(df["uri"], df["ms_played"]):
        yr = track_meta_map.get(uri, {}).get("release_year")
        if yr is None or pd.isna(yr):
            continue
        bucket = (int(yr) // YEAR_BUCKET) * YEAR_BUCKET
        d[bucket] += ms
    return dict(d)


def cosine(a, b):
    if isinstance(a, np.ndarray):
        a, b = {i: float(v) for i, v in enumerate(a)}, {i: float(v) for i, v in enumerate(b)}
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


def feature_set(df, track_meta_map, artist_genres_map):
    return {
        "hour":  hour_vec(df),
        "month": month_vec(df),
        "artist": artist_dist(df),
        "genre":  genre_dist(df, track_meta_map, artist_genres_map),
        "year":   year_dist(df, track_meta_map),
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


def split_halves(df, seed=SEED):
    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    mid = len(idx) // 2
    return df.iloc[idx[:mid]].reset_index(drop=True), df.iloc[idx[mid:]].reset_index(drop=True)


# ---------- plots: similarity ----------

def plot_similarity_radar(sim):
    cats = ["hour", "month", "artist", "genre", "year"]
    vals = [sim[c] for c in cats]
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    vals += vals[:1]; angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw={"projection": "polar"})
    ax.plot(angles, vals, color="#8e44ad", linewidth=2)
    ax.fill(angles, vals, color="#8e44ad", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.capitalize() for c in cats])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title(f"Vanessa vs. Darelle — Similarity by Category\n(composite = {sim['composite']:.2f})",
                 pad=20)
    plt.savefig(VIZ_DIR / "01_similarity_radar.png")
    plt.close()


def plot_validation(same_a, same_b, cross_a, cross_b):
    cats = ["hour", "month", "artist", "genre", "year", "composite"]
    same_avg = [(same_a[c] + same_b[c]) / 2 for c in cats]
    cross_avg = [(cross_a[c] + cross_b[c]) / 2 for c in cats]

    x = np.arange(len(cats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, same_avg, w, label="Same user (split halves)", color="#27ae60")
    ax.bar(x + w / 2, cross_avg, w, label="Cross user (Vanessa vs Darelle)", color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in cats])
    ax.set_ylabel("Similarity score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Validation: Same-User vs. Cross-User Similarity\n"
                 "(same-user should score higher if the measure is meaningful)")
    ax.legend()
    for i, (s, c) in enumerate(zip(same_avg, cross_avg)):
        ax.text(i - w / 2, s + 0.01, f"{s:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, c + 0.01, f"{c:.2f}", ha="center", fontsize=9)
    plt.savefig(VIZ_DIR / "02_similarity_validation.png")
    plt.close()


def plot_hour_of_day(user_data):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, df in user_data.items():
        v = hour_vec(df)
        v = v / v.sum() * 100 if v.sum() else v
        ax.plot(range(24), v, marker="o", label=name, color=COLORS[name], linewidth=2)
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("% of total listening time")
    ax.set_title("Listening Distribution by Hour of Day — 2025")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.savefig(VIZ_DIR / "03_hour_of_day.png")
    plt.close()


def plot_top_genres(user_data, track_meta_map, artist_genres_map):
    user_top = {}
    for name, df in user_data.items():
        d = genre_dist(df, track_meta_map, artist_genres_map)
        total = sum(d.values()) or 1
        user_top[name] = sorted(((g, v / total * 100) for g, v in d.items()),
                                key=lambda kv: kv[1], reverse=True)[:TOP_N_GENRES]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for ax, (name, items) in zip(axes, user_top.items()):
        gs, ps = zip(*items)
        ax.barh(range(len(gs)), ps, color=COLORS[name])
        ax.set_yticks(range(len(gs)))
        ax.set_yticklabels(gs)
        ax.invert_yaxis()
        ax.set_xlabel("% of matched listening time")
        ax.set_title(f"{name} — Top {TOP_N_GENRES} Genres")
        for i, p in enumerate(ps):
            ax.text(p + 0.3, i, f"{p:.1f}%", va="center", fontsize=9)
    fig.suptitle("Top Genres by Listening Time (artist-tagged, equal-split across multi-genre artists)")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "04_top_genres.png")
    plt.close()


def plot_release_years(user_data, track_meta_map):
    fig, ax = plt.subplots(figsize=(10, 5))
    all_buckets = set()
    user_dist = {}
    for name, df in user_data.items():
        d = year_dist(df, track_meta_map)
        total = sum(d.values()) or 1
        user_dist[name] = {k: v / total * 100 for k, v in d.items()}
        all_buckets |= set(d.keys())
    buckets = sorted(all_buckets)
    x = np.arange(len(buckets))
    w = 0.38
    for i, (name, d) in enumerate(user_dist.items()):
        vals = [d.get(b, 0) for b in buckets]
        ax.bar(x + (i - 0.5) * w, vals, w, label=name, color=COLORS[name])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}" for b in buckets], rotation=45)
    ax.set_xlabel("Release year (5-year buckets)")
    ax.set_ylabel("% of listening time with known release year")
    ax.set_title("Release-Year Distribution by User — 2025\n"
                 "(tracks without MusicBrainz release data excluded)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "05_release_years.png")
    plt.close()


# ---------- classification (re-run for plots) ----------

def build_classification_dataset(plays, tracks, artists, top_genre_list):
    agg = plays.groupby("uri").agg(
        play_count=("ms_played", "size"),
        total_ms=("ms_played", "sum"),
        artist=("artist", "first"),
    ).reset_index()
    artist_ms = plays.groupby("artist")["ms_played"].sum().rename("artist_total_ms")
    agg = agg.merge(artist_ms, left_on="artist", right_index=True, how="left")
    agg["artist_other_ms"] = agg["artist_total_ms"] - agg["total_ms"]
    agg["log_artist_other_ms"] = np.log1p(agg["artist_other_ms"].clip(lower=0))
    agg = agg.merge(tracks, on="uri", how="left").merge(artists, on="artist_mbid", how="left")
    agg["genres"] = agg["genres"].fillna("")
    agg["has_release_year"] = agg["release_year"].notna().astype(int)
    median_year = agg["release_year"].median()
    agg["release_year"] = agg["release_year"].fillna(median_year)
    agg["has_genre"] = (agg["genres"] != "").astype(int)
    agg["genre_count"] = agg["genres"].apply(lambda s: len([g for g in s.split("|") if g]))
    for g in top_genre_list:
        col = f"genre_{g.replace(' ', '_').replace('-', '_').replace('/', '_')}"
        agg[col] = agg["genres"].apply(lambda s, g=g: int(g in s.split("|")))
    agg["liked"] = (agg["play_count"] >= 2).astype(int)
    return agg


def run_classification(user_data, tracks, artists):
    counter = Counter()
    for g in artists["genres"]:
        for tag in g.split("|"):
            if tag:
                counter[tag] += 1
    top_genre_list = [g for g, _ in counter.most_common(15)]
    genre_cols = [f"genre_{g.replace(' ', '_').replace('-', '_').replace('/', '_')}" for g in top_genre_list]
    feat_cols = (["release_year", "has_release_year", "has_genre", "genre_count",
                  "log_artist_other_ms"] + genre_cols)
    results = {}
    for name, plays in user_data.items():
        ds = build_classification_dataset(plays, tracks, artists, top_genre_list)
        X = ds[feat_cols].values
        y = ds["liked"].values
        baseline = max(y.mean(), 1 - y.mean())
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=SEED
        )
        dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=SEED).fit(X_train, y_train)
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                    n_jobs=-1, random_state=SEED).fit(X_train, y_train)
        results[name] = {
            "baseline": baseline,
            "dt_acc": accuracy_score(y_test, dt.predict(X_test)),
            "rf_acc": accuracy_score(y_test, rf.predict(X_test)),
            "rf_importance": sorted(zip(feat_cols, rf.feature_importances_),
                                    key=lambda kv: kv[1], reverse=True)[:10],
            "n_tracks": len(ds),
            "liked_pct": y.mean(),
        }
    return results


def plot_classification_accuracy(results):
    names = list(results.keys())
    x = np.arange(len(names))
    w = 0.27
    baselines = [results[n]["baseline"] for n in names]
    dt = [results[n]["dt_acc"] for n in names]
    rf = [results[n]["rf_acc"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, baselines, w, label="Majority-class baseline", color="#95a5a6")
    ax.bar(x,     dt,        w, label="Decision Tree",          color="#2ecc71")
    ax.bar(x + w, rf,        w, label="Random Forest",          color="#8e44ad")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Test-set accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Classification Accuracy — 'Would user play this track again?'")
    ax.legend()
    for i, n in enumerate(names):
        for offset, val in [(-w, baselines[i]), (0, dt[i]), (w, rf[i])]:
            ax.text(i + offset, val + 0.01, f"{val:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "06_classification_accuracy.png")
    plt.close()


def plot_feature_importance(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (name, r) in zip(axes, results.items()):
        feats, imps = zip(*r["rf_importance"])
        ax.barh(range(len(feats)), imps, color=COLORS[name])
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Feature importance (Random Forest)")
        ax.set_title(f"{name} — Top 10 Features")
    fig.suptitle("What drives 'liked' prediction?  (Random Forest importances)")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "07_feature_importance.png")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    args = ap.parse_args()
    year_filter = args.year if args.year else None

    VIZ_DIR.mkdir(exist_ok=True)
    print(f"loading data (year={year_filter or 'all'})...")
    tracks, artists = load_enrichment()
    track_meta_map = tracks.set_index("uri")[["release_year", "artist_mbid"]].to_dict("index")
    artist_genres_map = {
        r.artist_mbid: [g for g in r.genres.split("|") if g]
        for r in artists.itertuples()
    }
    user_data = {name: load_plays_raw(path, year_filter) for name, path in USERS.items()}
    for name, df in user_data.items():
        print(f"  {name}: {len(df):,} plays, {df['ms_played'].sum() / 3_600_000:.0f}h")

    # ---- task 1: similarity ----
    print("\ntask 1: similarity...")
    feats = {n: feature_set(df, track_meta_map, artist_genres_map) for n, df in user_data.items()}
    names = list(user_data.keys())
    sim_cross = similarity(feats[names[0]], feats[names[1]])
    print(f"  cross-user composite: {sim_cross['composite']:.3f}")

    halves = {}
    for name, df in user_data.items():
        a, b = split_halves(df)
        halves[f"{name}_a"] = feature_set(a, track_meta_map, artist_genres_map)
        halves[f"{name}_b"] = feature_set(b, track_meta_map, artist_genres_map)
    same_a = similarity(halves[f"{names[0]}_a"], halves[f"{names[0]}_b"])
    same_b = similarity(halves[f"{names[1]}_a"], halves[f"{names[1]}_b"])
    cross_a = similarity(halves[f"{names[0]}_a"], halves[f"{names[1]}_a"])
    cross_b = similarity(halves[f"{names[0]}_b"], halves[f"{names[1]}_b"])

    plot_similarity_radar(sim_cross)
    plot_validation(same_a, same_b, cross_a, cross_b)
    plot_hour_of_day(user_data)
    plot_top_genres(user_data, track_meta_map, artist_genres_map)
    plot_release_years(user_data, track_meta_map)

    # ---- task 2: classification ----
    print("\ntask 2: classification...")
    cls = run_classification(user_data, tracks, artists)
    for name, r in cls.items():
        print(f"  {name}: baseline={r['baseline']:.3f} DT={r['dt_acc']:.3f} RF={r['rf_acc']:.3f}")
    plot_classification_accuracy(cls)
    plot_feature_importance(cls)

    print(f"\nwrote {len(list(VIZ_DIR.glob('*.png')))} PNGs to {VIZ_DIR.relative_to(HERE.parent)}/")


if __name__ == "__main__":
    main()
