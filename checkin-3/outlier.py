"""
Outlier detection: anomalous tracks per user.

For each user, aggregates plays into per-track feature rows
(play_count, total_ms, avg_ms, skip_rate, shuffle_rate, hour_mean,
hour_std, release_year), then runs IsolationForest to flag anomalies.
A complementary genre-rarity pass surfaces tracks whose primary genre
accounts for <1% of the user's listening time.

Usage:
  python outlier.py            # year 2025 only (matches check-in 2 scope)
  python outlier.py --year 0   # all years
"""
import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
USERS = {
    "vanessa": HERE / ".." / "vans-bs" / "vans-listening-data_2021-2026.csv",
    "darelle": HERE / ".." / "vans-bs" / "l-listening-data_2017-26.csv",
}
TRACK_CACHE = HERE / "track_metadata.csv"
ARTIST_CACHE = HERE / "artist_metadata.csv"

MIN_PLAYS = 2          # only score tracks the user actually engaged with
TOP_N = 10
CONTAMINATION = 0.05   # ~5% of tracks flagged
NUMERIC_FEATURES = [
    "play_count", "total_ms", "avg_ms", "skip_rate",
    "shuffle_rate", "hour_mean", "hour_std", "release_year",
]


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
                "track": r.get("master_metadata_track_name") or "",
                "artist": r.get("master_metadata_album_artist_name") or "",
                "ms_played": int(r.get("ms_played") or 0),
                "skipped": (r.get("skipped") or "").lower() == "true",
                "shuffle": (r.get("shuffle") or "").lower() == "true",
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


def per_track_features(rows, track_meta, artist_meta):
    bucket = defaultdict(list)
    for r in rows:
        if r["uri"]:
            bucket[r["uri"]].append(r)

    out = []
    for uri, plays in bucket.items():
        if len(plays) < MIN_PLAYS:
            continue
        total_ms = sum(p["ms_played"] for p in plays)
        meta = track_meta.get(uri, {})
        genres = artist_meta.get(meta.get("artist_mbid", ""), [])
        out.append({
            "uri": uri,
            "track": plays[0]["track"],
            "artist": plays[0]["artist"],
            "play_count": len(plays),
            "total_ms": total_ms,
            "avg_ms": total_ms / len(plays),
            "skip_rate": mean(1.0 if p["skipped"] else 0.0 for p in plays),
            "shuffle_rate": mean(1.0 if p["shuffle"] else 0.0 for p in plays),
            "hour_mean": mean(p["ts"].hour for p in plays),
            "hour_std": pstdev(p["ts"].hour for p in plays) if len(plays) > 1 else 0.0,
            "release_year": meta.get("release_year") or 0,
            "primary_genre": genres[0] if genres else "unknown",
        })
    return pd.DataFrame(out)


def detect_outliers(df, features):
    X = StandardScaler().fit_transform(df[features].values)
    iso = IsolationForest(
        contamination=CONTAMINATION, random_state=42, n_estimators=200
    )
    iso.fit(X)
    df = df.copy()
    df["anom_score"] = -iso.score_samples(X)  # higher = more anomalous
    df["is_outlier"] = iso.predict(X) == -1
    return df


def explain_outlier(row, baseline):
    reasons = []
    for feat, (mu, sigma) in baseline.items():
        if sigma == 0:
            continue
        z = (row[feat] - mu) / sigma
        if abs(z) >= 2:
            direction = "high" if z > 0 else "low"
            reasons.append(f"{feat}={row[feat]:.1f} ({direction}, z={z:+.1f})")
    return "; ".join(reasons) if reasons else "subtle multivariate anomaly"


def genre_rarity_outliers(df, top_n=TOP_N):
    genre_share = df.groupby("primary_genre")["total_ms"].sum()
    total = genre_share.sum()
    if total == 0:
        return df.iloc[:0]
    genre_share = genre_share / total
    rare_genres = genre_share[genre_share < 0.01].index
    rare = df[df["primary_genre"].isin(rare_genres)].copy()
    rare["genre_share"] = rare["primary_genre"].map(genre_share)
    return rare.sort_values("total_ms", ascending=False).head(top_n)


def report_user(name, rows, track_meta, artist_meta, out_dir):
    df = per_track_features(rows, track_meta, artist_meta)
    print(f"\n=== {name} === ({len(df)} tracks with >= {MIN_PLAYS} plays)")
    if len(df) < 20:
        print("  too few tracks to fit IsolationForest, skipping")
        return df

    flagged = detect_outliers(df, NUMERIC_FEATURES)
    baseline = {f: (df[f].mean(), df[f].std()) for f in NUMERIC_FEATURES}

    print(f"\n  IsolationForest top {TOP_N} outliers:")
    top = flagged.sort_values("anom_score", ascending=False).head(TOP_N)
    for _, r in top.iterrows():
        year = int(r["release_year"]) if r["release_year"] else "unk"
        print(f"    [{r['anom_score']:.3f}] {r['artist']} — {r['track']}")
        print(
            f"      plays={r['play_count']}, avg_ms={r['avg_ms']:.0f}, "
            f"skip={r['skip_rate']:.2f}, hour_mean={r['hour_mean']:.1f}, "
            f"year={year}"
        )
        print(f"      why: {explain_outlier(r, baseline)}")

    rare = genre_rarity_outliers(flagged)
    print(
        f"\n  Genre-rarity outliers (genre <1% of listening, "
        f"top {TOP_N} by play time):"
    )
    if rare.empty:
        print("    (none — every primary genre accounts for >=1%)")
    for _, r in rare.iterrows():
        print(
            f"    {r['artist']} — {r['track']} "
            f"[genre={r['primary_genre']}, share={r['genre_share']*100:.2f}%, "
            f"plays={r['play_count']}]"
        )

    out_path = out_dir / f"outliers_{name}.csv"
    flagged[[
        "track", "artist", "primary_genre", "play_count", "total_ms",
        "avg_ms", "skip_rate", "shuffle_rate", "hour_mean", "release_year",
        "anom_score", "is_outlier",
    ]].sort_values("anom_score", ascending=False).to_csv(out_path, index=False)
    print(f"\n  saved -> {out_path.relative_to(HERE.parent)}")
    return flagged


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

    for name, rows in user_rows.items():
        report_user(name, rows, track_meta, artist_meta, HERE)


if __name__ == "__main__":
    main()
