"""
Streamlit dashboard for check-in 3 (BONUS task).

Loads Vanessa + Darelle as baselines and lets the user upload one or more
additional Spotify extended-streaming-history CSVs via the sidebar (each
file becomes a new user, auto-named from its filename). All uploaded users
are included in every comparison view (Overview, Similarity, Clustering,
Outliers).

Run from the repo root:
    streamlit run dashboard.py
"""
import csv
import io
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# similarity.py and outlier.py live in checkin-3/ — make their pure functions importable
sys.path.insert(0, str(Path(__file__).parent / "checkin-3"))

from similarity import (
    USERS as BASELINE_USER_PATHS,
    TRACK_CACHE,
    ARTIST_CACHE,
    load_track_meta,
    load_artist_meta,
    features as build_similarity_features,
    similarity,
    genre_dist,
)
from outlier import (
    per_track_features,
    detect_outliers,
    genre_rarity_outliers,
    NUMERIC_FEATURES,
    MIN_PLAYS,
)

st.set_page_config(page_title="Group 9 — Music Wrapped Dashboard", layout="wide")
HERE = Path(__file__).parent


def _parse_play_rows(reader, year_filter):
    rows = []
    for r in reader:
        ts = r.get("ts") or ""
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            continue
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


@st.cache_data(show_spinner="Loading baseline user...")
def _load_plays_baseline(name, year_filter):
    path = BASELINE_USER_PATHS[name]
    with open(path) as f:
        return _parse_play_rows(csv.DictReader(f), year_filter)


def _load_plays_uploaded(file_bytes, year_filter):
    text = file_bytes.decode("utf-8-sig")
    return _parse_play_rows(csv.DictReader(io.StringIO(text)), year_filter)


@st.cache_data
def _load_track_meta_cached():
    return load_track_meta(TRACK_CACHE)


@st.cache_data
def _load_artist_meta_cached():
    return load_artist_meta(ARTIST_CACHE)


# === sidebar ===
st.sidebar.title("Controls")
year_choice = st.sidebar.selectbox("Year filter", ["2025", "all years"], index=0)
year_filter = 2025 if year_choice == "2025" else None

uploaded_files = st.sidebar.file_uploader(
    "Upload Spotify CSV(s) — one per user",
    type="csv",
    accept_multiple_files=True,
    help="Spotify extended-streaming-history schema. Each file becomes a new "
         "user, auto-named from its filename.",
)

st.sidebar.markdown("---")

# === load data ===
track_meta = _load_track_meta_cached()
artist_meta = _load_artist_meta_cached()

user_rows = {n: _load_plays_baseline(n, year_filter) for n in BASELINE_USER_PATHS}

for f in uploaded_files or []:
    name = Path(f.name).stem
    if name in user_rows:
        i = 2
        while f"{name}_{i}" in user_rows:
            i += 1
        name = f"{name}_{i}"
    user_rows[name] = _load_plays_uploaded(f.getvalue(), year_filter)
    st.sidebar.success(f"Loaded {len(user_rows[name]):,} plays for {name}")

names = list(user_rows.keys())

# === main layout ===
st.title("Group 9 — Music Wrapped Dashboard")
st.caption(
    f"Year filter: **{year_choice}** "
)

tab_overview, tab_sim, tab_cluster, tab_outlier = st.tabs(
    ["Overview", "Similarity", "Clustering", "Outliers"]
)


with tab_overview:
    st.subheader("Per-user overview")
    cols = st.columns(len(names))
    for col, name in zip(cols, names):
        rows = user_rows[name]
        col.markdown(f"### {name}")
        if not rows:
            col.warning("no plays in selected year")
            continue
        total_ms = sum(r["ms_played"] for r in rows)
        date_min = min(r["ts"] for r in rows).date()
        date_max = max(r["ts"] for r in rows).date()
        col.metric("plays", f"{len(rows):,}")
        col.metric("listening hours", f"{total_ms / 3_600_000:,.1f}")
        col.caption(f"{date_min} → {date_max}")

        artists = pd.Series(
            [r["artist"] for r in rows if r["artist"]]
        ).value_counts().head(10)
        col.markdown("**Top artists**")
        col.dataframe(
            artists.rename_axis("artist").reset_index(name="plays"),
            use_container_width=True, hide_index=True,
        )

        gd = genre_dist(rows, track_meta, artist_meta)
        if gd:
            top_g = pd.Series(gd).sort_values(ascending=False).head(10)
            top_g_pct = (top_g / top_g.sum() * 100).round(1)
            col.markdown("**Top genres** (% of enriched ms)")
            col.dataframe(
                top_g_pct.rename_axis("genre").reset_index(name="%"),
                use_container_width=True, hide_index=True,
            )
        else:
            col.caption("(no enrichment matches)")


with tab_sim:
    st.subheader("Pairwise similarity")
    st.caption(
        "Composite = mean of 5 sub-scores: 24-hour cosine, 12-month cosine, "
        "artist weighted Jaccard, genre cosine, release-year cosine."
    )
    feats = {n: build_similarity_features(user_rows[n], track_meta, artist_meta)
             for n in names if user_rows[n]}
    valid_names = list(feats.keys())

    if len(valid_names) < 2:
        st.warning("Need at least 2 users with plays in the selected year.")
    else:
        sim_mat = pd.DataFrame(index=valid_names, columns=valid_names, dtype=float)
        for a in valid_names:
            for b in valid_names:
                sim_mat.loc[a, b] = (
                    1.0 if a == b
                    else similarity(feats[a], feats[b])["composite"]
                )
        st.markdown("**Composite similarity matrix**")
        st.dataframe(
            sim_mat.style.format("{:.3f}").background_gradient(
                cmap="viridis", vmin=0, vmax=1
            ),
            use_container_width=True,
        )

        st.markdown("**Per-category breakdown for a selected pair**")
        c1, c2 = st.columns(2)
        user_a = c1.selectbox("User A", valid_names, index=0, key="sim_a")
        user_b = c2.selectbox(
            "User B", valid_names, index=min(1, len(valid_names) - 1), key="sim_b"
        )
        if user_a == user_b:
            st.info("Pick two different users.")
        else:
            scores = similarity(feats[user_a], feats[user_b])
            st.bar_chart(pd.Series(scores).rename("similarity"))


with tab_cluster:
    st.subheader("Clustering — daily behavioral features")
    st.caption(
        "Each point = one day of listening. Features per day: total_ms, "
        "track_count, shuffle_ratio, skip_ratio, unique_artists, avg_hour. "
        "PCA → 2D."
    )

    frames = []
    for name in names:
        rows = user_rows[name]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["ts"]).dt.date
        agg = df.groupby("date").agg(
            total_ms=("ms_played", "sum"),
            track_count=("ms_played", "count"),
            shuffle_ratio=("shuffle", "mean"),
            skip_ratio=("skipped", "mean"),
            unique_artists=("artist", "nunique"),
            avg_hour=("ts", lambda x: x.dt.hour.mean()),
        ).reset_index()
        agg["person"] = name
        frames.append(agg)

    if not frames:
        st.warning("No daily data in selected year for any user.")
    else:
        combined = pd.concat(frames, ignore_index=True)
        feat_cols = ["total_ms", "track_count", "shuffle_ratio", "skip_ratio",
                     "unique_artists", "avg_hour"]
        X_scaled = StandardScaler().fit_transform(combined[feat_cols])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        combined["PC1"], combined["PC2"] = X_pca[:, 0], X_pca[:, 1]

        k = st.slider("k for KMeans", 2, 5, len(frames))
        clusters = KMeans(
            n_clusters=k, random_state=42, n_init=10
        ).fit_predict(X_scaled)
        combined["cluster"] = ["cluster " + str(c) for c in clusters]

        var1, var2 = pca.explained_variance_ratio_ * 100
        st.caption(f"PC1: {var1:.1f}% of variance · PC2: {var2:.1f}% of variance")

        c1, c2 = st.columns(2)
        c1.markdown("**Days colored by user (ground truth)**")
        c1.scatter_chart(combined, x="PC1", y="PC2", color="person")
        c2.markdown(f"**Days colored by KMeans cluster (k={k}, unsupervised)**")
        c2.scatter_chart(combined, x="PC1", y="PC2", color="cluster")


with tab_outlier:
    st.subheader("Outlier tracks per user")
    st.caption(
        "IsolationForest on per-track features (play_count, total/avg ms, "
        "skip/shuffle rate, hour mean+std, release year). Genre-rarity pass "
        "lists tracks whose primary genre accounts for <1% of listening time."
    )
    target = st.selectbox("User", names, key="outlier_user")
    rows = user_rows[target]
    df_tracks = per_track_features(rows, track_meta, artist_meta)

    if len(df_tracks) < 20:
        st.info(
            f"{target} has only {len(df_tracks)} tracks with >= {MIN_PLAYS} "
            "plays — too few to fit IsolationForest reliably."
        )
    else:
        flagged = detect_outliers(df_tracks, NUMERIC_FEATURES)
        top = flagged.sort_values("anom_score", ascending=False).head(15)
        st.markdown("**Top 15 IsolationForest outliers**")
        st.dataframe(
            top[["artist", "track", "primary_genre", "play_count", "total_ms",
                 "avg_ms", "skip_rate", "hour_mean", "release_year",
                 "anom_score"]],
            use_container_width=True, hide_index=True,
        )

        rare = genre_rarity_outliers(flagged)
        st.markdown(
            "**Genre-rarity outliers** (primary genre <1% of listening time)"
        )
        if rare.empty:
            st.info("No rare-genre tracks for this user.")
        else:
            st.dataframe(
                rare[["artist", "track", "primary_genre", "genre_share",
                      "play_count", "total_ms"]],
                use_container_width=True, hide_index=True,
            )
