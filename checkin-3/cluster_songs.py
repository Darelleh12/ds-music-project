"""Track-level K-means clustering on per-track behavioral + metadata features.

Each point is one track played at least 5 times across both users. Features:
skip_rate, shuffle_rate, avg_ms_played, log(plays), avg_hour, release_year.
Clusters are compared against ground truth (primary listener: Vanessa or Darelle).

Output: cluster_songs.png (side-by-side ground-truth vs KMeans clusters).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).parent
REPO = HERE.parent
DATA = REPO / 'vans-bs'
META = HERE / 'track_metadata.csv'
OUT = HERE / 'cluster_songs.png'

URI = 'spotify_track_uri'
MIN_PLAYS = 5
FEATURES = ['skip_rate', 'shuffle_rate', 'avg_ms', 'log_plays', 'avg_hour', 'release_year']


def main() -> None:
    van = pd.read_csv(DATA / 'vans-listening-data_2021-2026.csv')
    van['user'] = 'Vanessa'
    dar = pd.read_csv(DATA / 'l-listening-data_2017-26.csv')
    dar['user'] = 'Darelle'

    df = pd.concat([van, dar], ignore_index=True)
    df['ts'] = pd.to_datetime(df['ts'])
    df['hour'] = df['ts'].dt.hour
    df = df[df[URI].notna()]

    agg = df.groupby(URI).agg(
        track=('master_metadata_track_name', 'first'),
        artist=('master_metadata_album_artist_name', 'first'),
        plays=('ms_played', 'count'),
        avg_ms=('ms_played', 'mean'),
        skip_rate=('skipped', 'mean'),
        shuffle_rate=('shuffle', 'mean'),
        avg_hour=('hour', 'mean'),
        vanessa_plays=('user', lambda s: (s == 'Vanessa').sum()),
        darelle_plays=('user', lambda s: (s == 'Darelle').sum()),
    ).reset_index()

    tracks = agg[agg['plays'] >= MIN_PLAYS].copy()
    tracks['primary'] = np.where(
        tracks['vanessa_plays'] >= tracks['darelle_plays'], 'Vanessa', 'Darelle'
    )
    tracks['log_plays'] = np.log1p(tracks['plays'])

    tmeta = pd.read_csv(META)[['spotify_track_uri', 'release_year']]
    tracks = tracks.merge(tmeta, on='spotify_track_uri', how='left')
    tracks['release_year'] = tracks['release_year'].fillna(tracks['release_year'].median())

    X_scaled = StandardScaler().fit_transform(tracks[FEATURES].values)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    tracks['PC1'], tracks['PC2'] = X_pca[:, 0], X_pca[:, 1]
    tracks['cluster'] = KMeans(n_clusters=2, random_state=42, n_init=20).fit_predict(X_scaled)

    ari = adjusted_rand_score(tracks['primary'], tracks['cluster'])
    v1, v2 = pca.explained_variance_ratio_ * 100

    print(f'Total plays: {len(df):,}')
    print(f'Tracks with >={MIN_PLAYS} plays: {len(tracks):,}')
    print(f'PC1: {v1:.1f}%  PC2: {v2:.1f}%  total {v1 + v2:.1f}%')
    print(f'ARI: {ari:.3f}')
    print(f"Primary: {tracks['primary'].value_counts().to_dict()}")
    print(f"Cluster: {tracks['cluster'].value_counts().to_dict()}")
    print('\nCross-tab:')
    print(pd.crosstab(tracks['primary'], tracks['cluster']))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    sns.scatterplot(
        data=tracks, x='PC1', y='PC2', hue='primary',
        alpha=0.5, ax=ax0, s=20, edgecolor='none',
        palette={'Vanessa': '#ff7f0e', 'Darelle': '#1f77b4'},
    )
    ax0.set_title('Tracks colored by primary listener (ground truth)', fontsize=12)
    ax0.set_xlabel(f'PC1 ({v1:.1f}% var)')
    ax0.set_ylabel(f'PC2 ({v2:.1f}% var)')
    ax0.legend(title='Primary listener', loc='upper right')

    sns.scatterplot(
        data=tracks, x='PC1', y='PC2', hue='cluster',
        alpha=0.5, ax=ax1, s=20, edgecolor='none',
        palette={0: '#888888', 1: '#2ca02c'},
    )
    ax1.set_title('Tracks colored by K-means cluster (k=2, unsupervised)', fontsize=12)
    ax1.set_xlabel(f'PC1 ({v1:.1f}% var)')
    ax1.set_ylabel(f'PC2 ({v2:.1f}% var)')
    ax1.legend(title='Cluster', loc='upper right')

    fig.suptitle(
        "Do tracks separate the two users by how they're consumed?\n"
        f'Each point = one track ({len(tracks):,} tracks with >={MIN_PLAYS} plays). '
        f"Features: {', '.join(FEATURES)}.\n"
        f'ARI = {ari:.2f}',
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches='tight')
    print(f'\nSaved {OUT}')


if __name__ == '__main__':
    main()
