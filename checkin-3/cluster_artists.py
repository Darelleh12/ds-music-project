"""Artist-level K-means clustering on TF-IDF genre tags.

For each of the top-200 artists by combined listening time, builds a TF-IDF
vector over genre tags from MusicBrainz, then clusters with k=2 and compares
against ground truth (primary listener: Vanessa or Darelle).

Output: cluster_artists.png (side-by-side ground-truth vs KMeans clusters).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score


HERE = Path(__file__).parent
REPO = HERE.parent
DATA = REPO / 'vans-bs'
META = HERE / 'artist_metadata.csv'
OUT = HERE / 'cluster_artists.png'

TOP_N = 200


def per_artist(df: pd.DataFrame, user: str) -> pd.DataFrame:
    g = df.groupby('master_metadata_album_artist_name', dropna=True)['ms_played'].sum().reset_index()
    g.columns = ['artist', f'{user}_ms']
    return g


def main() -> None:
    van = pd.read_csv(DATA / 'vans-listening-data_2021-2026.csv')
    dar = pd.read_csv(DATA / 'l-listening-data_2017-26.csv')

    artists = (
        per_artist(van, 'vanessa')
        .merge(per_artist(dar, 'darelle'), on='artist', how='outer')
        .fillna(0)
    )
    artists['total_ms'] = artists['vanessa_ms'] + artists['darelle_ms']
    artists['primary'] = np.where(
        artists['vanessa_ms'] >= artists['darelle_ms'], 'Vanessa', 'Darelle'
    )

    meta = pd.read_csv(META)[['artist_name', 'genres']].rename(columns={'artist_name': 'artist'})
    artists = artists.merge(meta, on='artist', how='left')
    artists = (
        artists[artists['genres'].notna()]
        .sort_values('total_ms', ascending=False)
        .head(TOP_N)
        .reset_index(drop=True)
    )

    # TF-IDF on '|'-separated genre tags
    vec = TfidfVectorizer(
        tokenizer=lambda s: s.split('|'),
        token_pattern=None,
        lowercase=False,
        min_df=2,
    )
    X = vec.fit_transform(artists['genres'])

    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X)
    artists['PC1'], artists['PC2'] = X_2d[:, 0], X_2d[:, 1]

    artists['cluster'] = KMeans(n_clusters=2, random_state=42, n_init=20).fit_predict(X.toarray())

    ari = adjusted_rand_score(artists['primary'], artists['cluster'])
    v1, v2 = svd.explained_variance_ratio_ * 100

    print(f'Top {TOP_N} artists used; {X.shape[1]} genre features after min_df=2')
    print(f'SVD variance: {v1:.1f}% + {v2:.1f}% = {v1 + v2:.1f}%')
    print(f'ARI(cluster vs primary listener): {ari:.3f}')
    print(f"Primary: {artists['primary'].value_counts().to_dict()}")
    print(f"Cluster: {artists['cluster'].value_counts().to_dict()}")
    print('\nCross-tab:')
    print(pd.crosstab(artists['primary'], artists['cluster']))

    # Identify the niche cluster (smaller of the two) for annotation
    sizes = artists['cluster'].value_counts()
    niche_id = int(sizes.idxmin())
    niche = artists[artists['cluster'] == niche_id]
    niche_pure = (niche['primary'] == niche['primary'].mode().iat[0]).mean() * 100

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    sns.scatterplot(
        data=artists, x='PC1', y='PC2', hue='primary',
        alpha=0.75, ax=ax0, s=50, edgecolor='none',
        palette={'Vanessa': '#ff7f0e', 'Darelle': '#1f77b4'},
    )
    ax0.set_title('Artists colored by primary listener (ground truth)', fontsize=12)
    ax0.set_xlabel(f'SVD1 ({v1:.1f}% var)')
    ax0.set_ylabel(f'SVD2 ({v2:.1f}% var)')
    ax0.legend(title='Primary listener', loc='upper right')

    sns.scatterplot(
        data=artists, x='PC1', y='PC2', hue='cluster',
        alpha=0.75, ax=ax1, s=50, edgecolor='none',
        palette={c: '#888888' if c != niche_id else '#2ca02c' for c in artists['cluster'].unique()},
    )
    ax1.set_title('Artists colored by K-means cluster (k=2, unsupervised)', fontsize=12)
    ax1.set_xlabel(f'SVD1 ({v1:.1f}% var)')
    ax1.set_ylabel(f'SVD2 ({v2:.1f}% var)')
    ax1.legend(title='Cluster', loc='upper right')

    nx, ny = niche['PC1'].mean(), niche['PC2'].mean()
    ax1.annotate(
        f'Niche cluster\n({len(niche)} artists, {niche_pure:.0f}% one user)',
        xy=(nx, ny), xytext=(nx + 0.15, ny + 0.25),
        fontsize=10, color='#2ca02c', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5),
    )

    fig.suptitle(
        "Do artists' genre profiles separate the two users?\n"
        f'Each point = one artist (top {TOP_N} by total listening). KMeans on TF-IDF genre tags.\n'
        f'ARI = {ari:.2f}',
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches='tight')
    print(f'\nSaved {OUT}')


if __name__ == '__main__':
    main()
