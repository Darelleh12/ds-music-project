from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Load data
DATA_DIR = Path(__file__).parent
van_df = pd.read_csv(DATA_DIR / 'vans-listening-data_2021-2026.csv')
l_df = pd.read_csv(DATA_DIR / 'l-listening-data_2017-26.csv')

# 2. Aggregation Function (Behavioral Features)
def process_data(df, name):
    df['ts'] = pd.to_datetime(df['ts'])
    daily = df.groupby(df['ts'].dt.date).agg(
        total_ms=('ms_played', 'sum'),
        track_count=('ms_played', 'count'),
        shuffle_ratio=('shuffle', 'mean'),
        skip_ratio=('skipped', 'mean'),
        unique_artists=('master_metadata_album_artist_name', 'nunique'),
        avg_hour=('ts', lambda x: x.dt.hour.mean())
    ).reset_index()
    daily['person'] = name
    return daily

# Combine dataset
combined = pd.concat([process_data(van_df, 'Vanessa'), process_data(l_df, 'Darelle')])

# 3. Scaling & PCA
features = ['total_ms', 'track_count', 'shuffle_ratio', 'skip_ratio', 'unique_artists', 'avg_hour']
X_scaled = StandardScaler().fit_transform(combined[features])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
combined['PC1'], combined['PC2'] = X_pca[:, 0], X_pca[:, 1]

# 4. K-Means Clustering
combined['cluster'] = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X_scaled)

# 5. Plotting
var1, var2 = pca.explained_variance_ratio_ * 100
xlabel = f'PC1 ({var1:.1f}% of variance)'
ylabel = f'PC2 ({var2:.1f}% of variance)'

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

# Plot actual identities
sns.scatterplot(data=combined, x='PC1', y='PC2', hue='person', alpha=0.5, ax=ax0)
ax0.set_title('Days colored by actual user (ground truth)')
ax0.set_xlabel(xlabel)
ax0.set_ylabel(ylabel)
ax0.legend(title='User')

# Plot K-Means results
sns.scatterplot(data=combined, x='PC1', y='PC2', hue='cluster', palette='viridis', alpha=0.5, ax=ax1)
ax1.set_title('Days colored by K-means cluster (k=2, unsupervised)')
ax1.set_xlabel(xlabel)
ax1.set_ylabel(ylabel)
ax1.legend(title='Cluster')

fig.suptitle(
    'Do daily listening habits separate the two users?\n'
    'Each point = one day of listening, projected to 2D via PCA.\n'
    f'Features: {", ".join(features)}',
    fontsize=11,
)

plt.tight_layout()
plt.show()