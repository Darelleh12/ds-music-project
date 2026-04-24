import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Load data
van_df = pd.read_csv('vans-listening-data_2021-2026.csv')
l_df = pd.read_csv('l-listening-data_2017-26.csv')

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
combined = pd.concat([process_data(van_df, 'Van'), process_data(l_df, 'l')])

# 3. Scaling & PCA
features = ['total_ms', 'track_count', 'shuffle_ratio', 'skip_ratio', 'unique_artists', 'avg_hour']
X_scaled = StandardScaler().fit_transform(combined[features])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
combined['PC1'], combined['PC2'] = X_pca[:, 0], X_pca[:, 1]

# 4. K-Means Clustering
combined['cluster'] = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X_scaled)

# 5. Plotting
plt.figure(figsize=(14, 6))

# Plot actual identities
plt.subplot(1, 2, 1)
sns.scatterplot(data=combined, x='PC1', y='PC2', hue='person', alpha=0.5)
plt.title('Actual Labels (Identity)')

# Plot K-Means results
plt.subplot(1, 2, 2)
sns.scatterplot(data=combined, x='PC1', y='PC2', hue='cluster', palette='viridis', alpha=0.5)
plt.title('K-Means Clusters (Discovered Patterns)')

plt.tight_layout()
plt.show()