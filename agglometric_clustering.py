import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Create synthetic data for clustering (you can replace this with your own data)
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Perform agglomerative clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)  # You can specify the number of clusters
labels = agg_clustering.fit_predict(X)

# Plot the data points colored by their cluster assignments
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("Agglomerative Clustering")
plt.show()

# Create a dendrogram to visualize the hierarchy of cluster mergers
linkage_matrix = linkage(X, method='ward')  # Ward's linkage is commonly used
dendrogram(linkage_matrix)
plt.title("Dendrogram")
plt.show()
