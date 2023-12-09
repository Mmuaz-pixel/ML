import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample data (you should replace this with your own dataset)
data = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])

# Specify the number of clusters (K)
k = 2

# Create a K-Means model
kmeans = KMeans(n_clusters=k)

# Fit the model to your data
kmeans.fit(data)

# Get cluster labels for each data point
labels = kmeans.labels_

# Get cluster centroids
centroids = kmeans.cluster_centers_
print(centroids)

# Plot the data points and centroids
colors = ["g.", "r."]

for i in range(len(data)):
    plt.plot(data[i][0], data[i][1], colors[labels[i]], markersize=10)
    
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
plt.show()
