import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


data_frame = pd.read_csv(Path(__file__).with_name("cluster_data.csv"))
data = data_frame.to_numpy(dtype=float)

k = 5 

random_indices = np.random.default_rng(42).choice(data.shape[0], size=k, replace=False)
centroids = data[random_indices]

clusters = [{"center": centroids[i], "points": []} for i in range(k)]
    

def euclideanDistance(center,point2) :
    dist = np.sqrt(np.sum( ( np.array(center) - np.array(point2) ) **2 ))
    return dist


def assignPoint(data,clusters):

    for c in clusters:

        c["points"] = []

    for point in data:

        dist = []
    
        for cluster in clusters:

            dist.append(euclideanDistance(cluster["center"],point))

        center = np.argmin(dist)
        clusters[center]["points"].append(point)
        

def calculateCentroids(Clusters):


    for cluster in Clusters:
        if len(cluster["points"]) > 0:
            cluster["center"] = np.mean(cluster["points"], axis=0)  

    return Clusters
    
    


max_iters = 100

for iteration in range(1, max_iters + 1):
  old_centers = [c["center"].copy() for c in clusters]

  assignPoint(data, clusters)
  calculateCentroids(clusters)

  new_centers = [c["center"] for c in clusters]

  if np.allclose(old_centers, new_centers):
    break

# Evaluate assignments against the final centroids, including at the iteration limit.
assignPoint(data, clusters)
centers = np.array([cluster["center"] for cluster in clusters])
squared_distances = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
labels = np.argmin(squared_distances, axis=1)
inertia = np.sum(squared_distances[np.arange(len(data)), labels])
cluster_sizes = np.bincount(labels, minlength=k)

print(f"K-means evaluation (k={k}, iterations={iteration})")
print(f"Inertia (within-cluster sum of squares; lower is better): {inertia:.4f}")
if 1 < len(np.unique(labels)) < len(data):
    print(f"Silhouette score (higher is better): {silhouette_score(data, labels):.4f}")
    print(f"Davies-Bouldin index (lower is better): {davies_bouldin_score(data, labels):.4f}")
    print(f"Calinski-Harabasz index (higher is better): {calinski_harabasz_score(data, labels):.4f}")
else:
    print("Clustering scores require between 2 and n_samples - 1 nonempty clusters.")
for index, size in enumerate(cluster_sizes):
    print(f"Cluster {index + 1}: {size} points")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
fig.suptitle(f"K-means Clustering (k={k})")
colors = plt.get_cmap("tab10")(np.arange(k) % 10)
for index in range(k):
    points = data[labels == index]
    axes[0].scatter(points[:, 0], points[:, 1], color=colors[index],
                    s=25, alpha=0.7, label=f"Cluster {index + 1}")
axes[0].scatter(centers[:, 0], centers[:, 1], marker="X", s=180,
                c="black", edgecolors="white", label="Centroids")
axes[0].set_xlabel(data_frame.columns[0])
axes[0].set_ylabel(data_frame.columns[1])
axes[0].set_title("Clusters and Centroids")
axes[0].legend()

bars = axes[1].bar(np.arange(1, k + 1), cluster_sizes, color=colors)
axes[1].bar_label(bars, padding=3)
axes[1].set_xticks(np.arange(1, k + 1))
axes[1].set_xlabel("Cluster")
axes[1].set_ylabel("Number of Points")
axes[1].set_title("Cluster Sizes")
axes[1].margins(y=0.15)

output_path = Path(__file__).with_name("kmc_evaluation.png")
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Visualization saved to: {output_path}")
plt.show()
