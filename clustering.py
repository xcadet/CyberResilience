# Code for clustering the data
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
import numpy as np


def agglomerative_clustering(all_vectors, num_clusters=3, linkage_method="ward"):
    """
    Perform agglomerative hierarchical clustering using scikit-learn given a distance matrix.

    Parameters:
    - distance_matrix (numpy.ndarray): A square symmetric distance matrix.
    - num_clusters (int): The desired number of clusters.
    - linkage (str): The linkage method ('ward', 'complete', 'average', 'single').

    Returns:
    - clusters (numpy.ndarray): Cluster labels for each data point.
    """
    model = AgglomerativeClustering(
        n_clusters=num_clusters,
        affinity="euclidean",
        linkage=linkage_method,
    )

    clusters = model.fit(all_vectors)
    return clusters.labels_


def reorder_clusters_by_size(clusters: np.ndarray) -> np.ndarray:
    cluster_to_count = Counter(clusters)
    remap = {
        k: v
        for v, k in enumerate(
            sorted(cluster_to_count, key=cluster_to_count.get, reverse=True)
        )
    }
    clusters = np.array([remap[c] for c in clusters])
    return clusters
