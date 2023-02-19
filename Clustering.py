from sklearn.cluster import DBSCAN
import numpy as np

max_dist = 0.5
min_samples = 2

# Uses euclidean distance so all need to be across the same range
def ClusterMetrics(metrics):
    clustering = DBSCAN(eps=max_dist, min_samples=min_samples).fit(metrics)
    labels = clustering.labels_
    return labels