from sklearn.cluster import DBSCAN

# Uses euclidean distance so all need to be across the same range
def ClusterMetrics(metrics, max_dist=0.5, min_samples=2):
    clustering = DBSCAN(eps=max_dist, min_samples=min_samples).fit(metrics)
    labels = clustering.labels_
    return labels