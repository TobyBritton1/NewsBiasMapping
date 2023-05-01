from sklearn.cluster import AgglomerativeClustering, Birch

def BirchCluster(metrics, threshold=0.4, n_clusters=None):
    clustering = Birch(n_clusters=n_clusters, threshold=threshold).fit(metrics)
    labels = clustering.labels_
    return labels

def AgglomerativeCluster(metrics, threshold=5.5, n_clusters=None):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=threshold, linkage='ward').fit(metrics)
    labels = clustering.labels_
    return labels