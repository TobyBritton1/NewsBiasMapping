from sklearn.cluster import DBSCAN
import numpy as np

# Uses euclidean distance so all need to be across the same range
def ClusterMetrics(metrics, max_dist=0.5, min_samples=2):
    clustering = DBSCAN(eps=max_dist, min_samples=min_samples).fit(metrics)
    labels = clustering.labels_
    return labels

def getBestDistance(metrics):
    max = 0
    bestSamples = 0
    bestDist = 0
    for samples in range(6,10):
        dist = 0.01
        while dist <= 0.5:
            labels = ClusterMetrics(metrics, dist, samples)
            labelsCount = len(set(labels))
            withGroup = 1 - (np.count_nonzero(labels == -1) / len(labels))
            maximizeValue = withGroup * withGroup * labelsCount # Trying to maximize the number of groups and minimize the number of articles without a group
            if maximizeValue > max and withGroup >= 0.9: # Threshold to ensure over 90% of the articles must have a group
                bestSamples = samples
                bestDist = dist
                max = maximizeValue
            dist = round(dist + 0.01,3)
    return bestSamples, bestDist

def getBestLabels(metrics):
    samples, dist = getBestDistance(metrics)
    labels = ClusterMetrics(metrics,dist,samples)
    return labels
