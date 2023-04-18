from collections import Counter
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from Metrics import getSentiment, getQuoteBased, getSensationalized, getMudslinging, getSpin, getInformal
from Clustering import BirchCluster


def NewsBiasClustering(articles, threshold=0.32, n_clusters=None):    
    # Get Metrics
    metrics = []
    for i in range(len(articles)):
        sentiment = getSentiment(articles.at[i,'body'])
        quoteBased = getQuoteBased(articles.at[i,'body'])
        sensationalized = getSensationalized(articles.at[i,'body'])
        mudslinging = getMudslinging(articles.at[i,'body'])
        spin = getSpin(articles.at[i,'body'])
        informal = getInformal(articles.at[i,'body'])
        metrics.append([sentiment, quoteBased, sensationalized, mudslinging, spin, informal])
    
    # Cluster metrics
    labels = BirchCluster(metrics, threshold=threshold, n_clusters=n_clusters)
    print(Counter(labels))

    # Reduce Data Dimensions
    data = pd.DataFrame(metrics,columns=['sentiment','quoteBased','sensationalized','mudslinging','spin','informal'])
    scalar = StandardScaler()
    scaledData = pd.DataFrame(scalar.fit_transform(data),columns=['sentiment','quoteBased','sensationalized','mudslinging','spin','informal']) #scaling the data
    #sns.heatmap(scaledData.corr())
    pca = PCA(n_components = 2)
    pca.fit(scaledData)
    dataPCA = pca.transform(scaledData)
    dataPCA = pd.DataFrame(dataPCA,columns=['PC1','PC2'])
    #sns.heatmap(dataPCA.corr())

    # Output Clustered Graph
    for i in range(len(Counter(labels))):
        pc1 = []
        pc2 = []
        for j in range(len(labels)):
            if labels[j] == i:
                pc1.append(dataPCA.iloc[j]['PC1'])
                pc2.append(dataPCA.iloc[j]['PC2'])
        plt.scatter(pc1,pc2,s=5) 
    plt.show()

def NewsBiasClusteringFromMetrics(metrics, threshold=0.32, n_clusters=None):    
    # Cluster metrics
    labels = BirchCluster(metrics, threshold=threshold, n_clusters=n_clusters)
    print(Counter(labels))

    # Reduce Data Dimensions
    data = pd.DataFrame(metrics,columns=['sentiment','quoteBased','sensationalized','mudslinging','spin','informal'])
    scalar = StandardScaler()
    scaledData = pd.DataFrame(scalar.fit_transform(data),columns=['sentiment','quoteBased','sensationalized','mudslinging','spin','informal']) #scaling the data
    #sns.heatmap(scaledData.corr())
    pca = PCA(n_components = 2)
    pca.fit(scaledData)
    dataPCA = pca.transform(scaledData)
    dataPCA = pd.DataFrame(dataPCA,columns=['PC1','PC2'])
    #sns.heatmap(dataPCA.corr())

    # Output Clustered Graph
    for i in range(len(Counter(labels))):
        pc1 = []
        pc2 = []
        for j in range(len(labels)):
            if labels[j] == i:
                pc1.append(dataPCA.iloc[j]['PC1'])
                pc2.append(dataPCA.iloc[j]['PC2'])
        plt.scatter(pc1,pc2,s=5)
    plt.xlim(-8, 10)
    plt.ylim(-5, 15)
    plt.show()

    # Get Dominant Metrics For Each Label
    data['label'] = labels

    means = data.groupby(['label']).mean()
    print(means)

    dominantMetrics = []
    for i in range(len(Counter(labels))):
        dominant = []
        for col in means.columns:
            mean = means[col].mean()
            std = means[col].std()
            outliers = means[col][(means[col] < mean-(std*0.8)) | (means[col] > mean+(std*0.8))]
            try:
                dominant.append([col, round(outliers[i],1)])
            except:
                continue
        if len(dominant) == 0:
            maxDist = 0
            maxMean = 0
            maxCol = None
            for col in means.columns:
                mean = means[col].mean()
                if abs(means.iloc[i][col] - mean) > maxDist:
                    maxDist = abs(means.iloc[i][col] - mean)
                    maxCol = col
                    maxMean = round(means.iloc[i][col],1)
            dominant.append([maxCol, maxMean])
        dominantMetrics.append(dominant)
    dominantMetricsDataFrame = pd.DataFrame([i for i in range(len(Counter(labels)))], columns=['label'])
    dominantMetricsDataFrame['dominantMetrics'] = dominantMetrics
    return labels, dataPCA, dominantMetricsDataFrame