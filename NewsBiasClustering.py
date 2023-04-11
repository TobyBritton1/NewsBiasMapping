from collections import Counter
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from Metrics import getSentiment, getQuoteBased, getSensationalized, getMudslinging, getSpin, getInformal
from Clustering import BirchCluster


def NewsBiasClustering(articles, threshold=0.4, n_clusters=None):    
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
    scaled_data = pd.DataFrame(scalar.fit_transform(data),columns=['sentiment','quoteBased','sensationalized','mudslinging','spin','informal']) #scaling the data
    #sns.heatmap(scaled_data.corr())
    pca = PCA(n_components = 2)
    pca.fit(scaled_data)
    data_pca = pca.transform(scaled_data)
    data_pca = pd.DataFrame(data_pca,columns=['PC1','PC2'])
    #sns.heatmap(data_pca.corr())

    # Output Clustered Graph
    for i in range(len(Counter(labels))):
        pc1 = []
        pc2 = []
        for j in range(len(labels)):
            if labels[j] == i:
                pc1.append(data_pca.iloc[j]['PC1'])
                pc2.append(data_pca.iloc[j]['PC2'])
        plt.scatter(pc1,pc2,s=5) 
    plt.show()

def NewsBiasClusteringFromMetrics(metrics, threshold=0.4, n_clusters=None):    
    # Cluster metrics
    labels = BirchCluster(metrics, threshold=threshold, n_clusters=n_clusters)
    print(Counter(labels))

    # Reduce Data Dimensions
    data = pd.DataFrame(metrics,columns=['sentiment','quoteBased','sensationalized','mudslinging','spin','informal'])
    scalar = StandardScaler()
    scaled_data = pd.DataFrame(scalar.fit_transform(data),columns=['sentiment','quoteBased','sensationalized','mudslinging','spin','informal']) #scaling the data
    #sns.heatmap(scaled_data.corr())
    pca = PCA(n_components = 2)
    pca.fit(scaled_data)
    data_pca = pca.transform(scaled_data)
    data_pca = pd.DataFrame(data_pca,columns=['PC1','PC2'])
    #sns.heatmap(data_pca.corr())

    # Output Clustered Graph
    for i in range(len(Counter(labels))):
        pc1 = []
        pc2 = []
        for j in range(len(labels)):
            if labels[j] == i:
                pc1.append(data_pca.iloc[j]['PC1'])
                pc2.append(data_pca.iloc[j]['PC2'])
        plt.scatter(pc1,pc2,s=5)
    plt.xlim(-8, 10)
    plt.ylim(-5, 15)
    plt.show()