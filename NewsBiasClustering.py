import matplotlib.pyplot as plt
import pandas as pd
 # import seaborn as sns
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from Metrics import getSentiment, getQuoteBased, getSensationalised, getMudslinging, getSpin, getInformal
from Clustering import BirchCluster

def getMetrics(articles):
    # Get Metrics
    metrics = []
    for i in range(len(articles)):
        sentiment = getSentiment(articles.at[i,'body'])
        quoteBased = getQuoteBased(articles.at[i,'body'])
        sensationalised = getSensationalised(articles.at[i,'body'])
        mudslinging = getMudslinging(articles.at[i,'body'])
        spin = getSpin(articles.at[i,'body'])
        informal = getInformal(articles.at[i,'body'])
        metrics.append([sentiment, quoteBased, sensationalised, mudslinging, spin, informal])
    return metrics

def getWord(value):
    if value < 0.2:
        return 'very low '
    elif value < 0.4:
        return 'low '
    elif value < 0.6:
        return 'medium '
    elif value < 0.8:
        return 'high '
    else:
        return 'very high '

def NewsBiasClusteringFromMetrics(metrics, threshold=0.32, n_clusters=None):
    # Reduce Data Dimensions
    data = pd.DataFrame(metrics,columns=['sentiment','quoteBased','sensationalised','mudslinging','spin','informal'])
    scalar = StandardScaler()
    scaledData = pd.DataFrame(scalar.fit_transform(data),columns=['sentiment','quoteBased','sensationalised','mudslinging','spin','informal']) #scaling the data
    # sns.heatmap(scaledData.corr())
    # plt.show()
    pca = PCA(n_components = 2)
    pca.fit(scaledData)
    dataPCA = pca.transform(scaledData)
    dataPCA = pd.DataFrame(dataPCA,columns=['PC1','PC2'])
    # sns.heatmap(dataPCA.corr())
    # plt.show()

    # Cluster metrics
    labels = BirchCluster(metrics, threshold=threshold, n_clusters=n_clusters)
    print(Counter(labels))

    # Get Dominant Metrics For Each Label
    data = pd.DataFrame(metrics,columns=['sentiment','quoteBased','sensationalised','mudslinging','spin','informal'])
    data['label'] = labels
    means = data.groupby(['label']).mean()
    dominantMetrics = []
    dominantMetricsWords = []
    for i in range(len(Counter(labels))):
        dominant = []
        dominantWords = []
        for col in means.columns:
            mean = means[col].mean()
            std = means[col].std()
            outliers = means[col][(means[col] < mean-(std*0.8)) | (means[col] > mean+(std*0.8))]
            try:
                dominant.append([col, round(outliers[i],1)])
                dominantWords.append(getWord(outliers[i])+col)
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
                    maxMean = means.iloc[i][col]
            dominant.append([maxCol, round(maxMean,1)])
            dominantWords.append(getWord(maxMean)+maxCol)
        dominantMetrics.append(dominant)
        dominantMetricsWords.append(dominantWords)
    dominantMetricsDataFrame = pd.DataFrame([i for i in range(len(Counter(labels)))], columns=['label'])
    dominantMetricsDataFrame['dominantMetrics'] = dominantMetrics
    dominantMetricsDataFrame['dominantMetricsWords'] = dominantMetricsWords

    # Output Clustered Graph
    for i in range(len(Counter(labels))):
        pc1 = []
        pc2 = []
        for j in range(len(labels)):
            if labels[j] == i:
                pc1.append(dataPCA.iloc[j]['PC1'])
                pc2.append(dataPCA.iloc[j]['PC2'])
        plt.scatter(pc1,pc2,s=5,label=', '.join(dominantMetricsWords[i]))
    plt.xlim(-8, 10)
    plt.ylim(-5, 15)
    plt.legend()
    plt.show()
    return labels, dataPCA, dominantMetricsDataFrame

def NewsBiasClustering(articles, threshold=0.32, n_clusters=None):    
    metrics = getMetrics(articles)
    return NewsBiasClusteringFromMetrics(metrics, threshold, n_clusters)