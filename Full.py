
import json
import nltk
import pandas as pd

from Metrics import getSentiment, getQuoteBased, getSensationalized, getMudslinging, getSpin, getInformal
from NewsBiasClustering import NewsBiasClusteringFromMetrics

nltk.download('vader_lexicon')

print("Loading File")
file = open("rust-articles-backup.json", encoding="utf8")
articles = pd.DataFrame.from_dict(json.load(file))
print("File Loaded")


for i in range(len(articles)):
    if i % 10000 == 0:
        print(i,len(articles))
    articles.at[i,'sentiment'] = getSentiment(articles.at[i,'body'])
    articles.at[i,'quoteBased'] = getQuoteBased(articles.at[i,'body'])
    articles.at[i,'sensationalized'] = getSensationalized(articles.at[i,'body'])
    articles.at[i,'mudslinging'] = getMudslinging(articles.at[i,'body'])
    articles.at[i,'spin'] = getSpin(articles.at[i,'body'])
    articles.at[i,'informal'] = getInformal(articles.at[i,'body'])
articles.to_json('rust-articles-backup-metrics.json')


print('Loading File')
file = open('rust-articles-backup-metrics.json', encoding='utf8')
articles = pd.DataFrame.from_dict(json.load(file))
print('File Loaded')


fullArticles = pd.DataFrame()
fullDominantMetrics = pd.DataFrame()
for publisher in list(set(articles['publisher'])):
    print(publisher)
    publisherArticles = articles[articles['publisher'] == publisher].reset_index(drop=True)
    metrics = []
    for i in range(len(publisherArticles)):
        sentiment = publisherArticles.iloc[i]['sentiment']
        quoteBased = publisherArticles.iloc[i]['quoteBased']
        sensationalized = publisherArticles.iloc[i]['sensationalized']
        mudslinging = publisherArticles.iloc[i]['mudslinging']
        spin = publisherArticles.iloc[i]['spin']
        informal = publisherArticles.iloc[i]['informal']
        metrics.append([sentiment, quoteBased, sensationalized, mudslinging, spin, informal])
    labels, dataPCA, dominantMetrics = NewsBiasClusteringFromMetrics(metrics)
    publisherArticles['publisherPC1'] = dataPCA['PC1']
    publisherArticles['PublisherPC2'] = dataPCA['PC2']
    publisherArticles['publisherLabel'] = labels
    fullArticles = pd.concat([fullArticles, publisherArticles])
    dominantMetrics['publisher'] = publisher
    fullDominantMetrics = pd.concat([fullDominantMetrics, dominantMetrics])
fullArticles = fullArticles.reset_index(drop=True)


metrics = []
for i in range(len(fullArticles)):
    sentiment = fullArticles.iloc[i]['sentiment']
    quoteBased = fullArticles.iloc[i]['quoteBased']
    sensationalized = fullArticles.iloc[i]['sensationalized']
    mudslinging = fullArticles.iloc[i]['mudslinging']
    spin = fullArticles.iloc[i]['spin']
    informal = fullArticles.iloc[i]['informal']
    metrics.append([sentiment, quoteBased, sensationalized, mudslinging, spin, informal])
labels, dataPCA, dominantMetrics = NewsBiasClusteringFromMetrics(metrics)
fullArticles['fullPC1'] = dataPCA['PC1']
fullArticles['fullPC2'] = dataPCA['PC2']
fullArticles['fullLabel'] = labels
dominantMetrics['publisher'] = 'all'
fullDominantMetrics = pd.concat([fullDominantMetrics,dominantMetrics])
fullDominantMetrics = fullDominantMetrics.reset_index(drop=True)

fullArticles.to_json('rust-articles-full.json')
fullDominantMetrics.to_json('rust-articles-full-dominant-metrics.json')