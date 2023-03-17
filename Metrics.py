from nltk.sentiment.vader import SentimentIntensityAnalyzer

def getSentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

def getFactBased(text):
    text = text.replace('“','"').replace('”','"')
    quotes = text.count('"')
    length = len(text) + 1 # Laplace smoothing
    factBased = quotes / length
    factBased = (factBased / 0.0075) - 1
    if factBased > 1:
        factBased = 1
    elif factBased < -1:
        factBased = -1
    return round(factBased,5), quotes, length