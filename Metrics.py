from nltk.sentiment.vader import SentimentIntensityAnalyzer

def getSentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

def getFactBased(text):
    text = text.replace('“','"').replace('”','"')
    quotes = text.count('"')
    length = len(text.split(' '))
    factBased = round(quotes/length,5)
    return factBased