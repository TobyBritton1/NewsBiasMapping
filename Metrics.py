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

def getSensationalized(text):
    sensationalizedLanguage = [
        'shock',
        'terrify',
        'explosive',
        'epic',
        'mind-blowing',
        'unbelievable',
        'stunning',
        'scandalous',
        'sensational',
        'incredible',
        'jaw-dropping',
        'unprecedented',
        'apocalypse',
        'catastrophic',
        'devastating',
        'heartbreaking',
        'horrific',
        'intense',
        'massive',
        'monstrous',
        'phenomenal',
        'record-breaking',
        'shock-and-awe',
        'spectacular',
        'traumatic',
        'ultimate',
        'unimaginable',
        'apocalyptic',
        'bombshell',
        'brutal',
        'chaotic',
        'deadly',
        'doom',
        'eerie',
        'frenzy',
        'gripping',
        'insane',
        'killer',
        'lethal',
        'meltdown',
        'nightmarish',
        'pandemonium',
        'rampage',
        'savage',
        'tragic',
        'unhinged',
        'vicious',
        'widespread',
        'x-treme',
        'zany'
    ]
    text = text.lower()
    sensationalizedCount = 0
    for word in sensationalizedLanguage:
        sensationalizedCount += text.count(word)
    return sensationalizedCount