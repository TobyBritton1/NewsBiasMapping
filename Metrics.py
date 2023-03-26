from nltk.sentiment.vader import SentimentIntensityAnalyzer

def getSentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = (sia.polarity_scores(text)['compound'] + 1) / 2
    return sentiment

def getQuoteBased(text):
    text = text.replace('“','"').replace('”','"')
    quotes = text.count('"')
    length = len(text) + 1 # Laplace smoothing
    quoteBased = quotes / length
    quoteBased = quoteBased / 0.015
    if quoteBased > 1:
        quoteBased = 1
    return round(quoteBased,5), quotes, length

def getWordCountMetric(text, metricWords, scaling):
    text = text.lower()
    count = 0
    for word in metricWords:
        count += text.count(word)
    length = len(text) + 1 # Laplace smoothing
    metric = count / length
    metric = metric / scaling
    if metric > 1:
        metric = 1
    return round(metric,5)

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
    score = getWordCountMetric(text, sensationalizedLanguage, 0.0085)
    return score

def getMudslinging(text):
    mudslingingLanguage = [
        'liar',
        'cheat',
        'fraud',
        'phony',
        'hypocrite',
        'backstabber',
        'schemer',
        'connive',
        'manipulator',
        'deceive',
        'charlatan',
        'scoundrel',
        'crook',
        'swindle',
        'thief',
        'snake',
        'rat',
        'weasel',
        'slimeball',
        'grifter',
        'greedy',
        'selfish',
        'narcissist',
        'arrogant',
        'entitled',
        'egotistical',
        'maniacal',
        'vindictive',
        'ruthless',
        'heartless',
        'cruel',
        'malicious',
        'mean-spirited',
        'hateful',
        'spiteful',
        'toxic',
        'poisonous',
        'venomous',
        'disgusting',
        'vile',
        'repulsive',
        'abhorrent',
        'revolting',
        'obnoxious',
        'offensive',
        'insufferable',
        'insidious',
        'treacherous',
        'despicable',
        'deplorable'
    ]
    score = getWordCountMetric(text, mudslingingLanguage, 0.012)
    return score
