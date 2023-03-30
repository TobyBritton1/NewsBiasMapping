from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize

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

def getSpin(text):
    spinLanguage = [
        'maybe',
        'sort of',
        'kind of',
        'possibly',
        'apparently',
        'basically',
        'pretty',
        'seems like',
        'more or less',
        'roughly',
        'approximately',
        'virtually',
        'almost',
        'nearly',
        'largely',
        'generally',
        'mostly',
        'typically',
        'commonly',
        'frequently',
        'usually',
        'oftentimes',
        'often',
        'regularly',
        'habitually',
        'customarily',
        'traditionally',
        'normally',
        'ordinarily',
        'standardly',
        'consistently',
        'constantly',
        'continuously',
        'in a way',
        'might',
        'could',
        'supposedly',
        'apparently',
        'basically',
        'probably',
        'perhaps',
        'quite',
        'fairly',
        'several',
        'stuff',
        'plenty',
        'few',
        'many',
        'lots',
        'thing',
    ]
    score = getWordCountMetric(text, spinLanguage, 0.012)
    return score

def getInformal(text):
    personalPronouns = [
        'i',
        'me',
        'you',
        'we',
        'he',
        'she',
        'they',
        'him',
        'her',
        'them',
        'us',
        'myself',
        'his',
        'hers',
        'theirs'
    ]
    sentenceCount = 1 # Laplace smoothing
    wordCount = 1 # Laplace smoothing
    characterCount = 0
    personalPronounsCount = 0
    for sentence in sent_tokenize(text):
        tokenizedText = word_tokenize(sentence)
        sentenceCount += 1
        for i in range(len(tokenizedText)-1):
            wordCount += 1
            characterCount += len(tokenizedText[i])
            for word in personalPronouns:
                if tokenizedText[i].lower() == word:
                    personalPronounsCount += 1
    sentenceLength = (wordCount / sentenceCount) / 50
    if sentenceLength > 1:
        sentenceLength = 1
    wordLength = ((characterCount / wordCount) - 3) / 3
    if wordLength > 1:
        wordLength = 1
    elif wordLength < 0:
        wordLength = 0
    length = len(text) + 1 # Laplace smoothing
    personalPronouns = (personalPronounsCount / length) / 0.03
    if personalPronouns > 1:
        personalPronouns = 1
    score = (3*personalPronouns+2*sentenceLength+wordLength)/6
    return round(score,5)