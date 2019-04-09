# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:19:24 2019

@author: s-hana16
"""

"""
TASKS:
Lemmatize - DONE
POS - DONE
Cleaning for the URLs and stuff - DONE
Removing white space - DONE
Remove punctuation - DONE
Get capitalization count - DONE
Make words lowercase - DONE
Character n-grams - DONE
Sentiment Analyzer (VADER) - DONE
Sentiment + Subjectivity (TextBlob) - DONE
Word count - DONE

"""

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
import re

from nltk.sentiment.vader import SentimentIntensityAnalyzer as Senti
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer as WN
from nltk import word_tokenize
from nltk import bigrams, trigrams
from textblob import TextBlob

#nltk.download('stopwords')
#nltk.download('vader_lexicon')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')

stopwords = nltk.corpus.stopwords.words("english")
sentiment_analyzer = Senti()
lemmatizer = WN()

#Cleans the data - takes out extra whitespace, URLs, Mentions, Punctuation
#INPUT: an individual tweet
def clean(input_text):
    space_pattern = '\s+'
    url = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention = '@[\w\-]+'
    text = re.sub(space_pattern, ' ', input_text)
    text = re.sub(url, 'url', text)
    text = re.sub(mention, 'mention', text)
    text = re.sub("&amp", "and", text)
    remove = string.punctuation
    pattern = r"[{}]".format(remove)
    new_text = re.sub(pattern, "", text) 
    return new_text

#Returns lemmatized words that are lowercase
#INPUT: a processed tweet
def lem(input_text):
    words = input_text.lower()
    pos_tags = POS(words)
    lemm = Lemmatize(pos_tags)
    return lemm

#takes in processed tweet, returns number of allcaps words and number of words
def words_caps(input_text):
    raw_words = input_text.split()
    allcaps = sum(map(str.isupper, raw_words))
    return allcaps, len(raw_words)

#Converts the nltk tagger POS to ones accepted by WordNetLemmatizer
#INPUT: tag
def POSConverter(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

#Gets the POS
#INPUT: a processed tweet
def POS(input_text):
    text = word_tokenize(input_text)
    pos_tags = nltk.pos_tag(text)
    return pos_tags

#Gets the POS list and converts to string
#INPUT: the return of POS
def indiv_POS(input_pos):
    pos = []
    for x,y in input_pos:
        pos.append(y)
    pos_str = " ".join(pos)
    return pos_str

#Gets TF-IDF for the set of all tweets
#INPUT: all the tweets
def POS_tfidf(corpus):
    all_tags = []
    for s in corpus:
        s_c = clean(s)
        pos = indiv_POS(POS(s_c))
        all_tags.append(pos)
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(pd.Series(all_tags)).toarray()
    return tfidf
    
#Lemmatizes the words
#INPUT: from POS the list of (word, tag) tuples
def Lemmatize(input_text):
    lem_words = []
    for x,y in input_text:
        tag = POSConverter(y)
        if (tag != ''): 
            lem_words.append(lemmatizer.lemmatize(x,pos=tag))
        else:
            lem_words.append(x)
    lem_str = " ".join(lem_words)
    return lem_str

def Ngrams(input_text, n):
    n_grams = [input_text[i:i+n] for i in range(len(input_text)-n+1)]
    n_grams2 = " ".join(n_grams)
    return n_grams2

def Ngrams_Full(corpus, n):
    all_n = []
    for s in corpus:
        s_c = clean(s)
        s_lem = lem(s_c)
        n_grams = Ngrams(s_lem, n)
        all_n.append(n_grams)
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(pd.Series(all_n)).toarray()
    return tfidf

#Outputs compound value from VaderSentimentAnalysis
#INPUT: pre-processed string
def VaderSentiment(input_text):
    sentiment = sentiment_analyzer.polarity_scores(input_text)
    return sentiment['compound']

#INPUT: pre-processed string
def BlobSentiment(input_text):
    text = TextBlob(input_text)
    return text.sentiment.polarity, text.sentiment.subjectivity

def FeatureExtraction(tweets):
    pos = POS_tfidf(tweets)
    n_grams3 = Ngrams_Full(tweets, 3)
    #n_grams4 = Ngrams_Full(tweets, 4)
    #n_grams5 = Ngrams_Full(tweets, 5)
    feature_arr = []
    for t in tweets:
        t_c = clean(t)
        lemmy = lem(t_c)
        sentiment = VaderSentiment(t_c)
        pol, subj = BlobSentiment(t_c)
        caps, words = words_caps(t_c)
        features = [pol, subj, sentiment, caps, words]
        feature_arr.append(features)
    print(feature_arr)
    M = np.concatenate((n_grams3, pos, np.array(feature_arr)), axis=1)
    return M
    
    
    





