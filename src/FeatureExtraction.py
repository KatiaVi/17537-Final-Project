# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:19:24 2019

@author: s-hana16
"""

"""
HOW TO USE: 
    To extract features for training data:
        -call FeatureExtraction(tweets)
        -output is M, the features, a TFIDF vectorizer for POS, 
            and a TFIDF vectorizer for n-grams.
        -use M to train the model, the other two are needed for testing
    To extract features for testing data:
        -call FeatureExtraction(tweets, pos vectorizer, ngram vectorizer)
        -output is M, the features
        -use M to test the model
"""

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

#Gets TF-IDF for POS (training data)
#INPUT: all the tweets
def POS_tfidf(corpus):
    all_tags = []
    for s in corpus:
        s_c = clean(s)
        pos = indiv_POS(POS(s_c))
        all_tags.append(pos)
    word_vectorizer = TfidfVectorizer(
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words='english',
            ngram_range=(1, 1),
            max_features=10000)
    word_vectorizer.fit(pd.Series(all_tags))
    tfidf = word_vectorizer.transform(pd.Series(all_tags)).toarray()
    pos_vocab = {v:i for i, v in enumerate(word_vectorizer.get_feature_names())}
    return tfidf, word_vectorizer, pos_vocab
    
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

#Gets the Ngrams TF-IDF for training data
#Input: All tweets
def Ngrams_Full(corpus):
    all_n = []
    for s in corpus:
        s_c = clean(s)
        s_lem = lem(s_c)
        all_n.append(s_lem)
    char_vectorizer = TfidfVectorizer(
            analyzer='char',
            stop_words='english',
            ngram_range=(3, 5),
            max_features=5000,
            max_df = 0.5)
    char_vectorizer.fit(pd.Series(all_n))
    tfidf = char_vectorizer.transform(pd.Series(all_n)).toarray()
    vocab = {v:i for i,v in enumerate(char_vectorizer.get_feature_names())}
    return tfidf, char_vectorizer, vocab

def Ngrams_Words(corpus):
    all_n = []
    for s in corpus:
        s_c = clean(s)
        s_lem = lem(s_c)
        all_n.append(s_lem)
    full_vectorizer = TfidfVectorizer(
            analyzer='char',
            stop_words='english',
            ngram_range=(1, 3),
            max_features=5000,
            max_df = 0.5)
    full_vectorizer.fit(pd.Series(all_n))
    tfidf = full_vectorizer.transform(pd.Series(all_n)).toarray()
    return tfidf, full_vectorizer

#Outputs compound value from VaderSentimentAnalysis
#INPUT: pre-processed string
def VaderSentiment(input_text):
    sentiment = sentiment_analyzer.polarity_scores(input_text)
    return sentiment['compound']

#Outputs polarity and subjectivity from TextBlob sentiment analysis
#INPUT: pre-processed string
def BlobSentiment(input_text):
    text = TextBlob(input_text)
    return text.sentiment.polarity, text.sentiment.subjectivity

#Extracts all the features (main function to call on training data)
def FeatureExtraction(tweets):
    pos, word_vectorizer, pos_names = POS_tfidf(tweets)
    n_grams3, char_vectorizer, n_names = Ngrams_Full(tweets)
    n_gramsw, full_vectorizer = Ngrams_Words(tweets)
    pos_keys = ['']*len(pos_names)
    n_keys = ['']*len(n_names)
    for k,v in pos_names.items():
        pos_keys[v] = k
    for k,v in n_names.items():
        n_keys[v] = k
    feature_arr = []
    for t in tweets:
        t_c = clean(t)
        sentiment = VaderSentiment(t_c)
        pol, subj = BlobSentiment(t_c)
        caps, words = words_caps(t_c)
        features = [pol, subj, sentiment, caps, words]
        feature_arr.append(features)
    M = np.concatenate((n_grams3, n_gramsw, pos, np.array(feature_arr)), axis=1)
    feature_names = ["polarity", "subjectivity", "sentiment", "caps", "words"]
    final_features = pos_keys + n_keys + feature_names
    return M, word_vectorizer, char_vectorizer, full_vectorizer, final_features

#Gets the Ngrams TF-IDF for testing data
#Input: all tweets, char_vectorizer from the training FeatureExtraction output
def Ngrams_Test(corpus, char_vectorizer):
    all_n = []
    for s in corpus:
        s_c = clean(s)
        s_lem = lem(s_c)
        all_n.append(s_lem)
    tfidf = char_vectorizer.transform(pd.Series(all_n)).toarray()
    return tfidf

def Ngrams_Twords(corpus, full_vectorizer):
    all_n = []
    for s in corpus:
        s_c = clean(s)
        s_lem = lem(s_c)
        all_n.append(s_lem)
    tfidf = full_vectorizer.transform(pd.Series(all_n)).toarray()
    return tfidf

#Gets the POS TF-IDF for testing data
#Input: all tweets, word_vectorizer from the training FeatureExtraction output
def Pos_Test(corpus, word_vectorizer):
    all_tags = []
    for s in corpus:
        s_c = clean(s)
        pos = indiv_POS(POS(s_c))
        all_tags.append(pos)
    tfidf = word_vectorizer.transform(pd.Series(all_tags)).toarray()
    return tfidf

#Main function to call for testing data feature extraction
#Input: testing tweets, word_vectorizer + char_vectorizer from training output
def FeatureExtractionTest(tweets, word_vectorizer, char_vectorizer, full_vectorizer):
    pos = Pos_Test(tweets, word_vectorizer)
    n_grams3 = Ngrams_Test(tweets, char_vectorizer)
    n_gramsw = Ngrams_Twords(tweets, full_vectorizer)
    feature_arr = []
    for t in tweets:
        t_c = clean(t)
        sentiment = VaderSentiment(t_c)
        pol, subj = BlobSentiment(t_c)
        caps, words = words_caps(t_c)
        features = [pol, subj, sentiment, caps, words]
        feature_arr.append(features)
    M = np.concatenate((n_grams3, n_gramsw, pos, np.array(feature_arr)), axis=1)
    return M



    
    





