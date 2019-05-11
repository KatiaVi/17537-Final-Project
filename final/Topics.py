# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:00:49 2019

@author: s-hana16
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn import decomposition
from sklearn.preprocessing import normalize
import nltk
import string
import re
import xlrd
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

from nltk.sentiment.vader import SentimentIntensityAnalyzer as Senti
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer as WN
from nltk import word_tokenize
from nltk import bigrams, trigrams
from nltk.util import ngrams
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem.snowball import SnowballStemmer

lemmatizer = WN()
p_stemmer =  SnowballStemmer("english")
 

en_stops = stopwords.words('english')
en_stops = x = [''.join(c for c in word if c not in string.punctuation) for word in en_stops]
en_stops.extend(['url', 'mention', 'lol', 'rt', 'like', 'im', 'dont', 'u', 
                 '8220mention', 'get', 'say', 'go','…','"','"'])       

twitter_topic1_data_path = "../data/raw-data/w_senti_labels.csv"
twitter_topic2_data_path = "../data/raw-data/mt_senti_labels.csv"
twitter_topic3_data_path = "../data/raw-data/blm_senti_labels.csv"
model_path = "../model/"

map_words = {}

def load_and_format_raw_data(data_path):
    with open(data_path) as csvfile:
        output = []
        readCSV = csv.reader(csvfile, delimiter=',')
        next(readCSV)
        for row in readCSV:
            if (int(row[2]) == 0):
                output.append(row[1])
            
    csvfile.close()
    return output


def clean(input_text):
    space_pattern = '\s+'
    url = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention = '@[\w\-]+'
    text = re.sub(space_pattern, ' ', input_text)
    text = re.sub(url, 'url', text)
    text = re.sub(mention, ' ', text)
    text = re.sub("&amp", "and", text)
    text = ''.join(i for i in text if not i.isdigit()) # remove any numbers
    remove = string.punctuation
    pattern = r"[{}]".format(remove)
    new_text = re.sub(pattern, "", text) 
    final_text = re.sub(space_pattern, ' ', new_text)
    return final_text


def lem(input_text):
    words = input_text.lower()
    pos_tags = POS(words)
#    pos_tags_final = []
#    for x,y in pos_tags:
#        if y.startswith('N') or y.startswith('J'):
#            pos_tags_final.append((x,y))
    lemm = Lemmatize(pos_tags)
    print(len(map_words))
    return lemm

def tokenize(input_text):
    words = input_text.split()
    new_words = [word for word in words if word not in en_stops]
    stemmed_tokens = [p_stemmer.stem(word) for word in new_words]

    return stemmed_tokens
            
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

#Lemmatizes the words
#INPUT: from POS the list of (word, tag) tuples
def Lemmatize(input_text):
    lem_words = []
    for x,y in input_text:
        tag = POSConverter(y)
        if (tag != ''): 
            l_word = lemmatizer.lemmatize(x,pos=tag)
            lem_words.append(l_word)
            if l_word not in map_words:
                map_words[l_word] = x
        else:
            lem_words.append(x)
            if x not in map_words:
                map_words[x] = x

    lem_str = " ".join(lem_words)
    return lem_str

def get_nmf_topics(model, n_topics, n_top_words):
    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
    feat_names = vectorizer.get_feature_names()
    word_dict = {};
    print(map_words)
    
    for i in range(n_topics):
        #for each topic, obtain the largest values, and add the words they map to into the dictionary.
        words_ids = model.components_[i].argsort()[:-n_top_words - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
    
    return pd.DataFrame(word_dict);

def get_lda_topics(model, num_topics, num_words):
    word_dict = {};
    print(map_words)
    for i in range(num_topics):
        words = model.show_topic(i, topn = num_words);
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words];
    return pd.DataFrame(word_dict);


df1 = pd.read_csv(twitter_topic1_data_path)
filtered1 = df1.loc[df1['Labels'] == 2]

df2 = pd.read_csv(twitter_topic2_data_path)
filtered2 = df2.loc[df2['Labels'] == 2]

df3 = pd.read_csv(twitter_topic3_data_path)
filtered3 = df3.loc[df3['Labels'] == 2]

filtered = pd.concat([filtered1, filtered2, filtered3], ignore_index=True)


hate_speech = filtered['Tweets']

cleaned = hate_speech.apply(clean)
#lemmatized = cleaned.apply(lem)
tokens = cleaned.apply(tokenize)
bigram = gensim.models.Phrases(tokens, min_count=5, threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
bigrams = [bigram_mod[doc] for doc in tokens]

dictionary = corpora.Dictionary(bigrams)
dictionary.filter_extremes(no_below=0.02)
corpus = [dictionary.doc2bow(text) for text in bigrams]
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

LNUM_TOPICS = 10
NUM_TOPICS = 10
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = LNUM_TOPICS, 
                                           id2word=dictionary, alpha='auto', 
                                           eta='auto')
ldamodel.save('model1.gensim')
print(ldamodel.print_topics())

cm = CoherenceModel(model=ldamodel, corpus=corpus, coherence='u_mass')
coherence = cm.get_coherence()  # get coherence value
topic_coherence = cm.get_coherence_per_topic()
print(topic_coherence)
print(coherence)

tweets_nostops = [' '.join(text) for text in tokens]
vectorizer = TfidfVectorizer(analyzer='word', max_features=5000, max_df = 0.5, stop_words="english");
counts = vectorizer.fit_transform(tweets_nostops);
normalized = normalize(counts, norm='l1', axis=1)

print("Get normalized stuff")

modelNMF = NMF(n_components=NUM_TOPICS, alpha=0.01, init='nndsvd');
modelNMF.fit(normalized)

nmf_topics = get_nmf_topics(modelNMF, NUM_TOPICS, 10)
lda_topics = get_lda_topics(ldamodel, LNUM_TOPICS, 10)
nmf_topics.to_csv('NMF_Twitter_Hate.csv')
lda_topics.to_csv('LDA_Twitter_Hate.csv')

"""[-8.736830709362398, -8.725395351426911, -7.866255491817352, -8.151695862259201, -5.944211978322235, -5.910124305032073, -7.948702018295331, -9.438366364877844]
-7.840197760174169
"""




    




