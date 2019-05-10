# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 03:59:44 2019

@author: shhan
"""
#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python



# 0. How much data to use for training vs testing? 
# 1. Parse the raw string text from .csv file into an array of tweets (will size be issue)
# 2. Call FeatureExtraction(tweets) to get 
#    - M, word_vectorizer, char_vectorizer
#    - M contains n_grams3, pos and feature array
# 3. Run cross validation to get desired model
# 4. Save model in file

import os
from argparse import ArgumentParser
import csv
import pickle
import pandas
import matplotlib.pyplot as plt
import numpy as np

from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from pygam import LogisticGAM
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier

from FeatureExtraction import *
            
raw_data_path = "./labeled_data.csv"

def main(model_path, input_data_path):
    
    # Create directory to save the learned model in given model_path
    if not os.path.exists(model_path):
        try:
            os.makedirs(model_path)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    # Load and format data to prepare for feature extraction
    tweets = pickle.load(open('./xtrain.p', 'rb'))
    labels = pickle.load(open('./ytrain.p', 'rb'))
    vinputs = pickle.load(open('./xvalid.p', 'rb'))
    vlabels = pickle.load(open('./yvalid.p', 'rb'))
    tinputs = pickle.load(open('./xtest.p', 'rb'))
    tlabels = pickle.load(open('./ytest.p', 'rb'))
    
    trvtweets = tweets + vinputs + tinputs
    trvlabels = labels + vlabels + tlabels
    
    print("Extracting Features ...")
    Mf, word_vectorizer, char_vectorizer, full_vectorizer, names = FeatureExtraction(trvtweets)
    
    # Train model
    print("Training Model ...")

    model = LogisticRegression(penalty='l2', class_weight='balanced', C=0.95, solver='liblinear', multi_class='ovr').fit(Mf, trvlabels)
    
    l_predict = model.predict(Mf)
    f1score = f1_score(trvlabels,l_predict, average=None)
    print(f1score)
    print(classification_report(trvlabels, l_predict))
    
    # Save trained model 
    trained_model_filename = model_path+"model.sav"
    joblib.dump(model, trained_model_filename)
    
    # Save word vectorizer
    word_vec_filename = model_path + "word_vectorizer.sav"
    joblib.dump(word_vectorizer, word_vec_filename)


    # Save char vectorizer
    char_vec_filename = model_path + "char_vectorizer.sav"
    joblib.dump(char_vectorizer, char_vec_filename)

    full_vec_filename = model_path + "full_vectorizer.sav"
    joblib.dump(full_vectorizer, full_vec_filename)
    
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model_path", type=str, default="./")
    parser.add_argument("--input_data_path", type=str, default="./")

    args = parser.parse_args()
    main(**vars(args))


