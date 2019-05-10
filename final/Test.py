#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python

# 1. Parse the raw string text from .csv file into an array of tweets
# 2. Load trained model and vectorizers
# 3. Call FeatureExtractionTest(tweets, word_vectorizer, char_vectorizer) to get M
# 4. Input M and trained model to predict labels
# 5. Compare labels to ground truth

import os
from argparse import ArgumentParser
import csv
import pickle

import sklearn
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from FeatureExtraction import *


def load_and_format_raw_data(data_path):
    with open(data_path) as csvfile:
        output = []
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            output.append(row[0])
            
    return output
    
    
def main(model_path, input_data_path, labels_data_path):
    
    # Load trained model and vectorizers
    if not os.path.exists(model_path):
        print("No Trained Model Exists!!")
        return
    else:
        model = joblib.load(model_path+"modellr.sav")
        word_vectorizer = joblib.load(model_path + "word_vectorizer.sav")
        char_vectorizer = joblib.load(model_path + "char_vectorizer.sav")
        full_vectorizer = joblib.load(model_path + "full_vectorizer.sav")
    
    
    # Load and format data to prepare for feature extraction
    tweets = pickle.load(open('xtest.p', 'rb'))
    true_labels = pickle.load(open('ytest.p', 'rb'))
    M = FeatureExtractionTest(tweets, word_vectorizer, char_vectorizer, full_vectorizer)
    
    # Predict Labels
    predicted_labels = model.predict(M)
    
    # Compute FScore
    f1score = sklearn.metrics.f1_score(true_labels,predicted_labels, average=None)
    print(f1score)
    print(classification_report(true_labels, predicted_labels))
    
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model_path", type=str, default="./")
    parser.add_argument("--input_data_path", type=str, default="./test_input_data.csv")
    parser.add_argument("--labels_data_path", type=str, default="./test_labels_data.csv")

    
    args = parser.parse_args()
    main(**vars(args))
