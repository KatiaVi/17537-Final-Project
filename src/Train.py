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

from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel


from FeatureExtraction import *

def load_and_format_raw_data(data_path):
    with open(data_path) as csvfile:
        output = []
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            output.append(row[0])
            
    return output
            
raw_data_path = "./labeled_data.csv"

def main(model_path, input_data_path, labels_data_path, vinput_data_path, vlabels_data_path, saved_features_path):
    
    # Create directory to save the learned model in given model_path
    if not os.path.exists(model_path):
        try:
            os.makedirs(model_path)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    # Load and format data to prepare for feature extraction
    tweets = pickle.load(open('../data/xtrain.p', 'rb'))
    labels = pickle.load(open('../data/ytrain.p', 'rb'))
    vinputs = pickle.load(open('../data/xvalid.p', 'rb'))
    vlabels = pickle.load(open('../data/yvalid.p', 'rb'))
    
    print("Extracting Features ...")
    M, word_vectorizer, char_vectorizer, full_vectorizer, names = FeatureExtraction(tweets)
    Mp = FeatureExtractionTest(vinputs, word_vectorizer, char_vectorizer, full_vectorizer)
    
    # Train model
    print("Training Model ...")

    model1 = LogisticRegression(penalty='l2', class_weight='balanced', C=0.95, solver='liblinear', multi_class='ovr')
    model2 = RandomForestClassifier(n_estimators=100, class_weight={0:0.85, 1:0.05, 2:0.10})
    model = VotingClassifier(estimators=[('lr', model1), ('rf', model2)], voting='hard', weights=[1,1]).fit(M, labels)
    
    l_predict = model.predict(M)
    f1score = f1_score(labels,l_predict, average=None)
    print(f1score)
    print(classification_report(labels, l_predict))
    
    predicted_labels = model.predict(Mp)
    
    # Compute FScore
    f1score = f1_score(vlabels,predicted_labels, average=None)
    print(f1score)
    print(classification_report(vlabels, predicted_labels))
    
    # Save trained model 
    trained_model_filename = model_path+"model.sav"
    joblib.dump(model, trained_model_filename)

    
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model_path", type=str, default="../model/")
    parser.add_argument("--input_data_path", type=str, default="./train_input_data.csv")
    parser.add_argument("--labels_data_path", type=str, default="./train_labels_data.csv")
    parser.add_argument("--vinput_data_path", type=str, default="./valid_input_data.csv")
    parser.add_argument("--vlabels_data_path", type=str, default="./valid_labels_data.csv")
    parser.add_argument("--saved_features_path", type=str, default="./extracted_features.p")
    
    args = parser.parse_args()
    main(**vars(args))
