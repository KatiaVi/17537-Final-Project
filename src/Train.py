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

from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel


from FeatureExtraction import *


def load_and_format_raw_data(data_path):
    with open(data_path) as csvfile:
        output = []
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            output.append(row[0])
            
    return output
            
    
    
def main(model_path, input_data_path, labels_data_path):
    
    # Create directory to save the learned model in given model_path
    if not os.path.exists(model_path):
        try:
            os.makedirs(model_path)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    # Load and format data to prepare for feature extraction
    tweets = load_and_format_raw_data(input_data_path)
    labels = load_and_format_raw_data(labels_data_path)
    
    print("Extracting Features ...")
    M, word_vectorizer, char_vectorizer = FeatureExtraction(tweets)
    
    
    # Obtain Most Relevant Features
  #  print("Finding Most Relevant Features ...")
  #  select = SelectFromModel(LogisticRegression(class_weight="balanced",penalty="l1", C=0.01))
  #  M_ = select.fit_transform(M,labels)
    
    # Train model
    print("Training Model ...")
    model = LogisticRegressionCV(cv=5, random_state=0, multi_class='ovr', max_iter = 100, C=0.01).fit(M, labels) # can change parameters here
    
    # Save trained model 
    trained_model_filename = model_path+"model.sav"
    joblib.dump(model, trained_model_filename)
    
    # Save word vectorizer
    word_vec_filename = model_path + "word_vectorizer.sav"
    joblib.dump(word_vectorizer, word_vec_filename)


    # Save char vectorizer
    char_vec_filename = model_path + "char_vectorizer.sav"
    joblib.dump(char_vectorizer, char_vec_filename)

    
    
    
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model_path", type=str, default="../model/")
    parser.add_argument("--input_data_path", type=str, default="../data/raw-data/train_input_data.csv")
    parser.add_argument("--labels_data_path", type=str, default="../data/raw-data/train_labels_data.csv")

    
    args = parser.parse_args()
    main(**vars(args))
