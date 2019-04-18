# 1. Split raw data into train and test in labeled_data.csv
# 2. Generates 4 new csv files:
#    - train_input_data.csv
#    - train_labels_data.csv
#    - test_input_data.csv
#    - test_labels_data.csv

import csv
import pickle
from sklearn import model_selection


raw_data_path = "../data/labeled_data.csv"

# Read Raw Data
with open(raw_data_path) as csvfile:
    inputs = []
    labels = []
    readCSV = csv.reader(csvfile, delimiter=',')
    # skip header row
    next(readCSV)
    for row in readCSV:
        inputs.append(row[6])
        labels.append(row[5])
  
csvfile.close()

# Split raw data into train and test
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(inputs, labels, test_size=0.2)   
tinputs, vinputs, tlabels, vlabels = model_selection.train_test_split(X_train, Y_train, test_size=0.2)   

pickle.dump(tinputs, open( "../data/xtrain.p", "wb" ))
pickle.dump(X_test, open( "../data/xtest.p", "wb" ))
pickle.dump(vinputs, open( "../data/xvalid.p", "wb" ))
pickle.dump(tlabels, open( "../data/ytrain.p", "wb" ))
pickle.dump(Y_test, open( "../data/ytest.p", "wb" ))
pickle.dump(vlabels, open( "../data/yvalid.p", "wb" ))
    