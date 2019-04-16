# 1. Split raw data into train and test in labeled_data.csv
# 2. Generates 4 new csv files:
#    - train_input_data.csv
#    - train_labels_data.csv
#    - test_input_data.csv
#    - test_labels_data.csv

import csv
import pickle
from sklearn import model_selection


raw_data_path = "./labeled_data.csv"

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
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(inputs, labels, test_size=0.1)   
tinputs, vinputs, tlabels, vlabels = model_selection.train_test_split(X_train, Y_train, test_size=0.3)   

pickle.dump(tinputs, open( "xtrain.p", "wb" ))
pickle.dump(X_test, open( "xtest.p", "wb" ))
pickle.dump(vinputs, open( "xvalid.p", "wb" ))
pickle.dump(tlabels, open( "ytrain.p", "wb" ))
pickle.dump(Y_test, open( "ytest.p", "wb" ))
pickle.dump(vlabels, open( "yvalid.p", "wb" ))

# Write train_input_data.csv  
"""with open('./train_input_data.csv', mode='w') as train_input_file:
    train_input_writer = csv.writer(train_input_file, delimiter=',', quotechar='"')
    for train_input in tinputs:
        train_input_writer.writerow([train_input])

train_input_file.close()
        
# Write train_labels_data.csv  
with open('./train_labels_data.csv', mode='w') as train_labels_file:
    train_labels_writer = csv.writer(train_labels_file, delimiter=',', quotechar='"')
    for train_labels in tlabels:
        train_labels_writer.writerow([train_labels])
 
train_labels_file.close()
       
        
# Write test_input_data.csv  
with open('./test_input_data.csv', mode='w') as test_input_file:
    test_input_writer = csv.writer(test_input_file, delimiter=',', quotechar='"')
    for test_input in X_test:
        test_input_writer.writerow([test_input])
        
test_input_file.close()

      
# Write test_labels_data.csv  
with open('./test_labels_data.csv', mode='w') as test_labels_file:
    test_labels_writer = csv.writer(test_labels_file, delimiter=',', quotechar='"')
    for test_labels in Y_test:
        test_labels_writer.writerow([test_labels])
    
test_labels_file.close()

with open('./valid_input_data.csv', mode='w') as valid_input_file:
    valid_input_writer = csv.writer(valid_input_file, delimiter=',', quotechar='"')
    for valid_input in vinputs:
        valid_input_writer.writerow([valid_input])
        
valid_input_file.close()

with open('./valid_labels_data.csv', mode='w') as valid_labels_file:
    valid_labels_writer = csv.writer(valid_labels_file, delimiter=',', quotechar='"')
    for valid_labels in vlabels:
        valid_labels_writer.writerow([valid_labels])
    
valid_labels_file.close()"""
    