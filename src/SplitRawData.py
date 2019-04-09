# 1. Split raw data into train and test in labeled_data.csv
# 2. Generates 4 new csv files:
#    - train_input_data.csv
#    - train_labels_data.csv
#    - test_input_data.csv
#    - test_labels_data.csv

import csv
from sklearn import model_selection


raw_data_path = "../data/raw-data/labeled_data.csv"

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
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(inputs, labels, random_state=42)
print(X_train[0])
print(Y_train[0])


# Write train_input_data.csv  
with open('../data/train_input_data.csv', mode='w') as train_input_file:
    train_input_writer = csv.writer(train_input_file, delimiter=',', quotechar='"')
    for train_input in X_train:
        train_input_writer.writerow([train_input])

train_input_file.close()
        
# Write train_labels_data.csv  
with open('../data/train_labels_data.csv', mode='w') as train_labels_file:
    train_labels_writer = csv.writer(train_labels_file, delimiter=',', quotechar='"')
    for train_labels in Y_train:
        train_labels_writer.writerow([train_labels])
 
train_labels_file.close()
       
        
# Write test_input_data.csv  
with open('../data/test_input_data.csv', mode='w') as test_input_file:
    test_input_writer = csv.writer(test_input_file, delimiter=',', quotechar='"')
    for test_input in X_test:
        test_input_writer.writerow([test_input])
        
test_input_file.close()

      
# Write test_labels_data.csv  
with open('../data/test_labels_data.csv', mode='w') as test_labels_file:
    test_labels_writer = csv.writer(test_labels_file, delimiter=',', quotechar='"')
    for test_labels in Y_test:
        test_labels_writer.writerow([test_labels])
    
test_labels_file.close()
    