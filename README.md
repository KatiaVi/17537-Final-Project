# 17537-Final-Project

The raw data in data/raw-data/labeled_data.csv is split up into train and test:
    - train_input_data.csv - the raw tweet strings for training
    - train_labels_data.csv - the classes of the raw tweet strings for training
    - test_input_data.csv - the raw tweet strings for testing
    - test_labels_data.csv - the classes of the raw tweet strings for testing

To train the model run:

python Train.py Train.py --input_data_path=<path to input data>[defaults to ../data/train_input_data.csv ] --labels_data_path=<path to input data>[defaults to ../data/train_labels_data.csv ] --model_path= <path to model>[defaults to ../model]

Model saved to <model_path>/model.sav
Word vectorizer saved to <model_path>/word_vectorizer.sav
Character vectorizer saved to <model_path>/char_vectorizer.sav

To test the model:

python Test.py --input_data_path=<path to input data>[defaults to ../data/test_input_data.csv ] --labels_data_path=<path to input data>[defaults to ../data/test_labels_data.csv ] --model_path= <path to model>[defaults to ../model]


