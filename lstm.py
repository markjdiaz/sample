import os, csv
import sklearn
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from KaggleWord2VecUtility import KaggleWord2VecUtility
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.layers.embeddings import Embedding
import pandas as pd
import numpy as np
import time
import matplotlib as plt

train_file = 'data/train_set.csv' 
test_file = 'data/test_set.csv'
output_file = 'nb_predictions.csv'

POST_INDEX = 1
TEST_SET_SIZE = 200
TEST_CLASS_INDEX = 3
TRAIN_CLASS_INDEX = 2

if __name__ == '__main__':
    def run_network(model=None, data=None):
        global_start_time = time.time()
        epochs = 1
        ratio = 0.5
        sequence_length = 50
       
        X_train, y_train, X_test, y_test = data

        try:
            model.fit(
                X_train, y_train,
                batch_size=200, nb_epoch=10, validation_split=0.10)
            predicted = model.predict(X_test)
            predicted = np.reshape(predicted, (predicted.size,))
        except KeyboardInterrupt:
            print 'Training duration (s) : ', time.time() - global_start_time
            return model, y_test, 0

        try:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(y_test[:100])
            plt.plot(predicted[:100])
            plt.show()
        except Exception as e:
            print str(e)
        print 'Training duration (s) : ', time.time() - global_start_time
    
        return model, y_test, predicted


    def build_model(train):
        model = Sequential()
        model.add(Embedding(100000, 128, dropout=0.2))
        model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
        model.add(Dense(1))
        model.add(Activation('softmax'))

        # try using different optimizers and different optimizer configs
        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        return model

    def vectorize(text):
        #vectorizes a list of strings (sentences)
        tknzr = Tokenizer(lower=True, split=" ")
        tknzr.fit_on_texts(text)

        result = tknzr.texts_to_sequences(text)
        max_seq = len(max(result, key=len))
        return result, max_seq

    #build and train LSTM model
    def train_tensor_lstm(x_ids, train, train_y, y_ids, test, test_y):
        train, max_train_len = vectorize(train)
        test, max_test_len = vectorize(test)

        max_len = max_train_len
        if max_train_len < max_test_len: max_len = max_test_len

        train = sequence.pad_sequences(train, maxlen=max_len)
        test = sequence.pad_sequences(test, maxlen=max_len)

        train_y = np.array(train_y)
        train_y = np.reshape(train_y, train_y.shape + (1,))

        results = run_network(model=None, data=(train, train_y, test, test_y))

        return results
        

    # Read the csv data into a list of strings.
    def read_data(filename):
      with open(filename) as f:
        r = csv.reader(f, dialect='excel')
        prev_id = None
        post_text = []
        text_class = []
        ids = []
        r.next()
        for i, row in enumerate(r):
            ids.append(row[0])
            text = row[POST_INDEX]
            if filename == 'data/test_set.csv':
                if i == TEST_SET_SIZE-1:
                    ids.pop()
                    break
                text_class.append(int(row[TEST_CLASS_INDEX]))
                post_text.append(text)
            else:
                text_class.append(row[TRAIN_CLASS_INDEX])
                post_text.append(text)
        return ids, post_text, text_class

    #Start
    x_ids, train, train_y = read_data(train_file)
    y_ids, test, test_y = read_data(test_file)

    post_dict = {k: v for k, v in zip(y_ids, test)}


    # Initialize an empty list to hold the clean posts
    clean_train_posts = []
    print "Cleaning and parsing the training set...\n"
    for post in train:
        clean_train_posts.append( " ".join( KaggleWord2VecUtility.review_to_wordlist( post )))


    # Create an empty list and append the clean posts one by one
    clean_test_reviews = []
    print "Cleaning and parsing the test set movie reviews...\n"
    clean_test_posts = []
    for post in test:
        clean_test_posts.append( " ".join( KaggleWord2VecUtility.review_to_wordlist( post )))

    #Train the LSTM and predict test set labels
    print "Predicting test labels...\n"
    result = train_tensor_lstm(x_ids, train, train_y, y_ids, test, test_y)
    output = pd.DataFrame( data={"id":y_ids, "y_actual":test_y, "y_predicted":result} )

    # Use pandas to write the comma-separated output file
    output.to_csv(os.path.join(os.path.dirname(__file__), 'data', output_file), index=False, quoting=3)
    print "Wrote results to " + output_file
   
    baseline_score = 0
    score = 0
    #Create a .csv file with false positive predictions for qualitative analysis
    with open("data/" + output_file) as f:
        file = open("data/nb_falsepos_lstm.csv", 'wb')
        wr = csv.writer(file)
        wr.writerow(["id", "actual", "predicted", "post"])
        r = csv.reader(f)
        r.next()
        for i, p in enumerate(r):
            if test_y[i] == str(0): baseline_score += 1
            if p[2] == test_y[i]: score += 1
            else:
                post = post_dict[p[0]]
                wr.writerow([p[0], p[1], p[2], post])

    #print R-Zero (baseline) accuracy and LSTM model accuracy
    print float(score)/len(test_y)  
    print float(baseline_score)/len(test_y)    