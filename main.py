from datetime import time

import pandas as pd
import numpy as np
import re
import nltk
from keras.optimizer_v2.adam import Adam
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from multiprocessing import Process, Queue, Pool
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from classifiers import *


# just import and read the data
df = pd.read_csv('./data/reviews.csv')
# print(df.describe())

# shuffling the data frame and removing the default index
df = df.sample(frac=1, random_state=6).reset_index(drop=True)

# selecting the relevant columns from the dataset
df = df[['Review Text', 'Recommended IND']]
# print(df.iloc[0:6])

# I see if there are some missing values
df.dropna(subset=['Review Text'], inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.iloc[0:6])
print(df.shape)


corpus = []

# importing the so called stopwords
nltk.download('stopwords')

# print(set(stopwords.words('english')))


# preprocessing function to clean the data
# attributes for the parallel execution
# s -> start index
# e -> end index
# q -> queue for sharing data
q = Queue()


# cleaning the data for all the classifiers except the CNN
def preprocessing(s, e):
    for i in range(s, e):
        # column Review Text, row ith
        review = re.sub('[^a-zA-Z]', ' ', df['Review Text'][i])

        # convert all cases to lower cases
        review = review.lower()

        # split to array
        review = review.split()

        # creating PorterStemmer object to take main stem
        # each word
        ps = PorterStemmer()

        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

        # join all string array elements
        # to create back into a string
        review = ' '.join(review)

        # append each string to create
        # array of clean text
        # corpus.append(review)

        # now we add the results in the queue
        q.put(review)



    print(f"From {s} to {e} has finished")
    print(q.qsize())


# parallel execution to reduce the time
def multiprocesses_it():
    index = df.shape[0] // 4
    p = Pool(4)
    params = ([0, index], [index, 2 * index], [2 * index, 3 * index], [3 * index, 4 * index + 1])
    p.starmap(preprocessing, params)
    p.close()
    # p1 = Process(target=preprocessing, args=(0, index,q))
    # p2 = Process(target=preprocessing, args=(index, 2 * index,q))
    # p3 = Process(target=preprocessing, args=(2 * index, 3 * index,q))
    # p4 = Process(target=preprocessing, args=(3 * index, 4 * index,q))
    # p5 = Process(target=preprocessing, args=(4 * index, 5 * index+1,q))
    #
    # # start child processes
    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()
    # p5.start()
    # # wait until child processes finish
    # p1.join()
    # p2.join()
    # p3.join()
    # p4.join()
    # p5.join()
    #
    while not q.empty():
        corpus.append(q.get())


# creating the Bag of Words model
def bag_of_words():
    # we extract till 1500 features
    # "max_features" is the attribute to
    # experiment with to get better results
    cv = CountVectorizer(max_features=1500)

    # X contains corpus transformed into an array
    global X
    X = cv.fit_transform(corpus).toarray()

    # y contains the targeted labels
    global y
    y = df.iloc[:, 2].values

# object that will perform the tf-idf process
def tf_idf():
    # tf = occurences of the term t in a document d / number of terms in d
    # idf = log(number of documents / number of documents containing the term t)
    vectorizer = TfidfVectorizer()
    global X
    X = vectorizer.fit_transform(corpus).toarray()

    global y
    y = df.iloc[:, 2].values

    print("n_samples: %d, n_features: %d" % X.shape)

    # # Select the first hundred documents from the data set
    # tf_idf = pd.DataFrame(X.todense()).iloc[:100]
    # tf_idf.columns = vectorizer.get_feature_names()
    # tfidf_matrix = tf_idf.T
    # tfidf_matrix.columns = ['response' + str(i) for i in range(1, 101)]
    # tfidf_matrix['count'] = tfidf_matrix.sum(axis=1)
    #
    # # Top 100 words
    # tfidf_matrix = tfidf_matrix.sort_values(by='count', ascending=False)[:100]
    #
    # # Print the first 100 words
    # print(tfidf_matrix.drop(columns=['count']).head(100))


# multiprocesses_it()
# bag_of_words()
# tf_idf()
print(len(corpus))
# print(X.shape)
# print(y.shape)

# splitting the dataset into train and test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6, stratify=y)

# results with the bag of words features extraction
# random_forest_classifier(X_train, y_train, X_test, y_test)
# SVM(X_train, y_train, X_test, y_test)
# multinomial_NB(X_train, y_train, X_test, y_test)


# results with the TD-IDF features extraction
# random_forest_classifier(X_train, y_train, X_test, y_test)
# SVM(X_train, y_train, X_test, y_test)
# multinomial_NB(X_train, y_train, X_test, y_test)

# preparing the data for the CNN model
def preprocessing_CNN():
    X = df['Review Text']
    y = df['Recommended IND']

    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6, stratify=y)

    global max_words
    max_words = len(set(" ".join(X_train).split()))
    global max_len
    max_len = X_train.apply(lambda x: len(x)).max()

    # based on the vocabulary it associates an index for each word/token
    tokenizer = Tokenizer(num_words=max_words)

    tokenizer.fit_on_texts(X_train)

    # then we return a list that assigns for each review the index of the words that it contains
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    #we pad to be sure that all the reviews have the same lenght
    X_train_seq = pad_sequences(X_train_seq,maxlen=max_len)


preprocessing_CNN()
model = get_cnn_model()

loss = 'categorical_crossentropy'
# loss = 'binary_crossentropy'
metrics = ['accuracy']

print("Starting...\n")

start_time = time.time()

print("\n\nCompliling Model ...\n")
learning_rate = 0.001
optimizer = Adam(learning_rate)
# optimizer = Adam()

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

verbose = 1
epochs = 100
batch_size = 128
validation_split = 0.2

print("Trainning Model ...\n")

model.fit(
    X_train_seq,
    Y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=verbose,
    callbacks=callbacks,
    validation_split=validation_split,
    class_weight =class_weight
)

elapsed_time = time.time() - start_time
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

print("\nElapsed Time: " + elapsed_time)
