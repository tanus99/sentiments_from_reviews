import pickle
import time

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

# preprocessing function to clean the data
# attributes for the parallel execution
# s -> start index
# e -> end index
# q -> queue for sharing data
corpus = []
q = Queue()
# cleaning the data for all the classifiers except the CNN
def preprocessing(df, s, e):
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
def multiprocesses_it(df):
    index = df.shape[0] // 4
    p = Pool(4)
    params = ([df,0, index], [df,index, 2 * index], [df,2 * index, 3 * index], [df,3 * index, 4 * index + 1])
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
        time.sleep(0.001)



# creating the Bag of Words model
def bag_of_words(df):
    # we extract till 1500 features
    # "max_features" is the attribute to
    # experiment with to get better results
    cv = CountVectorizer(max_features=1500)

    # X contains corpus transformed into an array
    X = cv.fit_transform(corpus).toarray()

    # y contains the targeted labels
    y = df.iloc[:, 1].values
    return X,y


# object that will perform the tf-idf process
def tf_idf(df):
    # tf = occurences of the term t in a document d / number of terms in d
    # idf = log(number of documents / number of documents containing the term t)
    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(corpus).toarray()

    y = df.iloc[:, 1].values

    print("n_samples: %d, n_features: %d" % X.shape)
    return X,y
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

# preparing the data for the CNN model
def preprocessing_CNN(X_train):

    # metto insieme tutte le parole in un unico testo, splitto
    # e conto il numero di parole uniche presenti
    max_words = len(set(" ".join(X_train).split()))

    # calcolo la lunghezza di ogni recensione e considero la massima
    max_len = X_train.apply(lambda x: len(x)).max()

    # based on the vocabulary it associates an index for each word/token
    tokenizer = Tokenizer(num_words=max_words)

    tokenizer.fit_on_texts(X_train)

    # then we return a list that assigns for each review the index of the words that it contains
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    # we pad to be sure that all the reviews have the same lenght
    X_train_seq = pad_sequences(X_train_seq, maxlen=max_len)



    return max_words,max_len,X_train_seq,tokenizer

