import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from multiprocessing import Process, Queue, Pool
from sklearn.model_selection import train_test_split, GridSearchCV
from classifiers import *


# just import and read the data
df = pd.read_csv('./data/reviews.csv')
# print(df.describe())

# shuffling the data frame and removing the default index
df = df.sample(frac=1, random_state=6).reset_index(drop=True)

# selecting the relevant columns from the dataset
df = df[['Review Text', 'Rating', 'Recommended IND']]
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
    # we extract till 500 features
    # "max_features" is the attribute to
    # experiment with to get better results
    cv = CountVectorizer(max_features=1500)

    # X contains corpus transformed into an array
    global X
    X = cv.fit_transform(corpus).toarray()

    # y contains the targeted labels
    global y
    y = df.iloc[:, 2].values


multiprocesses_it()
bag_of_words()
print(len(corpus))
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6, stratify=y)

# results with the bag of words features extraction

# random_forest_classifier(X_train, y_train, X_test, y_test)
# SVM(X_train, y_train, X_test, y_test)
# multinomial_NB(X_train, y_train, X_test, y_test)


# results with the TD-IDF features extraction

