import pickle
import time

import keras.models
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
from preparing_methods import *

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

# importing the so called stopwords
nltk.download('stopwords')

# print(set(stopwords.words('english')))


# multiprocesses_it(df)
# X, y = bag_of_words(df)
# X, y = tf_idf(df)
print(len(corpus))

# split del dataset in train e test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6, stratify=y)

# risultati con il metodo di estrazione delle features bag of words
# random_forest_classifier(X_train, y_train, X_test, y_test)
# SVM(X_train, y_train, X_test, y_test)
# multinomial_NB(X_train, y_train, X_test, y_test)


# risultati con il metodo di estrazione delle features TD-IDF
# random_forest_classifier(X_train, y_train, X_test, y_test)
# SVM(X_train, y_train, X_test, y_test)
# model = multinomial_NB(X_train, y_train, X_test, y_test)

# risultati con il modello CNN
X = df['Review Text']
y = df['Recommended IND']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6, stratify=y)

model, X_test_seq = get_cnn_model(X_train,X_test,y_train,y_test)

# salva l'oggetto contenente il classificatore addestrato
def save_obj(model, filename):
    model.save(filename)


# carica l'oggetto contenente il classificatore addestrato
def read_obj(filename):
    model = keras.models.load_model(filename)
    return model


save_obj(model, 'CNN')
reconstructed_model = read_obj('CNN')
y_preds = (reconstructed_model.predict(X_test_seq) > 0.5).astype('int32')
print(f'The accuracy score is {accuracy_score(y_test, y_preds)}')
print(f'The ROC AUC score is {roc_auc_score(y_test, y_preds)}')
