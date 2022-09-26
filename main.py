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
from utils import *

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

# --ALERT!-- decommentare il metodo multiprocesses_it solo se si vuole utilizzare il metodo di estrazione
# bag of words o tf-idf.

# --ALERT!-- decommentare un solo metodo di estrazione delle features per volta, o bag of words o tf_idf
# in quanto X,y corrispondenti verranno poi usati per effettuare lo splitting

# multiprocesses_it(df)
# X, y = bag_of_words(df)
# X, y = tf_idf(df)
# print(len(corpus))
# split del dataset in train e test set con i metodi di estrazione delle features bag of words o tf-idf
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6, stratify=y)

# risultati con il metodo di estrazione delle features bag of words o TF-IDF
# --ALERT!-- decommentare il modello per addestrarlo accompagnato dal metodo save_obj per salvare il modello
# Una volta memorizzato in locale, commentare il modello e il metodo save_obj e decommentare 'reconstructed_model'
# --ALERT!-- ripetere quanto descritto sopra per ogni classificatore

# start_time = time.time()
# model = random_forest_classifier(X_train, y_train)
# save_obj(model,'random_forest_bow')
# reconstructed_model = read_obj('random_forest_bow')
# save_obj(model,'random_forest_tfidf')
# reconstructed_model = read_obj('random_forest_tfidf')

# model =  SVM(X_train, y_train)

# save_obj(model,'svm_bow')
# reconstructed_model = read_obj('svm_bow')
# save_obj(model,'svm_tfidf')
# reconstructed_model = read_obj('svm_tfidf')

# model = multinomial_NB(X_train, y_train)
# elapsed_time = time.time() - start_time
# print(f"\nElapsed Time: {elapsed_time} sec")
# save_obj(model,'naive_bayes_bow')
# reconstructed_model = read_obj('naive_bayes_bow')
# save_obj(model,'naive_bayes_tfidf')
# reconstructed_model = read_obj('naive_bayes_tfidf')

# --ALERT!-- una volta letto il modello puoi valutare le performance
# print_performance_metrics(model,X_test,y_test)
# print_confusion_matrix(model,X_test,y_test, 'naive_bayes_tfidf')


# risultati con il modello CNN
X = df['Review Text']
y = df['Recommended IND']

# decommentare solo se visualizzare i risultati della rete CNN poichè l'input dello splitting è
# diverso dai casi precedenti
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6, stratify=y)

# --ALERT!-- decommentare il modello per addestrarlo accompagnato dal metodo save_obj per salvare il modello
# Una volta memorizzato in locale, commentare il modello e il metodo save_obj e decommentare 'reconstructed_model'

model, tokenizer = get_cnn_model(X_train,y_train)
save_obj(model, 'cnn')
save_obj(tokenizer, 'tokenizer')
# reconstructed_model = read_obj('cnn')
# tokenizer = read_obj('tokenizer')
_, max_len, _, _ = preprocessing_CNN(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_seq = pad_sequences(X_test_seq, maxlen=max_len)

print_performance_metrics(model,X_test_seq,y_test)
print_confusion_matrix(model, X_test_seq, y_test, 'cnn')