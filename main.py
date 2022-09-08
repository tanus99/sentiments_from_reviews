import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
<<<<<<< HEAD
from sklearn.feature_extraction.text import CountVectorizer
from multiprocessing import Process, Queue, Pool
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.tree import export_graphviz, plot_tree
from matplotlib import pyplot as plt
import seaborn as sns
from IPython.display import Image
import pydotplus
from six import StringIO
=======
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from multiprocessing import Process, Queue, Pool
from sklearn.model_selection import train_test_split, GridSearchCV
from classifiers import *

>>>>>>> 41798bf6dbfff1134243ce4b9a11ded304521245

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

<<<<<<< HEAD
=======

>>>>>>> 41798bf6dbfff1134243ce4b9a11ded304521245
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
<<<<<<< HEAD
=======


>>>>>>> 41798bf6dbfff1134243ce4b9a11ded304521245
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

<<<<<<< HEAD
=======


>>>>>>> 41798bf6dbfff1134243ce4b9a11ded304521245
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
<<<<<<< HEAD
    # we extract till 500 features
=======
    # we extract till 1500 features
>>>>>>> 41798bf6dbfff1134243ce4b9a11ded304521245
    # "max_features" is the attribute to
    # experiment with to get better results
    cv = CountVectorizer(max_features=1500)

    # X contains corpus transformed into an array
    global X
    X = cv.fit_transform(corpus).toarray()

    # y contains the targeted labels
    global y
    y = df.iloc[:, 2].values

<<<<<<< HEAD
multiprocesses_it()
bag_of_words()
print(len(corpus))
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=6, stratify=y)

# first classifier - RandomForest
def random_forest_classifier():

    # let's see different configurations for the RandomForestClassifier
    # using GridSearchCV
    model = RandomForestClassifier(criterion='entropy', random_state=6)

    params = [{'n_estimators': [500,1000,1500],'max_depth': [10,20,25],
               'min_samples_leaf':[10,20,25]}]

    grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy',
                               n_jobs=4, cv=5, verbose=3)

    grid_search.fit(X_train,y_train)

    # print the best configuration parameters and the model
    print(f'The best parameters for the model are {grid_search.best_params_}')
    print(f'The best score for the model is {grid_search.best_score_}')

    # extract the best model
    best_model = grid_search.best_estimator_

    y_preds = best_model.fit(X_train,y_train)

    # accuracy of the model (n. or % of corrected labeled data)
    print(f'Accuracy of the model {metrics.accuracy_score(y_test, y_preds)}')

    # roc_auc_score
    roc_auc = roc_auc_score(y_test, y_preds)
    print(f'ROC AUC : {roc_auc}')

    # print the confusion matrix to visualize correctly labeled data
    cm = confusion_matrix(y_test, y_preds)

    print('CONFUSION MATRIX\n\n', cm)

    print(f'True Positive: {cm[0, 0]}\n')

    print(f'True Negative: {cm[1, 1]}\n')

    print(f'False Positive: {cm[0, 1]}\n')

    print(f'False Negative: {cm[1, 0]}\n')

    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                             index=['Predict Positive:1', 'Predict Negative:0'])

    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    plt.show()

    # show the tree (the fifth just to see the general behavior)
    tree = best_model.estimators_[5]
    dot_data = StringIO()
    export_graphviz(tree, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True,
                    class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('reviews.png')
    Image(graph.create_png())


# second classifier - SVM
def SVM():
    svc = SVC(random_state=6)

    #we will discuss these different configuration
    params = [{'C':[0.001,0.01,0.1,1,10,100], 'kernel':['linear','poly', 'rbf'],
               'gamma':[0.001,0.01,0.1,1,10,100]}]

    #create the grid search object for the purpose
    grid_search = GridSearchCV(estimator=svc, param_grid=params, scoring='accuracy',
                               n_jobs=4, cv=5, verbose=3)

    grid_search.fit(X_train, y_train)

    # print the best configuration parameters and the model
    print(f'The best parameters for the model are {grid_search.best_params_}')
    print(f'The best score for the model is {grid_search.best_score_}')

    # extract the best model
    best_estimator = grid_search.best_estimator_

    #evaluate the accuracy score and the ROC AUC score for the best estimator
    y_preds = best_estimator.fit(X_train, y_train)

    print(f'The accuracy score for the best SVM classifier is {metrics.accuracy_score(y_test,y_preds)}')
    print(f'The ROC AUC score for the best SVM classifier is {roc_auc_score(y_test,y_preds)}')

    # extract the confusion matrix for the best estimator
    cm = confusion_matrix(y_test,y_preds)

    print('CONFUSION MATRIX\n\n', cm)

    print(f'True Positive: {cm[0, 0]}\n')

    print(f'True Negative: {cm[1, 1]}\n')

    print(f'False Positive: {cm[0, 1]}\n')

    print(f'False Negative: {cm[1, 0]}\n')

    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                             index=['Predict Positive:1', 'Predict Negative:0'])

    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    plt.show()



#random_forest_classifier()
SVM()
=======
def tf_idf():
    # object that will perform the tf-idf process
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


multiprocesses_it()
# bag_of_words()
tf_idf()
print(len(corpus))
print(X.shape)
print(y.shape)

# splitting the dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6, stratify=y)

# results with the bag of words features extraction
# random_forest_classifier(X_train, y_train, X_test, y_test)
# SVM(X_train, y_train, X_test, y_test)
# multinomial_NB(X_train, y_train, X_test, y_test)


# results with the TD-IDF features extraction
random_forest_classifier(X_train, y_train, X_test, y_test)
# SVM(X_train, y_train, X_test, y_test)
# multinomial_NB(X_train, y_train, X_test, y_test)

>>>>>>> 41798bf6dbfff1134243ce4b9a11ded304521245
