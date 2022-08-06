import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from multiprocessing import Process, Queue, Pool
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.tree import export_graphviz, plot_tree
from matplotlib import pyplot as plt
import seaborn as sns
from IPython.display import Image
import pydotplus
from six import StringIO

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


def multiprocesses_it():
    index = df.shape[0] // 4
    p = Pool(4)
    params = ([0, index], [index, 2 * index], [2 * index, 3 * index], [3 * index, 4 * index + 1])
    p.starmap(preprocessing, params)
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
    cv = CountVectorizer(max_features=500)

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

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=6, stratify=y)

def random_forest_classifier():
    model = RandomForestClassifier(n_estimators=501, criterion='entropy',
                                   n_jobs=4, max_depth=10, min_samples_leaf=10)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(y_pred)

    #print the confusion matrix to visualize correctly labeled data
    cm = confusion_matrix(y_test,y_pred)

    print('CONFUSION MATRIX\n\n', cm)

    print(f'True Positive: {cm[0, 0]}\n')

    print(f'True Negative: {cm[1, 1]}\n')

    print(f'False Positive: {cm[0, 1]}\n')

    print(f'False Negative: {cm[1, 0]}\n')

    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                                     index=['Predict Positive:1', 'Predict Negative:0'])

    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    plt.show()

    print(f'Accuracy of the model {model.score(X_train,y_train)}')
    print(f'Accuracy of the model {metrics.accuracy_score(y_test, y_pred)}')

    #show the tree (the fifth just to see the general behavior)
    tree = model.estimators_[5]
    dot_data = StringIO()
    export_graphviz(tree, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True,
                    class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('reviews.png')
    Image(graph.create_png())




random_forest_classifier()