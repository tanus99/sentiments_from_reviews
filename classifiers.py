import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.tree import export_graphviz, plot_tree
from sklearn.naive_bayes import MultinomialNB
from keras.optimizer_v2.adam import Adam
from matplotlib import pyplot as plt
import seaborn as sns
from IPython.display import Image
import pydotplus
from six import StringIO
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dropout, BatchNormalization, Dense
from keras.models import Sequential

# first classifier - RandomForest
from preparing_methods import preprocessing_CNN


def random_forest_classifier(X_train, y_train, X_test, y_test):
    # let's see different configurations for the RandomForestClassifier
    # using GridSearchCV
    model = RandomForestClassifier(criterion='entropy', random_state=6)

    params = [{'n_estimators': [500, 1000, 1500], 'max_depth': [10, 20, 25],
               'min_samples_leaf': [10, 20, 25]}]

    grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy',
                               n_jobs=-1, cv=5, verbose=3)

    grid_search.fit(X_train, y_train)

    # print the best configuration parameters and the model
    print(f'The best parameters for the model are {grid_search.best_params_}')
    print(f'The best score for the model is {grid_search.best_score_}')

    # extract the best model
    best_model = grid_search.best_estimator_

    y_preds = best_model.predict(X_test)

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
    return best_model


# second classifier - SVM
def SVM(X_train, y_train, X_test, y_test):
    svc = SVC(random_state=6)

    # we will discuss these different configuration
    params = [{'C': [0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'poly', 'rbf'],
               'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}]

    # create the grid search object for the purpose
    grid_search = GridSearchCV(estimator=svc, param_grid=params, scoring='accuracy',
                               n_jobs=-1, cv=5, verbose=3)

    grid_search.fit(X_train, y_train)

    # print the best configuration parameters and the model
    print(f'The best parameters for the model are {grid_search.best_params_}')
    print(f'The best score for the model is {grid_search.best_score_}')

    # extract the best model
    best_estimator = grid_search.best_estimator_

    # evaluate the accuracy score and the ROC AUC score for the best estimator
    y_preds = best_estimator.predict(X_test)

    print(f'The accuracy score for the best SVM classifier is {accuracy_score(y_test, y_preds)}')
    print(f'The ROC AUC score for the best SVM classifier is {roc_auc_score(y_test, y_preds)}')

    # extract the confusion matrix for the best estimator
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
    return best_estimator


# third classifier - Naive Bayes (Multinominal)

def multinomial_NB(X_train, y_train, X_test, y_test):
    MNB = MultinomialNB()
    MNB.fit(X_train, y_train)
    return MNB

    # trying to predict the sentiment (It's just a try man!)
    # string = 'I love this dress!'
    # review = re.sub('[^a-zA-Z]', ' ', string)
    #
    # # convert all cases to lower cases
    # review = review.lower()
    #
    # # split to array
    # review = review.split()
    #
    # # creating PorterStemmer object to take main stem
    # # each word
    # ps = PorterStemmer()
    #
    # review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #
    # # join all string array elements
    # # to create back into a string
    # review = ' '.join(review)
    #
    # review = [review]
    #
    # result = cv.transform(review)
    # ris = MNB.predict(result)
    # print(ris)


# Fourth classifier - CNN
def get_cnn_model(X_train, X_test, y_train):
    # ---------- attributi per la rete convoluzionale ----------
    loss = 'binary_crossentropy'
    metrics = ['accuracy']
    learning_rate = 0.001
    verbose = 1
    epochs = 50
    batch_size = 128
    validation_split = 0.2

    max_words, max_len, X_train_seq, _ = preprocessing_CNN(X_train, X_test)
    model = Sequential()

    # It is an improvement over more the traditional
    # bag-of-word model encoding schemes where large sparse
    # vectors were used to represent each word or to score
    # each word within a vector to represent an entire
    # vocabulary. These representations were sparse because
    # the vocabularies were vast and a given word or
    # document would be represented by a large vector
    # comprised mostly of zero values.
    # Instead, in an embedding, words are represented
    # by dense vectors where a vector represents the
    # projection of the word into a continuous vector space.
    # The position of a word within the vector space is
    # learned from text and is based on the words
    # that surround the word when it is used.
    model.add(Embedding(max_words, 100, input_length=max_len))

    model.add(Conv1D(1024, 3, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPooling1D())
    # The Dropout layer randomly sets input units to 0 with a
    # frequency of rate at each step during training time,
    # which helps prevent overfitting. Inputs not set to 0
    # are scaled up by 1/(1 - rate) such that the sum over all
    # inputs is unchanged.
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(2048, activation='relu'))

    model.add(Dropout(0.5))
    # normalizza il suo input in maniera tale da avere una
    # distribuzione con la media vicino allo 0 e deviazione
    # standard vicino a 1
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    print("Starting...\n")

    start_time = time.time()

    print("\n\nCompliling Model ...\n")

    optimizer = Adam(learning_rate)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    print("Trainning Model ...\n")
    model.fit(
        X_train_seq,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_split=validation_split,
        use_multiprocessing=True
    )

    elapsed_time = time.time() - start_time
    print(f"\nElapsed Time: {elapsed_time} sec")

    return model
