import os
import time

import pandas as pd
from keras import regularizers
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
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dropout, BatchNormalization, Dense, Flatten
from keras.models import Sequential

# Primo classificatore - RandomForest
from preparing_methods import preprocessing_CNN


def random_forest_classifier(X_train, y_train):
    # vediamo diverse configurazioni per RandomForestClassifier usando GridSearchCV
    model = RandomForestClassifier(criterion='entropy', random_state=6)

    params = [{'n_estimators': [500, 1000, 1500], 'max_depth': [10, 20, 25],
               'min_samples_leaf': [10, 20, 25]}]

    grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy',
                               n_jobs=-1, cv=5, verbose=3)

    grid_search.fit(X_train, y_train)

    # stampa i migliori parametri di configurazione e lo score del modello migliore
    print(f'The best parameters for the model are {grid_search.best_params_}')
    print(f'The best accuracy score for the model is {grid_search.best_score_}')

    # estrae il modello migliore
    best_model = grid_search.best_estimator_

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


# Secondo classificatore - SVM
def SVM(X_train, y_train):
    svc = SVC(random_state=6)

    # valuteremo la configurazione migliore considerando tutti questi parametri
    params = [{'C': [0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'poly', 'rbf'],
               'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}]

    # per lo scopo utilizziamo l'oggetto GridSearch
    grid_search = GridSearchCV(estimator=svc, param_grid=params, scoring='accuracy',
                               n_jobs=-1, cv=5, verbose=3)

    grid_search.fit(X_train, y_train)

    # stampa i migliori parametri di configurazione e lo score del modello migliore
    print(f'The best parameters for the model are {grid_search.best_params_}')
    print(f'The best accuracy score for the model is {grid_search.best_score_}')

    # estrae il modello migliore
    best_estimator = grid_search.best_estimator_

    return best_estimator


# Terzo classificatore - Naive Bayes (Multinominal)

def multinomial_NB(X_train, y_train):
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


# Quarto classificatore - CNN
def get_cnn_model(X_train, y_train):
    # ---------- attributi per la rete convoluzionale ----------
    loss = 'binary_crossentropy'
    metrics = ['accuracy']
    learning_rate = 0.001
    verbose = 1
    epochs = 30
    batch_size = 128
    validation_split = 0.2
    l2 = regularizers.L2(0.08)

    max_words, max_len, X_train_seq, tokenizer = preprocessing_CNN(X_train)
    model = Sequential()

    # -- EMBEDDING LAYER --
    # È un miglioramento rispetto ai tradizionali schemi di codifica del modello bag-of-word
    # in cui sono stati utilizzati grandi vettori sparsi per rappresentare ogni parola o per
    # assegnare un punteggio a ciascuna parola all'interno di un vettore per rappresentare un 
    # intero vocabolario. Queste rappresentazioni erano scarse perché i vocabolari erano vasti 
    # e una data parola o documento sarebbe stato rappresentato da un grande vettore composto 
    # principalmente da valori zero. Invece, in un embedding, le parole sono rappresentate da 
    # vettori densi in cui un vettore rappresenta la proiezione della parola in uno spazio vettoriale
    # continuo. La posizione di una parola all'interno dello spazio vettoriale viene appresa dal
    # testo e si basa sulle parole che circondano la parola quando viene utilizzata.
    
    # -- CONV1D LAYER --
    # Permette di realizzare la convoluzione ad una dimensione per poter estrarre le informazioni utili
    # alla predizione direttamente dalle recensioni proiettate nello spazio vettoriale grazie all'embedding
    # layer. Crea la feature map che riassume la presenza delle features cercate nell'input dato

    # -- GLOBALMAXPOOLING1D LAYER --
    # Estrae il valore massimo da ciascun filro. Serve per ridurre la dimensionalità
    # delle features estratte

    
    # -- DROPOUT LAYER --
    # Imposta in modo casuale le unità di input su 0 con una frequenza pari a quella indicata 
    # come parametro e questo ad ogni step durante il tempo di allenamento, il che aiuta a prevenire 
    # l'overfitting. Gli ingressi non impostati su 0 vengono aumentati di 1/(1 - rate) 
    # in modo tale che la somma di tutti gli ingressi rimanga invariata.

    # -- BATCH NORMALIZATION --
    # normalizza il suo input in maniera tale da avere una
    # distribuzione con la media vicino allo 0 e deviazione
    # standard vicino a 1. Serve ad accelerare il training e per permettere
    # l'utilizzo di learning rate più alti, rendendo il learning più semplice e veloce.
    
    model.add(Embedding(max_words, 100, input_length=max_len))
    model.add(Conv1D(16, 3, padding='valid', activation='relu', strides=1, kernel_regularizer=l2))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    model.add(Dense(250, activation='relu', kernel_regularizer=l2))
    model.add(Dropout(0.5))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2))

    model.summary()

    print("Starting...\n")

    start_time = time.time()

    print("\n\nCompliling Model ...\n")

    optimizer = Adam(learning_rate)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    print("Trainning Model ...\n")
    history = model.fit(
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

    root_path = os.path.join(os.getcwd(), 'img')

    # stampo l'andamento dell'accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f"{root_path}/accuracy.png")
    plt.show()

    # stampo l'andamento del loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f"{root_path}/loss.png")
    plt.show()

    return model,tokenizer
