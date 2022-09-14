import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
from tensorflow import keras


def print_performance_metrics(model, X_test, y_test):
    # valutiamo l'accuracy sulla parte di test
    # accuracy = model.evaluate(X_test_seq, y_test)
    # print(accuracy)
    # valutazione sulle predizioni (valori uguali) + ROC AUC score
    y_preds = (model.predict(X_test) > 0.5).astype('int32')
    print(f'The accuracy score is {accuracy_score(y_test, y_preds)}')
    print(f'The ROC AUC score is {roc_auc_score(y_test, y_preds)}')

def print_confusion_matrix(model, X_test, y_test):
    y_preds = (model.predict(X_test) > 0.5).astype('int32')
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

    # ricavo gli altri parametri di performance dalla confusion matrix
    TP = cm[0,0].astype(float)
    FP = cm[0,1].astype(float)
    FN = cm[1,0].astype(float)
    TN = cm[1,1].astype(float)

    # precision - recall/TPR - specificity - f1_score
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    specificity = TN/(TN+FP)
    f1_score = 2*precision*recall/(precision+recall)

    print(f'Precision: {precision}\n'
          f'Recall: {recall}\n'
          f'Specificity: {specificity}\n'
          f'F1 Score: {f1_score}')


# salva l'oggetto contenente il classificatore addestrato
def save_obj(model, filename):
    filename = filename.upper()
    if(filename == 'CNN'):
        model.save(filename)
    else:
        with open(f'{filename}.pkl','wb') as file:
            pickle.dump(model,file)


# carica l'oggetto contenente il classificatore addestrato
def read_obj(filename):
    filename = filename.upper()
    if(filename == 'CNN'):
        model = keras.models.load_model(filename)
        return model
    else:
        with open(f'{filename}.pkl', 'rb') as file:
            model = pickle.load(file)
            return model

