import os.path
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import seaborn as sns
from matplotlib import pyplot as plt
from tensorflow import keras

# crea la cartella img
def create_dir():
    global root_path
    root_path = os.path.join(os.getcwd(), 'img')

    if not os.path.exists(root_path):
        os.makedirs(root_path, exist_ok=True)

# stampo la curva ROC
def plot_roc_auc_curve(fpr, tpr, roc_auc, filename):

    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plt.savefig(f"{root_path}/{filename}_roc_curve.png")

# stampo le performance del modello sul test set
def print_performance_metrics(model, X_test, y_test):
    # valutiamo l'accuracy sulla parte di test
    # accuracy = model.evaluate(X_test_seq, y_test)
    # print(accuracy)
    # valutazione sulle predizioni (valore uguale al precedente metodo)
    # - accuratezza del modello(n.o % di dati etichettati corretti)
    # - ROC AUC score
    y_preds = (model.predict(X_test) > 0.5).astype('int32')
    print(f'The accuracy score on test set is {accuracy_score(y_test, y_preds)}')
    print(f'The ROC AUC score on test set is {roc_auc_score(y_test, y_preds)}')

# stampo la matrice di confusione
def print_confusion_matrix(model, X_test, y_test, filename):
    create_dir()

    y_preds = (model.predict(X_test) > 0.5).astype('int32')
    # stampa la matrice di confusione per visualizzare i dati correttamente etichettati
    cm = confusion_matrix(y_test, y_preds)

    print('CONFUSION MATRIX\n\n', cm)

    print(f'True Positive: {cm[0, 0]}\n')

    print(f'True Negative: {cm[1, 1]}\n')

    print(f'False Positive: {cm[0, 1]}\n')

    print(f'False Negative: {cm[1, 0]}\n')

    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                             index=['Predict Positive:1', 'Predict Negative:0'])

    plot = sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    root_path = os.path.join(os.getcwd(),'img')

    if not os.path.exists(root_path):
        os.makedirs(root_path,exist_ok=True)

    plot.figure.savefig(f"{root_path}/{filename}_cm.png")
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

    # ricavo i parametri utili alla rappresentazione della curva ROC
    fpr, tpr, _ = roc_curve(y_test,y_preds)
    roc_auc = roc_auc_score(y_test, y_preds)

    plot_roc_auc_curve(fpr,tpr,roc_auc, filename)


# salva l'oggetto contenente il classificatore addestrato
def save_obj(model, filename):
    root_path = os.getcwd()
    root_path = os.path.join(root_path, 'models')
    if not os.path.exists(root_path):
        os.makedirs(root_path,exist_ok=True)

    filename = filename.upper()
    if(filename == 'CNN'):
        model.save(filename)
    else:
        with open(f'{root_path}/{filename}.pkl','wb') as file:
            pickle.dump(model,file)


# carica l'oggetto contenente il classificatore addestrato
def read_obj(filename):
    root_path = os.getcwd()
    root_path = os.path.join(root_path, 'models')
    filename = filename.upper()
    if(filename == 'CNN'):
        model = keras.models.load_model(filename)
        return model
    else:
        with open(f'{root_path}/{filename}.pkl', 'rb') as file:
            model = pickle.load(file)
            return model

