
# Sentiment from reviews

*Sentiment analysis* è un approccio utilizzato per analizzare i dati 
e per recuperare il grado di approvazione celato dietro una frase 
di senso compiuto *(positivo, negativo, neutrale o più classificazioni)*. Con l'enorme aumento delle tecnologie web, 
il numero di persone che esprimono le proprie opinioni via web
sta crescendo. Queste informazioni sono soprattutto utili per
le aziende in quanto l'analisi permette loro di poter misurare
la risposta dei consumatori nei confronti dei prodotti acquistati
o di poter condurre analisi di mercato. ***Reviews sentiment 
analysis*** è nella fattispecia un'applicazione del sentiment 
analysis che ha per oggetto le recensioni degli utenti 
(nel caso specifico si tratta di indumenti).\
Il progetto si pone come obiettivo quello di realizzare un kernel
costituito dai seguenti classificatori per confrontarne le performance: ***Random Forest, SVM, Naive Bayes e CNN.***\
Il progetto è stato interamente realizzato in *Python* con le 
relative librerie messe a disposizione per il machine learning.
## Authors

- [@tanus99](https://github.com/tanus99)


## Run Locally

Clona il progetto

```bash
  git clone https://github.com/tanus99/sentiments_from_reviews
```

Spostati nella cartella del progetto e assicurati di attivare
l'ambiente virtuale.
```bash
cd sentiments_from_reviews
source /bin/activate
```
Per il CNN è necessario installare le librerie *tensorflow*
e *keras*

```bash
pip install tensorflow
pip install keras
```


## Script python

Il progetto presenta 4 script python:
- main.py
- preparing_methods.py
- classifiers.py
- utils.py

Il **primo** permette di richiamare tutti i classificatori implementati.\
Il **secondo** contiene tutte le funzioni necessarie ad effettuare il
pre-processing dei dati ed estrazione delle features *(BoW e TF-IDF)*.\
Il **terzo** contiene tutti classificatori utilizzati per lo scopo\
Il **quarto** continene funzioni di supporto e funzioni per stampare
le performance dei vari classificatori *(compreso di Confusion Matrix, ROC curve, ROC AUC score, accuracy score)*

⚠️ ***Seguire quanto scritto nei commenti per una corretta esecuzione
degli script***


## Results
Il classicatore migliore tra tutti quelli analizzati è risultato 
essere il CNN con un'accuracy dell'87% sul test set, ROC AUC score di
0.717.\
Di seguito la curva ROC ottenuta, l'andamento dell'accuracy e della
funzione dei costi.
![ROC curve](/img/cnn_roc_curve.png)
![Accuracy](/img/accuracy.png)
![Loss](/img/loss.png)
## License
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)


