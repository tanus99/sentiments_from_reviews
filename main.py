import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#just import and read the data
df = pd.read_csv('./data/reviews.csv')
#print(df.describe())

#shuffling the data frame and removing the default index
df = df.sample(frac=1, random_state=6).reset_index(drop=True)

#selecting the relevant columns from the dataset
df = df[['Review Text', 'Rating', 'Recommended IND']]
#print(df.head())

#I see if there are some missing values
#print(df.isna().sum())
df['Review Text'].fillna('')

corpus = []

#importing the so called stopwords
nltk.download('stopwords')

# print(set(stopwords.words('english')))



#preprocessing function to clean the data
def preprocessing():
    for i in range(0,5):
        #column Review Text, row ith
        review = re.sub('[^a-zA-Z]', ' ', df['Review Text'][i])

        #convert all cases to lower cases
        review = review.lower()

        #split to array
        review = review.split()

        #creating PorterStemmer object to take main stem
        #each word
        ps = PorterStemmer()

        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

        #join all string array elements
        #to create back into a string
        review = ' '.join(review)

        #append each string to create
        #arrat of clean text
        corpus.append(review)

preprocessing()