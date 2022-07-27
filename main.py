import pandas as pd
import numpy as np

#just import and read the data
df = pd.read_csv('./data/reviews.csv')
#print(df.describe())
#shuffling the data frame and removing the default index
df = df.sample(frac=1, random_state=6).reset_index(drop=True)

#selecting the relevant columns from the dataset
df = df[['Review Text', 'Rating', 'Recommended IND']]
#print(df.head())

#I see if there are some missing values
print(df.isna().sum())

df = df['Review Text'].fillna('')


