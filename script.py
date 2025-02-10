import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import spacy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize, sent_tokenize

df = pd.read_parquet('plain_text/train-00000-of-00001.parquet')


# Data validation
print(df.info())

print(df.duplicated().sum())

df = df.drop_duplicates()

print(df.shape)

# EDA

print(df.head(5))

print(df['label'].value_counts())

# Bar plot with the columns instead of showing 0 or 1 showing spam or ham
sns.barplot(df['label'].value_counts()) 
plt.show()


# Feature Engineering
df['Number_Characters'] = df['sms'].apply(len)

df['Number_Words'] = df['sms'].apply(lambda x: len(word_tokenize(x)))

df['Number_Sentences'] = df['sms'].apply(lambda x: len(sent_tokenize(x)))

print(df.describe())


# Visualizing the data

# Number of Characters
plt.figure(figsize=(10, 6))
sns.histplot(df[df['label']==0]['Number_Characters'], color='blue', label='Ham')
sns.histplot(df[df['label']==1]['Number_Characters'], color='red', label='Spam') 
plt.show()

# Training and Testing Data

y = df['label']

x_train, x_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.3, random_state=42)

count_vectorizer = CountVectorizer(stop_words='english')

count_train = count_vectorizer.fit_transform(x_train.values)

count_test = count_vectorizer.transform(x_test.values)

count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names_out())

nb_classifier = MultinomialNB()

nb_classifier.fit(count_train, y_train)
pred = nb_classifier.predict(count_test)

confusion_matrix(y_test, pred, labels=[0, 1])

