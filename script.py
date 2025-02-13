import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import spacy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('stopwords')
from nltk.corpus import stopwords

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

# Heatmap to show the correlation between the features
sns.heatmap(df[['Number_Characters', 'Number_Words', 'Number_Sentences']].corr(), annot=True)
plt.show()


# Visualizing the data

# Number of Characters
plt.figure(figsize=(10, 6))
sns.histplot(df[df['label']==0]['Number_Characters'], color='blue', label='Ham')
sns.histplot(df[df['label']==1]['Number_Characters'], color='red', label='Spam') 
plt.show()

# Preprocessing

df['transformed_text'] = df['sms'].str.lower()

df['transformed_text'] = df['transformed_text'].apply(word_tokenize)

# Remove special characters
df['transformed_text'] = df['transformed_text'].apply(lambda x: [re.sub(r'[^a-zA-Z0-9\s]', '', word) for word in x])

# Removing stopwords
stop_words = set(stopwords.words('english'))
df['transformed_text'] = df['transformed_text'].apply(lambda x: [word for word in x if word not in stop_words and word not in string.punctuation])

# Convert the preprocessed text back to string
df['transformed_text'] = df['transformed_text'].apply(lambda x: ' '.join(x))

# Display the preprocessed data
print(df[['sms', 'transformed_text']].head())


from wordcloud import WordCloud
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')

# Spam words Visualization
spam_wc = wc.generate(df[df['label'] == 1]['transformed_text'].str.cat(sep=" "))
plt.imshow(spam_wc)
plt.show()


# Ham words Visualization
ham_wc = wc.generate(df[df['label'] == 0]['transformed_text'].str.cat(sep=" "))
plt.imshow(ham_wc)
plt.show()


# Training and Testing Data

y = df['label']

x_train, x_test, y_train, y_test = train_test_split(df['transformed_text'], y, test_size=0.3, random_state=42)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train = tfidf_vectorizer.fit_transform(x_train.values)

tfidf_test = tfidf_vectorizer.transform(x_test.values)

# Print the first 10 features
print(tfidf_vectorizer.get_feature_names_out()[:10])

# Model Building
nb_classifier = MultinomialNB()
nb_classifier.fit(tfidf_train, y_train)
pred_nb = nb_classifier.predict(tfidf_test)

logistic_classifier = LogisticRegression()
logistic_classifier.fit(tfidf_train, y_train)
pred_logistic = logistic_classifier.predict(tfidf_test)

rf_classifier = RandomForestClassifier()
rf_classifier.fit(tfidf_train, y_train)
pred_rf = rf_classifier.predict(tfidf_test)

svm_classifier = SVC()
svm_classifier.fit(tfidf_train, y_train)
pred_svm = svm_classifier.predict(tfidf_test)

print('Accuracy of Naive Bayes:', accuracy_score(y_test, pred_nb))
print('Accuracy of Logistic Regression:', accuracy_score(y_test, pred_logistic))
print('Accuracy of Random Forest:', accuracy_score(y_test, pred_rf))
print('Accuracy of SVM:', accuracy_score(y_test, pred_svm))







