# Spam Detection using NLP

## **Project Overview**
--------------------

This project focuses on building a **spam detection system** using Natural Language Processing (NLP) techniques. The goal is to classify text messages (SMS) as either **spam** or **ham** (non-spam). The project involves preprocessing the text data, performing feature engineering, and training machine learning models to accurately detect spam messages.

## **Dataset**
-----------

The dataset used in this project is a collection of SMS messages labeled as either **spam** or **ham**. It was sourced from the **UCI Machine Learning Repository** and can be accessed via Hugging Face Datasets at this link: [SMS Spam Dataset](https://huggingface.co/datasets/ucirvine/sms_spam).

### **Dataset Structure**

The dataset contains the following columns:

*   **sms**: The raw text of the SMS message.
    
*   **label**: A binary label where 1 indicates spam and 0 indicates ham.
    

### **Dataset Statistics**

*   Total rows: 5,574
    
*   Duplicates: 403 (removed during preprocessing)
    
*   Final dataset size: 5,171 rows
    
*   Class distribution:
    
    *   Ham: 4,518 (87.4%)
        
    *   Spam: 653 (12.6%)
        

## **Project Steps**
-----------------

The project is divided into the following steps:

### **1\. Data Validation**

*   Check for missing values and duplicates.
    
*   Remove duplicates to ensure data quality.
    

### **2\. Exploratory Data Analysis (EDA)**

*   Analyze the distribution of spam vs. ham messages.
    
*   Visualize the length of messages (number of characters, words, and sentences) for spam and ham.
    
*   Generate word clouds for spam and ham messages to identify common words.
    

### **3\. Preprocessing**

*   Convert text to lowercase.
    
*   Tokenize the text into individual words.
    
*   Remove special characters and stopwords.
    
*   Stem or lemmatize words to reduce them to their root forms.
    
*   Convert the processed tokens back into strings.
    

### **4\. Feature Engineering**

*   Create new features such as:
    
    *   Number of characters in each message.
        
    *   Number of words in each message.
        
    *   Number of sentences in each message.
        
    *   Presence of URLs.
        
    *   Presence of special characters.
        
    *   Word diversity (ratio of unique words to total words).
        

### **5\. Vectorization**

*   Use **TF-IDF Vectorization** to convert text data into numerical features.
    
*   Limit the number of features to the top 3,000 most frequent terms.
    

### **6\. Model Training**

*   Train multiple machine learning models, including:
    
    *   **Naive Bayes**
        
    *   **Logistic Regression**
        
    *   **Random Forest**
        
    *   **Support Vector Machine (SVM)**
        
*   Evaluate models using accuracy and other metrics like precision, recall, and F1-score.
    

### **7\. Model Evaluation**

*   Compare the performance of the models.
    
*   Identify the best-performing model for spam detection.
    

## Code Implementation
-----------------------

The project is implemented in Python using the following libraries:

*   **Pandas** and **NumPy** for data manipulation.
    
*   **Matplotlib** and **Seaborn** for data visualization.
    
*   **NLTK** and **re** for text preprocessing.
    
*   **Scikit-learn** for feature extraction, model training, and evaluation.
    

## **Results**
-----------

The best-performing model was **Random Forest**, achieving an accuracy of **97.1%**. Below is a summary of the model performances:

*   **Naive Bayes**: 96.1%
    
*   **Logistic Regression**: 95.1%
    
*   **Random Forest**: 97.1%
    
*   **SVM**: 96.8%
    

**How to Run the Code**
-----------------------

1.  Copy pip install pandas numpy matplotlib seaborn nltk scikit-learn
    
2.  Run the provided Python script to preprocess the data, train the models, and evaluate their performance.
    

**Future Improvements**
-----------------------

*   Experiment with advanced NLP techniques like word embeddings (Word2Vec, GloVe, or BERT).
    
*   Address class imbalance using techniques like oversampling or class weighting.
    
*   Deploy the model as a web application using Flask or FastAPI.
