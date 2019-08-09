# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:10:59 2019

@author: Walmond
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset
dataset = pd.read_csv('dark_traffic.csv')

# create function to clean html text
import re
from urllib.parse import urlparse

def strip_url(text):
    url = urlparse(text)
    url = url._replace(scheme = ' ', query = ' ')
    return url.geturl()

def remove_special_characters(text):
    return re.sub('[/+.?:]|-|https:|html|www|/d', ' ', text)

def denoise_text(text):
    text = strip_url(text)
    text = remove_special_characters(text)
    return text

# Cleaning the text
# Set() are used to speed up the process of for loop
# Stopwords contain words that are not necessary
# Stemmer contain the process of reverting all words into its 
## original main form
# Corpus is collection of text
# Spartsity indicates the amount of sparmatrix within a data or 
## variable with very little amount
# Tokenization is a process of put unique into different column
# By cleaning out the text outside of CountVectorizer parametter,
## you'll have more option
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = [] #still Empty
for i in range (0, 669): # Modify this according to data size 
    review = denoise_text(dataset['Landing Page'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#Creating the bag of words model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Part 2 - Create the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# Unit = Amount of node in hidden layer ((Independent v. +1) /2)
# Kernel_Initializer = Weight adjuster (close to 0)
# activation = 'relu' (Reticfier)
# input_dim = number of independent v.
classifier.add(Dense(units = 703, kernel_initializer = 'uniform', activation = 'relu', input_dim = 1405))

# Adding Second hidden layer
classifier.add(Dense(units = 703, kernel_initializer = 'uniform', activation = 'relu'))

# Adding Third hidden layer
classifier.add(Dense(units = 703, kernel_initializer = 'uniform', activation = 'relu'))

# Adding output layer
# For multiple dependent v. the needed activation function is softmax
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'relu'))

# Compiling the ANN
# Optimizer + parameter which decides way to find the best weight. Adam is one of the best for stochastic algorithym
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test result
Y_Pred = classifier.predict(X_test)
Y_Pred = (Y_Pred > 0.5)

# Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_Pred)
