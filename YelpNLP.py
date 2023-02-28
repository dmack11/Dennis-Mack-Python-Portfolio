#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 21:14:20 2023

@author: dennismack
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
#%%
yelp = pd.read_csv('Downloads/Refactored_Py_DS_ML_Bootcamp-master/20-Natural-Language-Processing/yelp.csv')
yelp.info()
#%%
yelp.describe()
#%%
#EDA starts here 
yelp['text length'] = yelp['text'].apply(len)
#%%
graph = sns.FacetGrid(yelp, col='stars')
graph.map(plt.hist,'text length', bins = 50)
#%%
sns.boxplot(x='stars', y='text length', data = yelp)
#%%
sns.countplot(x='stars', data=yelp)
#%%
stars = yelp.groupby('stars').mean()
#%%
sns.heatmap(stars.corr(), cmap= 'coolwarm', annot=True)
#%%
'''
NLP Classification,
'''
yelp_class = yelp[(yelp.stars==1)| (yelp.stars==5)]
#%%
X = yelp_class['text']
y = yelp_class['stars']
#%%
cv = CountVectorizer()
X = cv.fit_transform(X)
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#%%
nb = MultinomialNB()
nb.fit(X_train,y_train)
#%%
predictions = nb.predict(X_test)
#%%
print(classification_report(y_test, predictions))
print('\n')
print(confusion_matrix(y_test, predictions))
#%%
pipeline = Pipeline([
    ('bow',CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('classifier', MultinomialNB())
    ])
#%%
X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
#%%
pipeline.fit(X_train,y_train)
#%%
pred1 = pipeline.predict(X_test)
#%%
print(classification_report(y_test,pred1))
print('\n')
print(confusion_matrix(y_test,pred1))












