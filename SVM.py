#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:56:16 2023

@author: dennismack
"""
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
#%%
iris = sns.load_dataset('iris')
#%%
sns.pairplot(iris, hue='species')
#%%
sns.kdeplot(x='sepal_width',y='sepal_length',data=iris[iris['species']=='setosa'], cmap='plasma')
#%%
X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
#%%
model = SVC()
#%%
model.fit(X_train, y_train)
#%%
predictions = model.predict(X_test)
#%%
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
#%%
#Gridsearch Practice
param_grid = {'C':[1,10,100,100,1000,10000], 'gamma':[1,0.1,0.01,0.001,0.0001,0.00001]}
#%%
grid = GridSearchCV(SVC(), param_grid, verbose=10)
grid.fit(X_train,y_train)
#%%
grid_pred = grid.predict(X_test)
#%%
print(confusion_matrix(y_test,grid_pred))
print(classification_report(y_test,grid_pred))