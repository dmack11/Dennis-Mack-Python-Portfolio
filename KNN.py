#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 19:40:53 2023

@author: dennismack
"""
#%%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
#%%
df = pd.read_csv('Data/KNN_Project_Data')
#%%
'''
EDA Section, data is artificial so here we'll just do one large pairplot'
'''
sns.pairplot(df, hue = 'TARGET CLASS')
#%%
scaler = StandardScaler()
#%%
scaler.fit(df.drop('TARGET CLASS', axis=1))
#%%
scaled_feat  = scaler.transform(df.drop('TARGET CLASS', axis=1))
#%%
df_feat = pd.DataFrame(scaled_feat, columns=df.columns[:-1])
#%%
X = df_feat
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#%%
knn = KNeighborsClassifier(n_neighbors=1)
#%%
knn.fit(X_train, y_train)
#%%
pred = knn.predict(X_test)
#%%
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
#%%
# Finding the best K value
error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
#%%
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate, color = 'blue',linestyle = '--', marker = 'o', markerfacecolor = 'red', markersize = 10)
#%%
#Retrain with new K value
knn = KNeighborsClassifier(n_neighbors=39)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test,pred))








