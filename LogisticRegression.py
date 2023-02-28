#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:57:08 2023

@author: dennismack
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
#%%
ad_data = pd.read_csv('Data/advertising.csv')
#%%
ad_data.info()
#%%
ad_data.describe()
#%%
#EDA
sns.set_style('whitegrid')
sns.histplot(x = 'Age', data = ad_data, bins = 30)
#%%
sns.jointplot(x='Age', y='Area Income', data = ad_data)
#%%
sns.jointplot(x='Age', y='Daily Time Spent on Site',color = 'red', data = ad_data, kind = 'kde')
#%%
sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data = ad_data)
#%%
sns.pairplot(ad_data, hue = 'Clicked on Ad')
#%%
# Model Creation and Training
ad_data.nunique()
#%%
X = ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
#%%
logmodel = LogisticRegression()
#%%
logmodel.fit(X_train, y_train)
#%%
predictions = logmodel.predict(X_test)
#%%
print(classification_report(y_test,predictions))