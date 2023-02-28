#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 22:08:46 2023

@author: dennismack
"""
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
#%%
loans = pd.read_csv('Data/loan_data.csv')
#%%
loans.info()
#%%
loans.describe()
#%%
#EDA
sns.histplot(x='fico',data=loans[loans['credit.policy']==1],color='red')
sns.histplot(x='fico',data=loans[loans['credit.policy']==0])
#%%
sns.histplot(x = 'fico', data = loans[loans['not.fully.paid'] == 1], color = 'red')
sns.histplot(x = 'fico', data = loans[loans['not.fully.paid'] == 0])
#%%
plt.figure(figsize=(10,6))
sns.countplot(loans['purpose'], hue = loans['not.fully.paid'])
#%%
sns.jointplot(x='fico', y='int.rate', data = loans)
#%%
sns.lmplot(x = 'fico', y='int.rate', data = loans, hue = 'credit.policy', col='not.fully.paid')
#%%
#Feature Engineering
cat_feats = ['purpose']
#%%
final_data = pd.get_dummies(loans, columns=cat_feats, drop_first=True)
#%%
#Model Creation and Training
X = final_data.drop('not.fully.paid', axis = 1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#%%
dtree = DecisionTreeClassifier()
#%%

dtree.fit(X_train,y_train)
#%%
predictions = dtree.predict(X_test)
#%%
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test, predictions))
#%%
'''
We trained one decision treee, now time for a random forest
'''
rfc = RandomForestClassifier()
#%%
rfc.fit(X_train,y_train)
#%%
rfc_pred = rfc.predict(X_test)
#%%
print(classification_report(y_test,rfc_pred))
print(confusion_matrix(y_test,rfc_pred))






