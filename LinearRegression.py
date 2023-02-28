#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:04:03 2023

@author: dennismack
"""
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#%%
customers = pd.read_csv('Data/Ecommerce Customers')
#%%
customers.describe()
#%%
customers.info()
#%%
#EDA
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers)
#%%
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers)
#%%
sns.jointplot(x='Time on App', y='Length of Membership', data=customers, kind = 'hex')
#%%
sns.pairplot(customers)
#%%
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)
#%% 
#Model Creation and Training
customers.columns
#%%
X = customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
X
#%%
y = customers['Yearly Amount Spent']
y
#%%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)
#%%
lm = LinearRegression()
#%%
lm.fit(X_train,y_train)
#%%
print(lm.coef_)
#%%
predictions = lm.predict(X_test)
predictions
#%%
plt.scatter(y_test, predictions)
#%%
print(metrics.mean_absolute_error(y_test,predictions))
print(metrics.mean_squared_error(y_test,predictions))
print(np.sqrt(metrics.mean_squared_error(y_test,predictions)))
#%%
sns.distplot((y_test-predictions))
#%%
pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])








