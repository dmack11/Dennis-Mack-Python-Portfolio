#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:10:21 2023

@author: dennismack
"""
#%%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report
#%%
colleges = pd.read_csv('Data/College_Data', index_col=[0])
#%%
colleges.info()
#%%
colleges.describe()
#%%
#EDA
sns.set_style('whitegrid')
sns.scatterplot(x = 'Room.Board', y = 'Grad.Rate', data = colleges, hue = 'Private', alpha = 0.5)
#%%
sns.scatterplot(x='Outstate',y='F.Undergrad',data=colleges,hue='Private',alpha=0.5)
#%%
sns.set_style('darkgrid')
g = sns.FacetGrid(colleges,hue="Private",palette='coolwarm',height=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)
#%%
h = sns.FacetGrid(colleges,hue="Private",palette='coolwarm',height=6,aspect=2)
h = h.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)
#%%
colleges[colleges['Grad.Rate']>100]
#%%
colleges['Cazenovia College']['Grad.Rate'] = 100
#%%
colleges.loc['Cazenovia College']
#%%
kmeans = KMeans(n_clusters=2)
#%%
kmeans.fit(colleges.drop('Private', axis=1))
#%%
kmeans.cluster_centers_
#%%
'''
Usually there is no perfect way to evaluate the model without the labels. But since we actually do have them here
we can evaluate for practice
'''
def isPrivate(cluster):
    if cluster == 'Yes':
        return 1
    else:
        return 0
#%%
colleges['Cluster'] = colleges['Private'].apply(isPrivate)
#%%
print(confusion_matrix(colleges['Cluster'],kmeans.labels_))
print(classification_report(colleges['Cluster'],kmeans.labels_))  
    
    
    
    
    
    
    
    
    
    
    
