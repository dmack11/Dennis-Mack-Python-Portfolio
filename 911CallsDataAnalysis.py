#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:37:21 2023

@author: dennismack
"""

#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%%
df = pd.read_csv('Data/911.csv')
#%%
df.info()
#%%
df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])
#%%
df['Reason'].value_counts()
#%%
sns.countplot(data=df, x='Reason')
#%%
dt = np.dtype(df['timeStamp'])
#%%
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
#%%
time = df['timeStamp']
#%%
hour = []
month = []
day_of_week = []
for index, row in df.iterrows():
    time = df['timeStamp'].iloc[index]
    hour.append(time.hour),
    month.append(time.month),
    day_of_week.append(time.day_of_week)
df['hour'] = hour
df['month'] = month
df['Day of Week'] = day_of_week
#%%
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
#%%
df['Day of Week'] = df['Day of Week'].map(dmap)
#%%
sns.countplot(x='Day of Week', data = df, hue = 'Reason')
#%%
sns.countplot(x = 'month', data = df, hue = 'Reason')
#%%
byMonth = df.groupby('month').count()
#%%
byMonth['e'].plot()
#%%
byMonth.reset_index(inplace= True)
#%%
sns.lmplot(x='month', y='twp', data = byMonth)
#%%
df['Date'] = df['timeStamp'].apply(lambda x: x.date())
#%%
dateGroup = df.groupby('Date').count()
#%%
dateGroup['e'].plot()
#%%
df[df['Reason']=='Traffic'].groupby('Date').count().plot()
#%%
df[df['Reason']=='Fire'].groupby('Date').count().plot()
#%%
df[df['Reason']=='EMS'].groupby('Date').count().plot()
#%%
df2 = df.groupby(by=['Day of Week', 'hour']).count()
#%%
dayHour = df.groupby(by=['Day of Week','hour']).count()['Reason'].unstack()
#%%
sns.heatmap(data=dayHour)
#%%
sns.clustermap(data=dayHour)
#%%
monthYear = df.groupby(by = ['Day of Week','month']).count()['Reason'].unstack()
#%%
sns.heatmap(data = monthYear)
#%%
sns.clustermap(data = monthYear)












