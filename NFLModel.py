#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
#%%
nfl = pd.read_csv('2021_NFL_DATA.csv')
#%%
nfl['result'].value_counts()
#%%
resultEncoder = {'result':{'W':1,'L':0,'T':0}}
nfl.replace(resultEncoder, inplace=True)
nfl['result'].value_counts()
#%%
sns.boxplot(x='result', y='1stD_offense', data=nfl)
#%%
features = nfl.iloc[:,8:]
#%%
scaler = StandardScaler()
scaler.fit(features)
X= scaler.transform(features)
#%%
y=nfl['result']
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
#%%
lrc = LogisticRegression()
lrc.fit(X_train, y_train)
#%%
y_pred=lrc.predict(X_test)
#%%
accuracy_score(y_test, y_pred)
#%%
penalties = ['l1', 'l2']
C = [0.01, 0.1, 1.0, 10.0, 1000.0]
for penalty in penalties:
    for c in C:
        lrc_tuned = LogisticRegression(penalty=penalty, C=c, solver='liblinear')
        lrc_tuned.fit(X_train, y_train)
        y_pred = lrc_tuned.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_rd = round(accuracy*100,1)
        print(f'Accuracy: {accuracy_rd}% | penalty = {penalty}, C = {c}')
#%%
 
'''
optimal penalty and C
penalty = 'l1'
C = 0.1
'''
test_sizes = [val/100 for val in range(20,36)]

for test_size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    lrc_tts = LogisticRegression(penalty = penalty, C = C, solver='liblinear')
    lrc_tts.fit(X_train, y_train)
    y_pred = lrc_tts.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_rd = round(accuracy*100,1)
    print(f'Accuracy: {accuracy_rd}% | test size = {test_size}') 
#%%
# Optimal test size and hyperparameters
test_size = 0.25
penalty = 'l1'
C = 0.1


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, random_state=42)
optLr = LogisticRegression(penalty = penalty, C = C, solver='liblinear')
optLr.fit(X_train, y_train)   
#%%

importance = abs(optLr.coef_[0])

sns.barplot(x=importance, y=features.columns)

plt.suptitle('Feature Importance for Logistic Regression')
plt.xlabel('Score')
plt.ylabel('Stat')
plt.show()


for i,v in enumerate(importance.round(2)):
    print(f'Feature: {features.columns[i]}, Score: {v}')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        