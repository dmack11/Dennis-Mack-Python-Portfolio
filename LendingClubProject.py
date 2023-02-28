# Column Feature Info Data
#%%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,classification_report
from tensorflow.keras.models import load_model
#%%
#data_info = pd.read_csv('/Data/lending_club_info.csv', index_col='LoanStatNew')
data_info = pd.read_csv('Data/lending_club_info.csv', index_col='LoanStatNew')
data_info.head()
print(data_info.loc['revol_util']['Description'])
#%%
#Function to print a column description for reference

def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])
    
feat_info('mort_acc')
#%%


df = pd.read_csv('lending_club_loan_two.csv')

df.head()
df.info()
df.describe().transpose()
#%%
#Project Tasks
#Project Goal is to build a Model to predict loan status
    #EDA
#%%
sns.countplot(x='loan_status', data=df)
#%%
sns.histplot(x='loan_amnt', data=df)
df.corr()
#%%
plt.figure(figsize=(14,8))
sns.heatmap(df.corr(), data=df, annot=True, cmap='coolwarm')
#%%
#Almost Perfect Correlation with the installment and loan amount feature
feat_info('installment')
print('\n')
feat_info('loan_amnt')

sns.scatterplot(x='installment', y='loan_amnt' ,data=df)
#%%
#Boxplot showing relationship between Loan Status and Amount
sns.boxplot(x='loan_status', y='loan_amnt', data=df)

#Summary Statistics for Loan Amount, grouped by loan status
df.groupby('loan_status').describe()
#%%
#Grade and SubGrade Analysis
df['grade'].unique()
df['sub_grade'].unique()

sns.countplot(x='grade', data=df,hue='loan_status')
#%%
plot_order = df['sub_grade'].unique()
plot_order.sort()
#%%
plt.figure(figsize=(15,8))
sns.countplot(x='sub_grade',data=df, order=plot_order, palette='coolwarm', hue='loan_status')
#%%
#F and G Subgrades have a high percentage of chargeoffs
chargeoffDF = df[df['grade'].isin(['F','G'])]
chargeoffDF.head()
#%%
chargeplot_order = chargeoffDF['sub_grade'].unique()
chargeplot_order.sort()
plt.figure(figsize=(12,6))
sns.countplot(x='sub_grade', data=chargeoffDF, hue='loan_status',order=chargeplot_order)
#%%
#Creating dummy for loan status
def paid_loan(status):
    if status =='Fully Paid':
        return 1
    else:
        return 0 
#%%
df['loan_repaid'] = df['loan_status'].apply(paid_loan)
df1 =df[['loan_repaid','loan_status']]
#%%
df
df1
df.corrwith(df1['loan_repaid']).sort_values().plot.bar()
#%%
#EDA is done. Time for Data Preprocessing before creating the model
#%%
len(df)
#%%
df.isna().sum()
#%%
df.isna().sum()/396030*100
#%%
feat_info('emp_title')
print('\n')
feat_info('emp_length')
#%%
df['emp_title'].value_counts()
#There are too many diferrent jobs to create dummys from. This will be droppped 
#%%
df = df.drop('emp_title',axis=1)
#%%
graphorder = ['< 1 year','1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10+ years']
plt.figure(figsize=(12,6))
sns.countplot(x='emp_length',data=df, order=graphorder)
#%%
plt.figure(figsize=(12,6))
sns.countplot(x='emp_length',data=df, order=graphorder, hue ='loan_status')
#%%

percent_by_emp = df[df['loan_status']=='Charged Off'].groupby('emp_length').count()['loan_status']/df.groupby('emp_length').count()['loan_status']
#%%
percent_be_emp2 = df[df['loan_status']=='Charged Off'].groupby('emp_length').count()['loan_status']/df[df['loan_status']=='Fully Paid'].groupby('emp_length').count()['loan_status']
#%%
percent_by_emp.plot.bar()
#Charge off rates are extremely similar across all emplyment rates. Lets drop this column
#%%
df.drop('emp_length',axis =1,inplace=True)
#%%
#How many more missing values do we have?
df.isna().sum()
#%%
df['title'].head()

df['purpose'].head()
# Title column is just a string representation of purpose column. We can drop
#%%
df.drop('title',axis=1,inplace=True)
#%%
# mort_acc has the most number of missing values. we need to work on this now
feat_info('mort_acc')
#%%
df['mort_acc'].value_counts()
#%%
df.corrwith(df['mort_acc']).sort_values()
# Total account correlates most with mortgage accounts
#%%
total_acc_avg = df.groupby('total_acc').mean()['mort_acc']
#%%
def fill_missing(total_acc,mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc
#%%
df['mort_acc'] = df.apply(lambda x: fill_missing(x['total_acc'], x['mort_acc']),axis=1)
#%%
df.isna().sum()
# Revolving utility and Bankruptcies have missing data but since they account for less than 0.5% of the data we can dropna
#%%
df = df.dropna()
#NO MORE MISSING DATA!
#%%
# Missing Data is done. NOw to work on Categorical String Values
list(df.select_dtypes(include='object'))
#Now that we have a list of the string features, we need to work through them 

#%%
df['term']=df['term'].apply(lambda term: int(term[:3]))
#%%
df['term']
#%%
#grade is already part of subgrade so we can drop it 
df.drop('grade',axis=1,inplace=True)
#%%
#Lets turn the subgrade into dummies
subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)
#%%
df.drop('sub_grade',axis=1,inplace=True)
#%%
df= pd.concat([df,subgrade_dummies],axis=1)
#%%
df.columns
#%%
df.select_dtypes('object').columns
#%%
application_dummies = pd.get_dummies(df[['verification_status','application_type','initial_list_status','purpose']],drop_first=True)
#%%
df=pd.concat([df,application_dummies],axis=1)
#%%
df.drop(['verification_status','application_type','initial_list_status','purpose'],axis=1,inplace=True)
#%%
df['home_ownership'].value_counts()
#%%
df['home_ownership'].replace(to_replace=['NONE','ANY'],value='OTHER',inplace=True)
#%%
home_own_dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
#%%
df=pd.concat([df,home_own_dummies],axis=1)
#%%
df['address']
#%%
df['zip_code']=df['address'].apply(lambda address: address[-6:])
#%%
zip_dummies=pd.get_dummies(df['zip_code'],drop_first=True)
#%%
df=pd.concat([df,zip_dummies],axis=1)
#%%
df.drop('home_ownership',axis=1,inplace=True)
#%%
df.drop('address',axis=1,inplace=True)
#%%
#We'll drop the issued feature becuase when using our model we wouldnt know beforehand whether or not a loan was issued
df.drop('issue_d',axis=1,inplace=True)
#%%
df.drop('zip_code',axis=1,inplace=True)
#%%
df['earliest_cr_line']
#Timestamp feature. lets get the year from this 
#%%
df['earliest_cr_year']=df['earliest_cr_line'].apply(lambda date: int(date[-4:]))
#%%
df.drop('earliest_cr_line',axis=1,inplace=True)
#%%
#Feature Engineering and Preprocessing is now DONE!. Time to actually create our model 
#%%

#%%
df.drop('loan_status',axis=1,inplace=True)
#%%
X= df.drop('loan_repaid',axis=1)
y=df['loan_repaid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=101) 
#%%
#Normalizing the data

scaler= MinMaxScaler()
#%%
X_train = scaler.fit_transform(X_train)
X_test =  scaler.transform(X_test)
#%%

#%%
#This is our model
model = Sequential()


model.add(Dense(units=78,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=38,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=19, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
#%%
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
#%%
model.fit(x=X_train, y=y_train, epochs=600,validation_data=(X_test, y_test), verbose=1,callbacks=[early_stop])
#%%
#Lets see how we did
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
#%%
predictions = (model.predict(X_test) > 0.5).astype("int32")
#%%

#%%
print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))
#%%
#Lets see if we can predict on a random customer
import random
random.seed(101)
random_ind = random.randint(0, len(df))

new_customer= df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer
#%%
(model.predict(new_customer.values.reshape(1,78)) >0.5).astype('int32')
#model predicts that customer paid. Did he?
#%%
df.iloc[random_ind]['loan_repaid']
#YES HE DID
#%%

model.save('LendingClubModel.h5')




