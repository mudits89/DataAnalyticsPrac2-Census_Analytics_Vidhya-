# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 19:53:16 2018

@author: mudit
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', None)

#when you want to see all the values in array (numpy)
np.set_printoptions(threshold=np.nan)

# Importing the dataset
path = "C:/Users/mudit/Desktop/Data Analytics/Census (Analytics Vidhya)/"
dataset = pd.read_csv(path + 'census-income.data', header = None )
dataset.iloc[:,:]

#To get unique values in all columns
for col in range(len(dataset.columns)):
    print("Unqiue value for " + str(col) + " are :")
    print(dataset[col].unique())
    print("\n\n")

#How to drop columns in a dataset
df = dataset.drop([31, 36, 38, 40],1)

#how to name the column headers
df.columns = ["AAGE", "ACLSWKR", "ADTIND", "ADTOCC", "AHGA", "AHRSPAY", "AHSCOL", 
              "AMARITL", "AMJIND", "AMJOCC","ARACE", "AREORGN", "ASEX", "AUNMEM",
              "AUNTYPE", "AWKSTAT", "CAPGAIN", "CAPLOSS", "DIVVAL" ,"FILESTAT", 
              "GRINREG", "GRINST", "HHDFMX", "HHDREL", "MARSUPWT", "MIGMTR1", 
              "MIGMTR3", "MIGMTR4","MIGSAME", "MIGSUN","PARENT","PEFNTVTY", 
              "PEMNTVTY", "PENATVTY", "PRCITSHP","SEOTR", "WKSWORK", "PTOTVAL"]

#getting value count for each column and appending to the list
lst = []
for col in df.columns:
    print(col)
    print(df[col].value_counts())
    lst.append(df[col].value_counts())

#Age plot
df['age_split'] = pd.cut(df['AAGE'], 4)    
fig, ax = plt.subplots()
df['age_split'].value_counts().plot(ax=ax, kind='bar')

#Class of worker plot
fig, ax = plt.subplots()
df['ACLSWKR'].value_counts().plot(ax=ax, kind='bar')

#Wage per hour plot
ct = df["AHRSPAY"] 
plt.boxplot(ct.values)
plt.show()

#Enrolled in college institute last week
fig, ax = plt.subplots()
df['AHSCOL'].value_counts().plot(ax=ax, kind='bar')

fig, ax = plt.subplots()
df['ADTOCC'].value_counts().plot(ax=ax, kind='bar')

fig, ax = plt.subplots()
df['ADTIND'].value_counts().plot(ax=ax, kind='bar')

fig, ax = plt.subplots()
df['AMARITL'].value_counts().plot(ax=ax, kind='bar')

fig, ax = plt.subplots()
df['AMJIND'].value_counts().plot(ax=ax, kind='bar')

fig, ax = plt.subplots()
df['AMJOCC'].value_counts().plot(ax=ax, kind='bar')

fig, ax = plt.subplots()
df['ARACE'].value_counts().plot(ax=ax, kind='bar')

fig, ax = plt.subplots()
df['AREORGN'].value_counts().plot(ax=ax, kind='bar')

fig, ax = plt.subplots()
df['ASEX'].value_counts().plot(ax=ax, kind='bar')

#correlation
df['AHSCOL'].corr(df['PTOTVAL'])
df['AAGE'].corr(df['PTOTVAL'])
df['AHRSPAY'].corr(df['PTOTVAL'])
df['AMARITL'].corr(df['PTOTVAL'])
df['ARACE'].corr(df['PTOTVAL'])
df['AREORGN'].corr(df['PTOTVAL'])
df['ASEX'].corr(df['PTOTVAL'])
df['AUNMEM'].corr(df['PTOTVAL'])
df['AUNTYPE'].corr(df['PTOTVAL'])
df['CAPGAIN'].corr(df['PTOTVAL'])
df['CAPLOSS'].corr(df['PTOTVAL'])
df['GRINREG'].corr(df['PTOTVAL'])
df['GRINST'].corr(df['PTOTVAL'])
df['HHDFMX'].corr(df['PTOTVAL'])
df['HHDREL'].corr(df['PTOTVAL'])
df['MARSUPWT'].corr(df['PTOTVAL'])
df['MIGMTR1'].corr(df['PTOTVAL'])
df['MIGMTR3'].corr(df['PTOTVAL'])
df['MIGMTR4'].corr(df['PTOTVAL'])
df['MIGSAME'].corr(df['PTOTVAL'])
df['MIGSUN'].corr(df['PTOTVAL'])
df['PARENT'].corr(df['PTOTVAL'])
df['PEFNTVTY'].corr(df['PTOTVAL'])
df['PEMNTVTY'].corr(df['PTOTVAL'])
df['PENATVTY'].corr(df['PTOTVAL'])
df['PRCITSHP'].corr(df['PTOTVAL'])
df['SEOTR'].corr(df['PTOTVAL'])
df['PTOTVAL'].corr(df['PTOTVAL'])
df['WKSWORK'].corr(df['PTOTVAL'])

#Education plot
fig, ax = plt.subplots()
df['AHGA'].value_counts().plot(ax=ax, kind='bar')

plt.matshow(df.corr())

df.corr()
temp_corr = df.corr().apply(lambda x: x.sort_values(ascending=False).head(10), axis=0)

# save in new variable
datset1 = df.copy()

# Encoding categorical data
# Encoding the Independent Variable
pd_dtypes = df.dtypes
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
for col in pd_dtypes.index:
    print (col)
    print (str(pd_dtypes[col]) + '\n')
    if pd_dtypes[col] in ["str","object"]:
        df.loc[:, col] = labelencoder_X.fit_transform(df.loc[:, col])
        
        
#Random Data Sampling          
train = df.sample(frac = 0.67, random_state=2)
test = df.loc[~df.index.isin(train.index), :]
test2 = test.copy()
#drop columns
train1 = train.drop(["age_split","wage_split"], 1)
test1 = test.drop(["age_split","wage_split", "PTOTVAL"], 1)


#drop columns for normalization of data
dataset1 = train1.copy()
dataset1 = dataset1.drop(["ADTIND","ADTOCC", "AHRSPAY", "CAPGAIN", "CAPLOSS", 
                          "DIVVAL", "HHDFMX", "MIGMTR1", "MIGMTR3", "MIGMTR4", 
                          "PEMNTVTY", "PENATVTY", "WKSWORK" ], 1)
dataset2 = test1.copy()
dataset2 = dataset2.drop(["ADTIND","ADTOCC", "AHRSPAY", "CAPGAIN", "CAPLOSS", 
                          "DIVVAL", "HHDFMX", "MIGMTR1", "MIGMTR3", "MIGMTR4", 
                          "PEMNTVTY", "PENATVTY", "WKSWORK" ], 1)

    # Fitting Logistic Regression
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

#matplotlib inline
pd.crosstab(df.AHRSPAY,df.PTOTVAL).plot(kind='bar')
plt.title('Frequency for Salary to Wages per hour')
plt.xlabel('AHRSPAY')
plt.ylabel('PTOTVAL')
plt.savefig('total_learning_wages_per_hour')

#RFE
data_final_vars=dataset1.columns.values.tolist()
X =dataset1.iloc[:,:-1]
y = dataset1.iloc[:, -1]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg)
rfe = rfe.fit(X, y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

#check P_value
cols = ['ACLSWKR', 'AHSCOL', 'AMARITL', 'ARACE','AREORGN','ASEX','AUNMEM', 
        'FILESTAT', 'MIGSAME','MIGSUN','PARENT']
import statsmodels.api as sm
X=dataset1[cols]
y=dataset1['PTOTVAL']
logit_model=sm.Logit(y, X)
result=logit_model.fit()
print(result.summary2())

cols = ['ACLSWKR', 'AHSCOL', 'AMARITL', 'ARACE','AREORGN','ASEX','AUNMEM', 
        'FILESTAT','MIGSUN','PARENT']
import statsmodels.api as sm
X=dataset1[cols]
y=dataset1['PTOTVAL']
logit_model=sm.Logit(y, X)
result=logit_model.fit()
print(result.summary2())

X_train = dataset1[cols]
y_train = dataset1['PTOTVAL']
X_test = dataset2[cols]
y_test = test2['PTOTVAL']

#logit
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

#y_predict
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'
      .format(logreg.score(X_test, y_test)))

#build confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
#number of occurences of each class
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


