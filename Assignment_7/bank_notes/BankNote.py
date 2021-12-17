#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import sklearn.neighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.neighbors import NearestCentroid
import math
from sklearn.linear_model import LogisticRegression


# In[3]:


df = pd.read_csv("./data_banknote_authentication.csv")
feature = df[['variance', 'skewness', 'curtosis', 'entropy']]
label = df.iloc[:, 4]


# # Question4

# In[4]:


# optimal k = 7
knn = KNeighborsClassifier(n_neighbors=7)
# all 4 feature
x = feature
scaler.fit(x)
x = scaler.transform(x)
y = label

xTrain, xTest, yTrain, yTest = train_test_split(x, y, random_state=0, test_size=0.5)
knn.fit(xTrain, yTrain)
yPredict = knn.predict(xTest)
print(accuracy_score(yTest, yPredict))
# f1 miss
x = feature[['skewness', 'curtosis', 'entropy']]
scaler.fit(x)
x = scaler.transform(x)
y = label

xTrain, xTest, yTrain, yTest = train_test_split(x, y, random_state=0, test_size=0.5)
knn.fit(xTrain, yTrain)
yPredict = knn.predict(xTest)
print(accuracy_score(yTest, yPredict))

# f2 miss
x = feature[['variance','curtosis', 'entropy']]
scaler.fit(x)
x = scaler.transform(x)
y = label

xTrain, xTest, yTrain, yTest = train_test_split(x, y, random_state=0, test_size=0.5)
knn.fit(xTrain, yTrain)
yPredict = knn.predict(xTest)
print(accuracy_score(yTest, yPredict))

# f3 miss
x = feature[['variance', 'skewness', 'entropy']]
scaler.fit(x)
x = scaler.transform(x)
y = label

xTrain, xTest, yTrain, yTest = train_test_split(x, y, random_state=0, test_size=0.5)
knn.fit(xTrain, yTrain)
yPredict = knn.predict(xTest)
print(accuracy_score(yTest, yPredict))

# f4 miss
x = feature[['variance', 'skewness', 'curtosis', ]]
scaler.fit(x)
x = scaler.transform(x)
y = label

xTrain, xTest, yTrain, yTest = train_test_split(x, y, random_state=0, test_size=0.5)
knn.fit(xTrain, yTrain)
yPredict = knn.predict(xTest)
print(accuracy_score(yTest, yPredict))


# # Question5

# In[6]:


log = LogisticRegression()
x = feature
scaler.fit(x)
x = scaler.transform(x)
y = label


xTrain, xTest, yTrain, yTest = train_test_split(x, y, random_state=0, test_size=0.5)
log.fit(xTrain, yTrain)
yPredict = log.predict(xTest)
print(accuracy_score(yTest, yPredict))

## Confusion Matrix I choose 
# 0 is positive good 1 is negative
temp = confusion_matrix(yTest, yPredict)
print(temp)

tp = temp[0][0]
fp = temp[1][0]
tn = temp[1][1]
fn = temp[0][1]

tpr = tp / (tp + fn)
tnr = tn / (tn + fp)

print('TPR = {}, TNR = {}, tp fp tn fn = {} {} {} {}'.format(tpr, tnr, tp, fp, tn, fn))


# BU ID 1194
x = {'variance':[1], 'skewness':[1], 'curtosis':[9], 'entropy':[4]}
x = pd.DataFrame.from_dict(x)
x = scaler.transform(x)
yPredict = log.predict(x)
print(yPredict)


# # Question6

# In[7]:


log = LogisticRegression()
# all 4 feature
x = feature
scaler.fit(x)
x = scaler.transform(x)
y = label

xTrain, xTest, yTrain, yTest = train_test_split(x, y, random_state=0, test_size=0.5)
log.fit(xTrain, yTrain)
yPredict = log.predict(xTest)
print(accuracy_score(yTest, yPredict))
# f1 miss
x = feature[['skewness', 'curtosis', 'entropy']]
scaler.fit(x)
x = scaler.transform(x)
y = label

xTrain, xTest, yTrain, yTest = train_test_split(x, y, random_state=0, test_size=0.5)
log.fit(xTrain, yTrain)
yPredict = log.predict(xTest)
print(accuracy_score(yTest, yPredict))

# f2 miss
x = feature[['variance','curtosis', 'entropy']]
scaler.fit(x)
x = scaler.transform(x)
y = label

xTrain, xTest, yTrain, yTest = train_test_split(x, y, random_state=0, test_size=0.5)
log.fit(xTrain, yTrain)
yPredict = log.predict(xTest)
print(accuracy_score(yTest, yPredict))

# f3 miss
x = feature[['variance', 'skewness', 'entropy']]
scaler.fit(x)
x = scaler.transform(x)
y = label

xTrain, xTest, yTrain, yTest = train_test_split(x, y, random_state=0, test_size=0.5)
log.fit(xTrain, yTrain)
yPredict = log.predict(xTest)
print(accuracy_score(yTest, yPredict))

# f4 miss
x = feature[['variance', 'skewness', 'curtosis', ]]
scaler.fit(x)
x = scaler.transform(x)
y = label

xTrain, xTest, yTrain, yTest = train_test_split(x, y, random_state=0, test_size=0.5)
log.fit(xTrain, yTrain)
yPredict = log.predict(xTest)
print(accuracy_score(yTest, yPredict))


# In[ ]:




