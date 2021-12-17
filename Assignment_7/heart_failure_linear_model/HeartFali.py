#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import seaborn as sn
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('./heart_failure_clinical_records_dataset.csv')
df = df[['creatinine_phosphokinase', 'serum_creatinine', 'serum_sodium', 'platelets', 'DEATH_EVENT']]
df


# # Question1

# In[3]:


df0 = df[df.DEATH_EVENT == 0]
df0Feature = df0[['creatinine_phosphokinase', 'serum_creatinine', 'serum_sodium', 'platelets']]
df1 = df[df.DEATH_EVENT == 1]
df1Feature = df1[['creatinine_phosphokinase', 'serum_creatinine', 'serum_sodium', 'platelets']]


# In[4]:


# corr matrix for death_event 0
sn.heatmap(df0Feature.corr(), annot=True)
plt.show()


# In[5]:


# corr matrix for death_event 1
sn.heatmap(df1Feature.corr(), annot=True)
plt.show()


# # Question2
# 
# Group3--> 
# X = serum sodium, Y = serum creatinine

# In[21]:


def residual(yTest, yPredict):
    temp = 0
    for (a, b) in zip(yTest, yPredict):
        temp += (a-b) * (a - b)
    return temp


# In[22]:


zeroSet = df0[[ 'serum_creatinine', 'serum_sodium']]
oneSet = df1[[ 'serum_creatinine', 'serum_sodium']]


# In[63]:


# for death_evetn = 0
# simple linear regression
x = zeroSet['serum_sodium']
y = zeroSet['serum_creatinine']
degree = 1

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=  0.5, random_state = 0)

weights = np.polyfit(xTrain, yTrain, degree)
model = np.poly1d(weights)
yPredict = model(xTest)
print(weights)
print(residual(yTest, yPredict))
plt.figure()
plt.scatter(xTest, yTest, color = 'green', label = 'True Data')
plt.scatter(xTest, yPredict, color = 'red', label = 'Predict')
plt.grid()
plt.legend()
plt.title('linear')

# quadratic
x = zeroSet['serum_sodium']
y = zeroSet['serum_creatinine']
degree = 2

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=  0.5, random_state = 0)

weights = np.polyfit(xTrain, yTrain, degree)
model = np.poly1d(weights)
yPredict = model(xTest)
print(weights)
print(residual(yTest, yPredict))
plt.figure()
plt.scatter(xTest, yTest, color = 'green', label = 'True Data')
plt.scatter(xTest, yPredict, color = 'red', label = 'Predict')
plt.grid()
plt.legend()
plt.title('quadratic')


# cubic
x = zeroSet['serum_sodium']
y = zeroSet['serum_creatinine']
degree = 3

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=  0.5, random_state = 0)

weights = np.polyfit(xTrain, yTrain, degree)
model = np.poly1d(weights)
yPredict = model(xTest)
print(weights)
print(residual(yTest, yPredict))
plt.figure()
plt.scatter(xTest, yTest, color = 'green', label = 'True Data')
plt.scatter(xTest, yPredict, color = 'red', label = 'Predict')
plt.grid()
plt.legend()
plt.title('cubic')

# GLM
x = zeroSet['serum_sodium']
x = np.log(x)
y = zeroSet['serum_creatinine']

degree = 1

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=  0.5, random_state = 0)

weights = np.polyfit(xTrain, yTrain, degree)
model = np.poly1d(weights)
yPredict = model(xTest)
print(weights)
print(residual(yTest, yPredict))
plt.figure()
plt.scatter(xTest, yTest, color = 'green', label = 'True Data')
plt.scatter(xTest, yPredict, color = 'red', label = 'Predict')
plt.grid()
plt.legend()
plt.title('y = alog(x) + b')

# GLM
x = zeroSet['serum_sodium']
x = np.log(x)
y = zeroSet['serum_creatinine']
y = np.log(y)

degree = 1

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=  0.5, random_state = 0)

weights = np.polyfit(xTrain, yTrain, degree)
model = np.poly1d(weights)
yPredict = model(xTest)
print(weights)
print(residual(np.exp(yTest), np.exp(yPredict)))
plt.figure()
plt.scatter(xTest, yTest, color = 'green', label = 'True Data')
plt.scatter(xTest, yPredict, color = 'red', label = 'Predict')
plt.grid()
plt.legend()
plt.title('log(y) = alog(x) + b')


# In[65]:


# for death_evetn = 1
# simple linear regression
x = oneSet['serum_sodium']
y = oneSet['serum_creatinine']
degree = 1

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=  0.5, random_state = 0)

weights = np.polyfit(xTrain, yTrain, degree)
model = np.poly1d(weights)
yPredict = model(xTest)
print(weights)
print(residual(yTest, yPredict))
plt.figure()
plt.scatter(xTest, yTest, color = 'green', label = 'True Data')
plt.scatter(xTest, yPredict, color = 'red', label = 'Predict')
plt.grid()
plt.legend()
plt.title('linear')


# quadratic
x = oneSet['serum_sodium']
y = oneSet['serum_creatinine']
degree = 2

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=  0.5, random_state = 0)

weights = np.polyfit(xTrain, yTrain, degree)
model = np.poly1d(weights)
yPredict = model(xTest)
print(weights)
print(residual(yTest, yPredict))
plt.figure()
plt.scatter(xTest, yTest, color = 'green', label = 'True Data')
plt.scatter(xTest, yPredict, color = 'red', label = 'Predict')
plt.grid()
plt.legend()
plt.title('quadratic')


# cubic
x = oneSet['serum_sodium']
y = oneSet['serum_creatinine']
degree = 3

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=  0.5, random_state = 0)

weights = np.polyfit(xTrain, yTrain, degree)
model = np.poly1d(weights)
yPredict = model(xTest)
print(weights)
print(residual(yTest, yPredict))
plt.figure()
plt.scatter(xTest, yTest, color = 'green', label = 'True Data')
plt.scatter(xTest, yPredict, color = 'red', label = 'Predict')
plt.grid()
plt.legend()
plt.title('cubic')

# GLM
x = oneSet['serum_sodium']
x = np.log(x)
y = oneSet['serum_creatinine']

degree = 1

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=  0.5, random_state = 0)

weights = np.polyfit(xTrain, yTrain, degree)
model = np.poly1d(weights)
yPredict = model(xTest)
print(weights)
print(residual(yTest, yPredict))
plt.figure()
plt.scatter(xTest, yTest, color = 'green', label = 'True Data')
plt.scatter(xTest, yPredict, color = 'red', label = 'Predict')
plt.grid()
plt.legend()
plt.title('y = alog(x) + b')

# GLM
x = oneSet['serum_sodium']
x = np.log(x)
y = oneSet['serum_creatinine']
y = np.log(y)

degree = 1

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=  0.5, random_state = 0)

weights = np.polyfit(xTrain, yTrain, degree)
model = np.poly1d(weights)
yPredict = model(xTest)
print(weights)
print(residual(np.exp(yTest), np.exp(yPredict)))
plt.figure()
plt.scatter(xTest, yTest, color = 'green', label = 'True Data')
plt.scatter(xTest, yPredict, color = 'red', label = 'Predict')
plt.grid()
plt.legend()
plt.title('log(y) = alog(x) + b')




# In[ ]:




