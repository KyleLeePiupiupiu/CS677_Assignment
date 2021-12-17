#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# # Question1

# In[4]:


bk = pd.read_csv('./data_banknote_authentication.csv')
bk = bk.rename({'class':'Class'}, axis='columns')

color = []
for c in bk.Class:
    if c == 0:
        color.append('Green')
    else:
        color.append('red')

bk['Color'] = color


# In[3]:


bk0 = bk[bk.Class == 0]
bk1 = bk[bk.Class == 1]
print(bk0.describe())
print()
print(bk1.describe())
print()
print(bk.describe())


# # Question2

# In[10]:


# Split data and pairplot

## bk0
x0 = bk0[['variance', 'skewness', 'curtosis', 'entropy']]
y0 = bk0[['Class']]
x0Train, x0Test, y0Train, y0Test = train_test_split(x0, y0, test_size=0.5, random_state=0)

f0 = sns.pairplot(x0Train)
f0.fig.suptitle("class 0")

## bk1
x1 = bk1[['variance', 'skewness', 'curtosis', 'entropy']]
y1 = bk1[['Class']]
x1Train, x1Test, y1Train, y1Test = train_test_split(x1, y1, test_size=0.5, random_state=0)

f1 = sns.pairplot(x1Train)
f1.fig.suptitle("class 1")


# In[11]:


# easy model
f = plt.figure()
f.set_size_inches(12,24)

## variance
va = f.add_subplot(4,2,1)
a0 = x0Train.variance
a1 = x1Train.variance
va.plot(a0, np.zeros_like(a0) + 0, '.', color = 'green')
va.plot(a1, np.zeros_like(a1) + 0.1, '.', color = 'red')
va.set_title('variance')

vah = f.add_subplot(4,2,2)
vah.hist(a0, color='green')
vah.hist(a1, color = 'red', alpha=0.3)
vah.set_title('variance')

## skewness   
sk = f.add_subplot(4,2,3)
a0 = x0Train.skewness
a1 = x1Train.skewness
sk.plot(a0, np.zeros_like(a0) + 0, '.', color = 'green')
sk.plot(a1, np.zeros_like(a1) + 0.1, '.', color = 'red')
sk.set_title('skewness')

skh = f.add_subplot(4,2,4)
skh.hist(a0, color='green')
skh.hist(a1, color = 'red', alpha=0.3)
skh.set_title('skewness')

## curtosis    
cu = f.add_subplot(4,2,5)
a0 = x0Train.curtosis
a1 = x1Train.curtosis
cu.plot(a0, np.zeros_like(a0) + 0, '.', color = 'green')
cu.plot(a1, np.zeros_like(a1) + 0.1, '.', color = 'red')
cu.set_title('curtosis')

cuh = f.add_subplot(4,2,6)
cuh.hist(a0, color='green')
cuh.hist(a1, color = 'red', alpha=0.3)
cuh.set_title('curtosis')

## entropy   
en = f.add_subplot(4,2,7)
a0 = x0Train.entropy
a1 = x1Train.entropy
en.plot(a0, np.zeros_like(a0) + 0, '.', color = 'green')
en.plot(a1, np.zeros_like(a1) + 0.1, '.', color = 'red')
en.set_title('entropy')

enh = f.add_subplot(4,2,8)
enh.hist(a0, color='green')
enh.hist(a1, color = 'red', alpha=0.3)
enh.set_title('entropy')



# In[7]:


# Predict lable
x = bk[['variance', 'skewness', 'curtosis', 'entropy']]
y = bk[['Class']]
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.5, random_state=0)

yPredict = []
for v in xTest.variance:
    if v >= 0 :
        yPredict.append(0)
    else:
        yPredict.append(1)

# True False
tp = 0
tn = 0
fp = 0
fn = 0
acc = 0
for (p, t) in zip(yPredict, yTest.Class):
    if p == 0 and t == 0:
        tp += 1
    elif p == 1 and t == 1:
        tn += 1
    elif p == 0 and t == 1:
        fp += 1
    elif p == 1 and t == 0:
        fn += 1
    
    if p == t:
        acc = acc + 1

print("TP:{} FP:{} TN:{} FN:{} TPR:{} TNR:{} Accuracy:{}".format(tp, fp, tn, fn, tp/(tp + fn), tn/(tn + fp), acc / len(yPredict)))


# # Qestion3

# In[8]:


# KNN
kList = [3,5,7,9,11]
accuracy = []
for k in kList:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(xTrain, yTrain)
    yPredict = knn.predict(xTest)
    accuracy.append(accuracy_score(yTest, yPredict))
   
plt.plot(kList, accuracy)
print(accuracy)


# In[9]:


# k = 7 is optimal
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(xTrain, yTrain)
yPredict = knn.predict(xTest)
# True False
tp = 0
tn = 0
fp = 0
fn = 0
acc = 0
for (p, t) in zip(yPredict, yTest.Class):
    if p == 0 and t == 0:
        tp += 1
    elif p == 1 and t == 1:
        tn += 1
    elif p == 0 and t == 1:
        fp += 1
    elif p == 1 and t == 0:
        fn += 1
    
    if p == t:
        acc = acc + 1

print("TP:{} FP:{} TN:{} FN:{} TPR:{} TNR:{} Accuracy:{}".format(tp, fp, tn, fn, tp/(tp + fn), tn/(tn + fp), acc / len(yPredict)))


# In[45]:


# BU ID 64501194
# Take 1 1 9 4
x = {'variance':[1], 'skewness':[1], 'curtosis':[9], 'entropy':[4]}
x = pd.DataFrame.from_dict(x)
## my simple classifier
yPredict = 1
print("my simple classifier: {}".format(yPredict))
## for best knn
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(xTrain, yTrain)
yPredict = knn.predict(x)
print("knn(n=7): {}".format(yPredict))



# In[ ]:




